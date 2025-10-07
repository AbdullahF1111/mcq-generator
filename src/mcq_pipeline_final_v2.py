# mcq_pipeline_final_v2.py
import re
import json
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import torch
from lmqg import TransformersQG
from typing import List, Dict, Any
from sentence_transformers import util
import random
import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# تحميل النماذج بشكل آمن مع Streamlit caching
@st.cache_resource(show_spinner=False)
def load_models():
    """تحميل جميع النماذج مرة واحدة مع معالجة الأخطاء"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # إذا لم يكن النموذج مثبتاً
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # استخدام CPU لتجنب مشاكل الذاكرة
    device = "cpu"
    
    try:
        # تحميل النماذج بشكل منفصل مع معالجة الأخطاء
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        # بديل أخف
        embedder = SentenceTransformer("paraphrase-albert-small-v2", device=device)
    
    try:
        # استخدام نموذج أخف للسؤال والإجابة
        qg = TransformersQG(language="en", model="lmqg/t5-small-squad-qg")
    except Exception as e:
        print(f"Error loading QG model: {e}")
        qg = None
    
    try:
        # استخدام نماذج أخف
        qa_pipe = pipeline(
            "question-answering", 
            model="distilbert-base-uncased-distilled-squad", 
            device=-1  # استخدام CPU
        )
    except Exception as e:
        print(f"Error loading QA model: {e}")
        qa_pipe = None
    
    try:
        # استخدام نموذج أخف لتوليد المشتتات
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1  # استخدام CPU
        )
    except Exception as e:
        print(f"Error loading distractor model: {e}")
        distractor_gen = None
    
    return nlp, embedder, qg, qa_pipe, distractor_gen

# تحميل النماذج مرة واحدة
nlp, embedder, qg, qa_pipe, distractor_gen = load_models()

# -------------------------
# باقي الدوال بدون تغيير
# -------------------------

def clean_text_generated(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    txt = txt.replace("Ġ", " ").replace("▁", " ").strip()
    txt = re.sub(r'^[A-Z]\s*([A-Z][a-z]+)', r'\1', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def is_short_noun_phrase(text: str, max_tokens: int = 3) -> bool:
    if not text or len(text.split()) == 0:
        return False
    if len(text.split()) > max_tokens:
        return False
    doc = nlp(text)
    if any(tok.pos_ == "VERB" for tok in doc):
        return False
    if not any(tok.pos_ in ("NOUN", "PROPN") for tok in doc):
        return False
    if re.fullmatch(r'[^A-Za-z0-9 ]+', text):
        return False
    return True

def detect_qtype(question: str) -> str:
    q = question.lower()
    if re.search(r'where', q):
        return "LOC"
    if re.search(r'who|whom', q):
        return "PERSON"
    if re.search(r'when|year|month|day|morning|evening|summer|winter', q):
        if re.search(r'what\s+.*\b(mountain|mountains|river|city|lake|island|park|trail|valley|coast|beach|state|country|village|town)\b', q):
            return "LOC"
        return "TIME"
    return "OTHER"

def extract_candidates_from_context(context: str) -> List[str]:
    doc = nlp(context)
    pool = set()
    for ent in doc.ents:
        txt = ent.text.strip()
        if 1 <= len(txt.split()) <= 4:
            pool.add(clean_text_generated(txt))
    for nc in doc.noun_chunks:
        txt = nc.text.strip()
        if 1 <= len(txt.split()) <= 4:
            pool.add(clean_text_generated(txt))
    years = re.findall(r'\d{4}', context)
    for y in years:
        pool.add(y)
    return list(pool)

def generate_distractors_by_lm(question: str, answer: str, num: int = 6) -> List[str]:
    if distractor_gen is None:
        return []
    
    prompt = (f"Generate {num} short plausible distractors (1-3 words) for this question. "
              f"Question: {question} | Correct answer: {answer}. "
              f"Return a comma-separated list. Do NOT repeat the correct answer.")
    try:
        out = distractor_gen(prompt, max_new_tokens=64, num_beams=4, do_sample=False)
        txt = out[0].get("generated_text", "")
        txt = clean_text_generated(txt)
        parts = [p.strip() for p in re.split(r'[,;\n]+', txt) if p.strip()]
        parts = [p for p in parts if len(p.split()) <= 4]
        return parts[:num]
    except Exception as e:
        print("distractor LM error:", e)
        return []

def semantic_select(answer: str, candidates: List[str], k: int = 3) -> List[str]:
    if not candidates or embedder is None:
        return []
    
    cand_emb = embedder.encode(candidates, convert_to_tensor=True)
    ans_emb = embedder.encode([answer], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
    
    SIM_MIN = 0.20
    SIM_MAX = 0.92
    PAIRWISE_MAX = 0.86
    
    idxs = [i for i, s in enumerate(sims) if SIM_MIN <= s <= SIM_MAX]
    if not idxs:
        idxs = sorted(range(len(candidates)), key=lambda i: -sims[i])[:min(len(candidates), k * 4)]
    else:
        idxs = sorted(idxs, key=lambda i: -sims[i])
    
    selected = []
    for i in idxs:
        emb_i = cand_emb[i]
        ok = True
        for j in selected:
            if util.pytorch_cos_sim(emb_i, cand_emb[j]).item() > PAIRWISE_MAX:
                ok = False
                break
        if ok:
            selected.append(i)
        if len(selected) >= k:
            break
    return [candidates[i] for i in selected]

def qa_answer_check_and_cleanup(question: str, context: str, answer: str) -> str:
    if qa_pipe is None:
        # fallback إذا لم يتم تحميل النموذج
        if answer and re.search(re.escape(answer.strip()), context, flags=re.IGNORECASE):
            return answer.strip()
        return ""
    
    try:
        res = qa_pipe(question=question, context=context)
        pred = res.get("answer", "").strip()
        score = float(res.get("score", 0.0))
        pred = clean_text_generated(pred)
        if pred and score >= 0.05:
            return pred
    except Exception:
        pass
    
    if answer and re.search(re.escape(answer.strip()), context, flags=re.IGNORECASE):
        return answer.strip()
    return ""

def generate_mcqs_from_text(context: str, num_questions: int = 5, desired_distractors: int = 3, verbose: bool = False) -> Dict[str, Any]:
    out = {"source_len": len(context.split()), "questions": []}
    
    if qg is None:
        print("QG model not available")
        return out
    
    try:
        qa_pairs_raw = qg.generate_qa(context, num_questions=num_questions)
    except Exception as e:
        print("QG error:", e)
        qa_pairs_raw = []

    qa_pairs = []
    seen_q = set()
    for rec in qa_pairs_raw:
        if isinstance(rec, dict):
            q = clean_text_generated(rec.get("question", ""))
            a = clean_text_generated(rec.get("answer", ""))
        elif isinstance(rec, (list, tuple)) and len(rec) == 2:
            q = clean_text_generated(rec[0])
            a = clean_text_generated(rec[1])
        else:
            continue
        
        if not q or len(q.split()) < 3:
            continue
        if q in seen_q:
            continue
        seen_q.add(q)
        qa_pairs.append((q, a))

    if verbose:
        print("QA pairs (cleaned):", qa_pairs)

    pool = extract_candidates_from_context(context)
    BLACKLIST_WORDS = {"option", "list", "adjectives", "unknown", "true", "false", "thing", "stuff"}

    for q, a in qa_pairs:
        trusted_answer = qa_answer_check_and_cleanup(q, context, a)
        if not trusted_answer:
            if verbose:
                print("Reject QA pair (answer not supported in context):", q, a)
            continue
        
        qtype = detect_qtype(q)
        lm_cands = generate_distractors_by_lm(q, trusted_answer, num=8)
        combined = list(dict.fromkeys(lm_cands + pool))

        filtered = []
        qnorm = re.sub(r'[^A-Za-z0-9 ]', '', q).lower()
        ansnorm = re.sub(r'[^A-Za-z0-9 ]', '', trusted_answer).lower()
        
        for c in combined:
            c_clean = clean_text_generated(c)
            if not c_clean:
                continue
            if len(c_clean) > 30:
                continue
            if any(b in c_clean.lower() for b in BLACKLIST_WORDS):
                continue
            if ansnorm and ansnorm in c_clean.lower():
                continue
            if sum(1 for w in c_clean.lower().split() if w in qnorm.split()) / max(1, len(c_clean.split())) > 0.6:
                continue
            if not is_short_noun_phrase(c_clean, max_tokens=3):
                continue
            filtered.append(c_clean)

        distractors = semantic_select(trusted_answer, filtered, k=desired_distractors)

        # إذا لم يكن هناك مشتتات كافية
        if len(distractors) < desired_distractors:
            pool_candidates = [p for p in pool if is_short_noun_phrase(p)]
            for pc in pool_candidates:
                if pc.lower() == trusted_answer.lower(): 
                    continue
                if pc in distractors: 
                    continue
                distractors.append(pc)
                if len(distractors) >= desired_distractors:
                    break

        distractors = distractors[:desired_distractors]
        options = distractors + [trusted_answer]
        random.shuffle(options)

        out["questions"].append({
            "question": q,
            "answer": trusted_answer,
            "options": options,
            "qtype": qtype,
            "meta": {"pool_used": len(pool), "lm_suggested": len(lm_cands)}
        })

    return out

# باقي الكود بدون تغيير...
