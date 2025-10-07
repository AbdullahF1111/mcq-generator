# Improved MCQ pipeline v2
# Save as mcq_pipeline_final_v2.py and run in Colab / local env.

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback if not preinstalled
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # ✅ Safe CPU-friendly load
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # ✅ Lightweight question generator
    qg = TransformersQG(language="en", model="lmqg/t5-small-squad-qg")

    # ✅ Use CPU for QA and distractor generation
    qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
    distractor_gen = pipeline("text-generation", model="gpt2", device=-1)

    return nlp, embedder, qg, qa_pipe, distractor_gen
# -------------------------
# Config / device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DEVICE = 0 if DEVICE == "cuda" else -1

# -------------------------
# Load models (may take time)
# -------------------------
print("Loading models... (may take a while)")
# Embeddings (لتقييم التشابه)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
 #QG model (سؤال + جواب)
qg = TransformersQG(model="lmqg/t5-base-squad-qg")   # QG + AE

qa_pipe = pipeline("question-answering",
                   model="deepset/roberta-base-squad2",
                   tokenizer="deepset/roberta-base-squad2",
                   device=PIPELINE_DEVICE)
# fallback generator for distractors (deterministic-ish)
distractor_gen = pipeline("text2text-generation",
                          model="google/flan-t5-base",
                          device=PIPELINE_DEVICE)
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Utilities / thresholds
# -------------------------
NUM_DISTRACTORS = 3
SIM_MIN = 0.20
SIM_MAX = 0.92
PAIRWISE_MAX = 0.86
QA_CONF_MIN = 0.05  # minimal score from qa_pipe to accept span correction

BLACKLIST_WORDS = {"option", "list", "adjectives", "unknown", "true", "false", "thing", "stuff"}


def clean_text_generated(txt: str) -> str:
    #Basic cleanup of LM outputs: strip, remove weird leading tokens like 'TGlobal' or 'Ġ' chars.
    if not txt:
        return ""
    # remove weird non-alphanumeric prefixes
    txt = txt.strip()
    # fix common tokenizer artifacts
    txt = txt.replace("Ġ", " ").replace("▁", " ").strip()
    # remove stray leading tokens like 'TGlobal' if starts with single letter + uppercase word (heuristic)
    txt = re.sub(r'^[A-Z]\s*([A-Z][a-z]+)', r'', txt)  # 'TGlobal' -> 'Global'
    # remove multiple spaces
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def is_short_noun_phrase(text: str, max_tokens: int = 3) -> bool:
    #Return True if text is noun phrase of <= max_tokens and not a sentence.
    if not text or len(text.split()) == 0:
        return False
    if len(text.split()) > max_tokens:
        return False
    doc = nlp(text)
    # reject if contains a verb
    if any(tok.pos_ == "VERB" for tok in doc):
        return False
    # require at least one NOUN or PROPN
    if not any(tok.pos_ in ("NOUN", "PROPN") for tok in doc):
        return False
    # reject if punctuation heavy
    if re.fullmatch(r'[^A-Za-z0-9 ]+', text):
        return False
    return True


def detect_qtype(question: str) -> str:
    q = question.lower()
    if re.search(r'where', q):
        return "LOC"
    if re.search(r'who|whom', q):
        return "PERSON"
    if re.search(r'when|year|(month|day|morning|evening|summer|winter)', q):
        # special-case: 'what ... mountains' should be LOC
        if re.search(r'what\s+.*\b(mountain|mountains|river|city|lake|island|park|trail|valley|coast|beach|state|country|village|town)\b', q):
            return "LOC"
        return "TIME"
    # fallback
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
    # add years/numbers found
    years = re.findall(r'\d{4}', context)
    for y in years:
        pool.add(y)
    return list(pool)


def generate_distractors_by_lm(question: str, answer: str, num: int = 6) -> List[str]:
    #Prompt LM to produce comma-separated short distractors. Deterministic-ish.
    prompt = (f"Generate {num} short plausible distractors (1-3 words) for this question. "
              f"Question: {question} | Correct answer: {answer}. "
              f"Return a comma-separated list. Do NOT repeat the correct answer.")
    try:
        out = distractor_gen(prompt, max_new_tokens=64, num_beams=4, do_sample=False, top_p=0.9, temperature=0.7)
        txt = out[0].get("generated_text", "")
        txt = clean_text_generated(txt)
        parts = [p.strip() for p in re.split(r'[,;\n]+', txt) if p.strip()]
        # filter very long parts
        parts = [p for p in parts if len(p.split()) <= 4]
        return parts[:num]
    except Exception as e:
        print("distractor LM error:", e)
        return []


def semantic_select(answer: str, candidates: List[str], k: int = NUM_DISTRACTORS) -> List[str]:
    #Select k candidates semantically close to answer but diverse among themselves.
    if not candidates:
        return []
    cand_emb = sbert.encode(candidates, convert_to_tensor=True)
    ans_emb = sbert.encode([answer], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
    # pick indices within SIM_MIN..SIM_MAX
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
    """
    Use qa_pipe to extract trusted span for the question from context.
    If QA returns a confident non-empty answer, prefer it (cleaned).
    Otherwise, if provided 'answer' appears verbatim in context, keep it.
    Else return empty string (meaning reject this pair).
    """
    try:
        res = qa_pipe(question=question, context=context)
        pred = res.get("answer", "").strip()
        score = float(res.get("score", 0.0))
        pred = clean_text_generated(pred)
        if pred and score >= QA_CONF_MIN:
            return pred
    except Exception:
        pass
    # fallback: check if original answer occurs verbatim in context (case-insensitive)
    if answer and re.search(re.escape(answer.strip()), context, flags=re.IGNORECASE):
        return answer.strip()
    return ""


# -------------------------
# Main pipeline function
# -------------------------
def generate_mcqs_from_text(context: str, num_questions: int = 5, desired_distractors: int = NUM_DISTRACTORS,
                            verbose: bool = False) -> Dict[str, Any]:
    out = {"source_len": len(context.split()), "questions": []}
    # 1) QG generation (may produce pairs)
    try:
        qa_pairs_raw = qg.generate_qa(context, num_questions=num_questions)
    except Exception as e:
        print("QG error:", e)
        qa_pairs_raw = []

    # convert to uniform (question, answer)
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
        # basic sanity filters
        if not q or len(q.split()) < 3:
            continue
        if q in seen_q:
            continue
        seen_q.add(q)
        qa_pairs.append((q, a))

    if verbose:
        print("QA pairs (cleaned):", qa_pairs)

    pool = extract_candidates_from_context(context)

    # process each pair
    for q, a in qa_pairs:
        # prefer QA-derived answer span if possible
        trusted_answer = qa_answer_check_and_cleanup(q, context, a)
        if not trusted_answer:
            if verbose:
                print("Reject QA pair (answer not supported in context):", q, a)
            continue
        # finalize qtype
        qtype = detect_qtype(q)

        # build candidate pool for distractors: combine context pool + LM suggestions
        lm_cands = generate_distractors_by_lm(q, trusted_answer, num=8)
        combined = list(dict.fromkeys(lm_cands + pool))  # preserve order, dedupe

        # filter candidates: only accept short noun phrases and not equal/substring of answer or question
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
            # remove if same as answer or contains answer
            if ansnorm and ansnorm in c_clean.lower():
                continue
            # remove if token overlap too high with question (avoid repeating question words)
            if sum(1 for w in c_clean.lower().split() if w in qnorm.split()) / max(1, len(c_clean.split())) > 0.6:
                continue
            # must be short noun phrase
            if not is_short_noun_phrase(c_clean, max_tokens=3):
                continue
            filtered.append(c_clean)

        # semantic select
        distractors = semantic_select(trusted_answer, filtered, k=desired_distractors)

        # QA sanity: avoid distractor that QA returns as answer for the question
        try:
            # if QA returns same as distractor for this question, drop it
            safe = []
            for d in distractors:
                try:
                    res = qa_pipe(question=q, context=context)
                    pred = clean_text_generated(res.get("answer", "")).lower()
                except Exception:
                    pred = ""
                if pred and d.lower() in pred:
                    # skip distractor that appears in QA predicted span
                    continue
                safe.append(d)
            distractors = safe
        except Exception:
            pass

        # if not enough distractors, add fillers from pool or fallback LM (and re-clean)
        if len(distractors) < desired_distractors:
            # try pool
            pool_candidates = [p for p in pool if is_short_noun_phrase(p)]
            for pc in pool_candidates:
                if pc.lower() == trusted_answer.lower(): continue
                if pc in distractors: continue
                distractors.append(pc)
                if len(distractors) >= desired_distractors:
                    break

        if len(distractors) < desired_distractors:
            # ask LM again but force do_sample=False for consistency
            extra = generate_distractors_by_lm(q, trusted_answer, num=6)
            for e in extra:
                if e.lower() == trusted_answer.lower(): continue
                if not is_short_noun_phrase(e): continue
                if e in distractors: continue
                distractors.append(e)
                if len(distractors) >= desired_distractors:
                    break

        # final trim
        distractors = distractors[:desired_distractors]
        # shuffle options but keep answer known
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


# -------------------------
# Example usage (main)
# -------------------------
if __name__ == "__main__":
    examples = {
        "clocks": (
            "The invention of the mechanical clock revolutionized timekeeping. Before clocks, "
            "people relied on sundials and water clocks. In the 14th century, European inventors "
            "created gears and weights to measure time more accurately. This allowed societies to "
            "organize work, prayer, and travel more efficiently. Today, atomic clocks are the most "
            "precise, used in GPS and scientific research."
        ),
        "books": (
            "Reading books is one of the most powerful ways to gain knowledge and expand your imagination. "
        ),
        "liam": (
            "Liam and his friends hiked through the Rocky Mountains last summer. They camped by a clear lake "
            "and watched the sunrise one morning. Samuel Harrison promised himself to return every year. The trip "
            "strengthened their friendship and inspired Liam to study nature."
        ),
        "climate": (
            "Global warming is the long-term rise in Earth's average temperature. It is mainly caused by human "
            "activities such as burning fossil fuels which increase greenhouse gas concentrations in the atmosphere. "
            "Climate scientists study temperature records, ice cores and other indicators to understand the rate of change."
        )
    }

    results_all = {}
    for name, text in examples.items():
        print("Processing example:", name)
        res = generate_mcqs_from_text(text, num_questions=5, desired_distractors=3, verbose=True)
        results_all[name] = res
        for i, q in enumerate(res["questions"], 1):
            print(f"{i}. {q['question']}")
            for j, opt in enumerate(q["options"]):
                print(f"  {chr(65+j)}. {opt}")
            print("  --> Correct:", q["answer"])
            print("  meta:", q["meta"])

    with open("mcq_output_final_v2.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print("Saved mcq_output_final_v2.json")

