# ------------------------------------------------------------
# mcq_pipeline_final_v2.py
# Improved MCQ Generation Pipeline (clean & syntax-correct)
# ------------------------------------------------------------
# Requirements:
# pip install -q transformers sentence-transformers spacy pdfplumber python-docx keybert streamlit lmqg torch numpy==1.26.4
# python -m spacy download en_core_web_sm
# ------------------------------------------------------------

import re
import random
import json
from typing import List, Dict, Any
import torch
from lmqg import TransformersQG
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy

# -------------------------
# Config / device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DEVICE = 0 if DEVICE == "cuda" else -1

# -------------------------
# Load models
# -------------------------
print("🔄 Loading models... (this may take a while)")

# Embeddings (for semantic similarity)
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# QG model (Question Generation)
qg = TransformersQG(model="lmqg/t5-base-squad-qg")

# QA model
qa_pipe = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=PIPELINE_DEVICE,
)

# LM for distractors
distractor_gen = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=PIPELINE_DEVICE,
)

# spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Config thresholds
# -------------------------
NUM_DISTRACTORS = 3
SIM_MIN = 0.20
SIM_MAX = 0.92
PAIRWISE_MAX = 0.86
QA_CONF_MIN = 0.05  # minimal score from qa_pipe to accept span correction

BLACKLIST_WORDS = {
    "option", "list", "adjectives", "unknown",
    "true", "false", "thing", "stuff"
}

# -------------------------
# Utilities
# -------------------------
def clean_text_generated(txt: str) -> str:
    """Clean model-generated text from artifacts."""
    if not txt:
        return ""
    txt = txt.strip().replace("Ġ", " ").replace("▁", " ")
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"^[A-Z]\s*([A-Z][a-z]+)", r"\1", txt)
    return txt.strip()


def is_short_noun_phrase(text: str, max_tokens: int = 3) -> bool:
    """Return True if text is noun phrase <= max_tokens."""
    if not text or len(text.split()) > max_tokens:
        return False
    doc = nlp(text)
    if any(tok.pos_ == "VERB" for tok in doc):
        return False
    if not any(tok.pos_ in ("NOUN", "PROPN") for tok in doc):
        return False
    if re.fullmatch(r"[^A-Za-z0-9 ]+", text):
        return False
    return True


def detect_qtype(question: str) -> str:
    q = question.lower()
    if re.search(r"\bwhere\b", q):
        return "LOC"
    if re.search(r"\bwho\b|\bwhom\b", q):
        return "PERSON"
    if re.search(r"\bwhen\b|\byear\b|\b(month|day|morning|evening|summer|winter)\b", q):
        if re.search(
            r"what\s+.*\b(mountain|river|city|lake|island|country|village|town)\b",
            q,
        ):
            return "LOC"
        return "TIME"
    return "OTHER"


def extract_candidates_from_context(context: str) -> List[str]:
    """Extract named entities and noun chunks."""
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
    years = re.findall(r"\b\d{4}\b", context)
    pool.update(years)
    return list(pool)


def generate_distractors_by_lm(question: str, answer: str, num: int = 6) -> List[str]:
    """Generate distractors via text generation LM."""
    prompt = (
        f"Generate {num} short plausible distractors (1–3 words) for this question. "
        f"Question: {question} | Correct answer: {answer}. "
        "Return a comma-separated list. Do NOT repeat the correct answer."
    )
    try:
        out = distractor_gen(
            prompt,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
            top_p=0.9,
            temperature=0.7,
        )
        txt = out[0].get("generated_text", "")
        txt = clean_text_generated(txt)
        parts = [p.strip() for p in re.split(r"[,;\n]+", txt) if p.strip()]
        parts = [p for p in parts if len(p.split()) <= 4]
        return parts[:num]
    except Exception as e:
        print("⚠️ Distractor LM error:", e)
        return []


def semantic_select(answer: str, candidates: List[str], k: int = NUM_DISTRACTORS) -> List[str]:
    """Select k semantically close and diverse distractors."""
    if not candidates:
        return []
    cand_emb = sbert.encode(candidates, convert_to_tensor=True)
    ans_emb = sbert.encode([answer], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
    idxs = [i for i, s in enumerate(sims) if SIM_MIN <= s <= SIM_MAX]
    if not idxs:
        idxs = sorted(range(len(candidates)), key=lambda i: -sims[i])[: k * 4]
    else:
        idxs = sorted(idxs, key=lambda i: -sims[i])
    selected = []
    for i in idxs:
        emb_i = cand_emb[i]
        if all(util.pytorch_cos_sim(emb_i, cand_emb[j]).item() <= PAIRWISE_MAX for j in selected):
            selected.append(i)
        if len(selected) >= k:
            break
    return [candidates[i] for i in selected]


def qa_answer_check_and_cleanup(question: str, context: str, answer: str) -> str:
    """Use QA model to verify answer correctness in context."""
    try:
        res = qa_pipe(question=question, context=context)
        pred = clean_text_generated(res.get("answer", "").strip())
        score = float(res.get("score", 0.0))
        if pred and score >= QA_CONF_MIN:
            return pred
    except Exception:
        pass
    if answer and re.search(re.escape(answer.strip()), context, flags=re.IGNORECASE):
        return answer.strip()
    return ""


# -------------------------
# Main MCQ pipeline
# -------------------------
def generate_mcqs_from_text(
    context: str,
    num_questions: int = 5,
    desired_distractors: int = NUM_DISTRACTORS,
    verbose: bool = False,
) -> Dict[str, Any]:
    out = {"source_len": len(context.split()), "questions": []}

    # Step 1: Generate QA pairs
    try:
        qa_pairs_raw = qg.generate_qa(context, num_questions=num_questions)
    except Exception as e:
        print("⚠️ QG error:", e)
        qa_pairs_raw = []

    # Normalize QA pairs
    qa_pairs = []
    seen_q = set()
    for rec in qa_pairs_raw:
        if isinstance(rec, dict):
            q = clean_text_generated(rec.get("question", ""))
            a = clean_text_generated(rec.get("answer", ""))
        elif isinstance(rec, (list, tuple)) and len(rec) == 2:
            q, a = clean_text_generated(rec[0]), clean_text_generated(rec[1])
        else:
            continue
        if not q or len(q.split()) < 3 or q in seen_q:
            continue
        seen_q.add(q)
        qa_pairs.append((q, a))

    if verbose:
        print("✅ QA pairs:", qa_pairs)

    pool = extract_candidates_from_context(context)

    # Step 2: Process each pair
    for q, a in qa_pairs:
        trusted_answer = qa_answer_check_and_cleanup(q, context, a)
        if not trusted_answer:
            if verbose:
                print("⛔ Rejected QA pair:", q)
            continue

        qtype = detect_qtype(q)
        lm_cands = generate_distractors_by_lm(q, trusted_answer, num=8)
        combined = list(dict.fromkeys(lm_cands + pool))

        filtered = []
        qnorm = re.sub(r"[^A-Za-z0-9 ]", "", q).lower()
        ansnorm = re.sub(r"[^A-Za-z0-9 ]", "", trusted_answer).lower()
        for c in combined:
            c_clean = clean_text_generated(c)
            if (
                not c_clean
                or len(c_clean) > 30
                or any(b in c_clean.lower() for b in BLACKLIST_WORDS)
                or (ansnorm and ansnorm in c_clean.lower())
            ):
                continue
            overlap = sum(1 for w in c_clean.lower().split() if w in qnorm.split()) / max(1, len(c_clean.split()))
            if overlap > 0.6 or not is_short_noun_phrase(c_clean, max_tokens=3):
                continue
            filtered.append(c_clean)

        distractors = semantic_select(trusted_answer, filtered, k=desired_distractors)

        # Add more if needed
        if len(distractors) < desired_distractors:
            pool_candidates = [p for p in pool if is_short_noun_phrase(p)]
            for pc in pool_candidates:
                if pc.lower() != trusted_answer.lower() and pc not in distractors:
                    distractors.append(pc)
                if len(distractors) >= desired_distractors:
                    break

        distractors = distractors[:desired_distractors]
        options = distractors + [trusted_answer]
        random.shuffle(options)

        out["questions"].append(
            {
                "question": q,
                "answer": trusted_answer,
                "options": options,
                "qtype": qtype,
                "meta": {"pool_used": len(pool), "lm_suggested": len(lm_cands)},
            }
        )

    return out


# -------------------------
# Test run
# -------------------------
if __name__ == "__main__":
    sample_text = (
        "Albert Einstein was a theoretical physicist who developed the theory of relativity. "
        "He was born in 1879 in Ulm, Germany. The speed of light is a central constant in physics."
    )
    result = generate_mcqs_from_text(sample_text, num_questions=3, verbose=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
