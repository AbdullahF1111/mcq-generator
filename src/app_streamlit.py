import streamlit as st
import re
import random
import json
from typing import List, Dict, Any
import torch
from lmqg import TransformersQG
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
import tempfile
import os

# تكوين الصفحة
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="❓",
    layout="wide"
)

# تحميل النماذج مع التخزين المؤقت
@st.cache_resource
def load_models():
    """تحميل جميع النماذج المطلوبة"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("جاري تحميل النماذج... قد يستغرق بضع دقائق"):
        # نموذج توليد الأسئلة
        qg = TransformersQG(model="lmqg/t5-base-squad-qg")
        
        # نموذج الإجابات
        qa_pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            device=0 if device == "cuda" else -1
        )
        
        # نموذج التضمين
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # نموذج spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("يجب تحميل نموذج spacy أولاً. تشغيل: python -m spacy download en_core_web_sm")
            return None
            
        # نموذج توليد المشتتات
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if device == "cuda" else -1
        )
    
    return {
        'qg': qg,
        'qa_pipe': qa_pipe,
        'embedder': embedder,
        'nlp': nlp,
        'distractor_gen': distractor_gen,
        'device': device
    }

# تهيئة الثوابت
NUM_DISTRACTORS = 3
SIM_MIN = 0.20
SIM_MAX = 0.92
PAIRWISE_MAX = 0.86
QA_CONF_MIN = 0.05

BLACKLIST_WORDS = {"option", "list", "adjectives", "unknown", "true", "false", "thing", "stuff"}

# الدوال المساعدة
def clean_text_generated(txt: str) -> str:
    """تنظيف النص المولد"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = txt.replace("Ġ", " ").replace(" ", " ").strip()
    txt = re.sub(r'^[A-Z]\s*([A-Z][a-z]+)', r'\1', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def is_short_noun_phrase(text: str, max_tokens: int = 3) -> bool:
    """التحقق من أن النص عبارة عن جملة اسمية قصيرة"""
    if not text or len(text.split()) == 0:
        return False
    if len(text.split()) > max_tokens:
        return False
    
    doc = st.session_state.models['nlp'](text)
    if any(tok.pos_ == "VERB" for tok in doc):
        return False
    if not any(tok.pos_ in ("NOUN", "PROPN") for tok in doc):
        return False
    if re.fullmatch(r'[^A-Za-z0-9 ]+', text):
        return False
    return True

def detect_qtype(question: str) -> str:
    """تحديد نوع السؤال"""
    q = question.lower()
    if re.search(r'\bwhere\b', q):
        return "LOC"
    if re.search(r'\bwho\b|\bwhom\b', q):
        return "PERSON"
    if re.search(r'\bwhen\b|\byear\b|\b(month|day|morning|evening|summer|winter)\b', q):
        if re.search(r'what\s+.*\b(mountain|mountains|river|city|lake|island|park|trail|valley|coast|beach|state|country|village|town)\b', q):
            return "LOC"
        return "TIME"
    return "OTHER"

def extract_candidates_from_context(context: str) -> List[str]:
    """استخراج المرشحين من النص"""
    doc = st.session_state.models['nlp'](context)
    pool = set()
    
    for ent in doc.ents:
        txt = ent.text.strip()
        if 1 <= len(txt.split()) <= 4:
            pool.add(clean_text_generated(txt))
    
    for nc in doc.noun_chunks:
        txt = nc.text.strip()
        if 1 <= len(txt.split()) <= 4:
            pool.add(clean_text_generated(txt))
    
    years = re.findall(r'\b\d{4}\b', context)
    for y in years:
        pool.add(y)
    
    return list(pool)

def generate_distractors_by_lm(question: str, answer: str, num: int = 6) -> List[str]:
    """توليد مشتتات باستخدام النموذج اللغوي"""
    prompt = (f"Generate {num} short plausible distractors (1-3 words) for this question. "
              f"Question: {question} | Correct answer: {answer}. "
              f"Return a comma-separated list. Do NOT repeat the correct answer.")
    try:
        out = st.session_state.models['distractor_gen'](
            prompt, max_new_tokens=64, num_beams=4, do_sample=False, top_p=0.9, temperature=0.7
        )
        txt = out[0].get("generated_text", "")
        txt = clean_text_generated(txt)
        parts = [p.strip() for p in re.split(r'[,\n;]+', txt) if p.strip()]
        parts = [p for p in parts if len(p.split()) <= 4]
        return parts[:num]
    except Exception as e:
        st.error(f"خطأ في توليد المشتتات: {e}")
        return []

def semantic_select(answer: str, candidates: List[str], k: int = NUM_DISTRACTORS) -> List[str]:
    """اختيار المشتتات بناءً على التشابه الدلالي"""
    if not candidates:
        return []
    
    sbert = st.session_state.models['embedder']
    cand_emb = sbert.encode(candidates, convert_to_tensor=True)
    ans_emb = sbert.encode([answer], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
    
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
    """التحقق من الإجابة وتنظيفها"""
    try:
        res = st.session_state.models['qa_pipe'](question=question, context=context)
        pred = res.get("answer", "").strip()
        score = float(res.get("score", 0.0))
        pred = clean_text_generated(pred)
        if pred and score >= QA_CONF_MIN:
            return pred
    except Exception:
        pass
    
    if answer and re.search(re.escape(answer.strip()), context, flags=re.IGNORECASE):
        return answer.strip()
    return ""

def generate_mcqs_from_text(context: str, num_questions: int = 5, desired_distractors: int = NUM_DISTRACTORS) -> Dict[str, Any]:
    """الدالة الرئيسية لتوليد الأسئلة"""
    out = {"source_len": len(context.split()), "questions": []}
    
    try:
        qa_pairs_raw = st.session_state.models['qg'].generate_qa(context, num_questions=num_questions)
    except Exception as e:
        st.error(f"خطأ في توليد الأسئلة: {e}")
        return out
    
    # معالجة أزواج الأسئلة والإجابات
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
        
        if not q or len(q.split()) < 3 or q in seen_q:
            continue
        
        seen_q.add(q)
        qa_pairs.append((q, a))
    
    pool = extract_candidates_from_context(context)
    
    # معالجة كل سؤال
    for q, a in qa_pairs:
        trusted_answer = qa_answer_check_and_cleanup(q, context, a)
        if not trusted_answer:
            continue
        
        qtype = detect_qtype(q)
        lm_cands = generate_distractors_by_lm(q, trusted_answer, num=8)
        combined = list(dict.fromkeys(lm_cands + pool))
        
        # تصفية المرشحين
        filtered = []
        qnorm = re.sub(r'[^A-Za-z0-9 ]', '', q).lower()
        ansnorm = re.sub(r'[^A-Za-z0-9 ]', '', trusted_answer).lower()
        
        for c in combined:
            c_clean = clean_text_generated(c)
            if not c_clean or len(c_clean) > 30:
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
                if pc.lower() == trusted_answer.lower() or pc in distractors:
                    continue
                distractors.append(pc)
                if len(distractors) >= desired_distractors:
                    break
        
        # الخلط النهائي للخيارات
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

# واجهة Streamlit
def main():
    st.title("🧠 مولّد أسئلة الاختيار من متعدد")
    st.markdown("أدخل النص لإنشاء أسئلة اختيار من متعدد تلقائياً")
    
    # تحميل النماذج
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            st.stop()
        st.session_state.models = models
    
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header("الإعدادات")
        num_questions = st.slider("عدد الأسئلة", 1, 10, 5)
        num_distractors = st.slider("عدد المشتتات لكل سؤال", 2, 5, 3)
        
        st.header("نصوص مثاليه")
        example_texts = {
            "الساعات الميكانيكية": "The invention of the mechanical clock revolutionized timekeeping. Before clocks, people relied on sundials and water clocks. In the 14th century, European inventors created gears and weights to measure time more accurately.",
            "القراءة": "Reading books is one of the most powerful ways to gain knowledge and expand your imagination.",
            "الاحتباس الحراري": "Global warming is the long-term rise in Earth's average temperature. It is mainly caused by human activities such as burning fossil fuels which increase greenhouse gas concentrations."
        }
        
        selected_example = st.selectbox("اختر مثالاً", list(example_texts.keys()))
        if st.button("استخدام المثال"):
            st.session_state.input_text = example_texts[selected_example]
    
    # إدخال النص
    input_text = st.text_area(
        "أدخل النص هنا:",
        height=200,
        value=st.session_state.get('input_text', '')
    )
    
    # زر التوليد
    if st.button("🎯 توليد الأسئلة", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ الرجاء إدخال نص أولاً")
            return
        
        with st.spinner("جاري توليد الأسئلة... قد يستغرق بضع ثوان"):
            results = generate_mcqs_from_text(
                input_text, 
                num_questions=num_questions,
                desired_distractors=num_distractors
            )
        
        # عرض النتائج
        st.subheader(f"الأسئلة المولدة ({len(results['questions'])} سؤال)")
        
        if not results['questions']:
            st.info("⚠️ لم يتم توليد أي أسئلة. حاول بإدخال نص أطول أو أكثر تفصيلاً.")
            return
        
        for i, q in enumerate(results['questions'], 1):
            with st.container():
                st.markdown(f"### السؤال {i}")
                st.markdown(f"**{q['question']}**")
                
                # عرض الخيارات
                cols = st.columns(2)
                for j, option in enumerate(q['options']):
                    with cols[j % 2]:
                        is_correct = option == q['answer']
                        emoji = "✅" if is_correct else "⚪"
                        st.markdown(f"{emoji} **{chr(65+j)}.** {option}")
                
                with st.expander("معلومات إضافية"):
                    st.write(f"**نوع السؤال:** {q['qtype']}")
                    st.write(f"**الإجابة الصحيحة:** {q['answer']}")
                    st.write(f"**المعلومات:** {q['meta']}")
                
                st.divider()
        
        # خيارات التصدير
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 تصدير كـ JSON"):
                st.download_button(
                    label="تحميل JSON",
                    data=json.dumps(results, ensure_ascii=False, indent=2),
                    file_name="mcq_questions.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📋 نسخ النتائج"):
                output_text = ""
                for i, q in enumerate(results['questions'], 1):
                    output_text += f"{i}. {q['question']}\n"
                    for j, opt in enumerate(q['options']):
                        output_text += f"   {chr(65+j)}. {opt}\n"
                    output_text += f"   الإجابة: {q['answer']}\n\n"
                
                st.code(output_text)

if __name__ == "__main__":
    main()
