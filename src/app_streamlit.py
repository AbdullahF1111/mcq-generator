import streamlit as st
import re
import random
import json
from typing import List, Dict, Any
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

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
        try:
            # محاولة تحميل spacy - إذا فشل نستخدم بديل
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                spacy_available = True
            except:
                st.warning("⚠️ لم يتم تحميل Spacy، سيتم استخدام معالجة نصية مبسطة")
                nlp = None
                spacy_available = False
            
            # نموذج توليد الأسئلة باستخدام pipeline مباشرة
            qg_pipe = pipeline(
                "text2text-generation",
                model="mrm8488/t5-base-finetuned-question-generation-ap",
                device=0 if device == "cuda" else -1
            )
            
            # نموذج الإجابات
            qa_pipe = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if device == "cuda" else -1
            )
            
            # نموذج التضمين
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
            # نموذج توليد المشتتات
            distractor_gen = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=0 if device == "cuda" else -1
            )
        
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {e}")
            return None
    
    return {
        'qg_pipe': qg_pipe,
        'qa_pipe': qa_pipe,
        'embedder': embedder,
        'nlp': nlp,
        'spacy_available': spacy_available,
        'distractor_gen': distractor_gen,
        'device': device
    }

# الدوال المساعدة
def clean_text_generated(txt: str) -> str:
    """تنظيف النص المولد"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def extract_candidates_simple(context: str) -> List[str]:
    """استخراج مرشحين بدون استخدام spacy"""
    # استخراج كلمات كبيرة (أسماء) باستخدام regex بسيط
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
    # استخراج أرقام سنوات
    years = re.findall(r'\b\d{4}\b', context)
    # استخراج كلمات شائعة بعد "the", "a", "an"
    nouns = re.findall(r'\b(?:the|a|an)\s+(\w+\s\w+|\w+)', context.lower())
    nouns = [n.title() for n in nouns]
    
    candidates = list(set(words + years + nouns))
    return [c for c in candidates if 1 <= len(c.split()) <= 3]

def generate_qa_pairs(context: str, num_questions: int = 3) -> List[tuple]:
    """توليد أزواج الأسئلة والإجابات"""
    qa_pairs = []
    
    try:
        # تقسيم النص إلى جمل بسيط
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
        
        for sentence in sentences[:num_questions * 2]:
            # توليد السؤال
            prompt = f"generate question: {sentence}"
            result = st.session_state.models['qg_pipe'](
                prompt,
                max_length=64,
                num_return_sequences=1,
                temperature=0.8
            )
            
            question = clean_text_generated(result[0]['generated_text'])
            
            if question and len(question.split()) >= 3 and '?' in question:
                # العثور على الإجابة
                try:
                    qa_result = st.session_state.models['qa_pipe'](
                        question=question,
                        context=context
                    )
                    answer = qa_result.get('answer', '').strip()
                    score = qa_result.get('score', 0)
                    
                    if answer and score > 0.1 and len(answer.split()) <= 4:
                        qa_pairs.append((question, answer))
                except:
                    continue
            
            if len(qa_pairs) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"خطأ في توليد الأسئلة: {e}")
    
    return qa_pairs

def generate_distractors_simple(answer: str, context: str, num: int = 3) -> List[str]:
    """توليد مشتتات باستخدام طريقة مبسطة"""
    candidates = extract_candidates_simple(context)
    
    # إزالة الإجابة الصحيحة
    candidates = [c for c in candidates if c.lower() != answer.lower()]
    
    # استخدام التضمين الدلالي إذا كان متاحاً
    if len(candidates) > num:
        try:
            sbert = st.session_state.models['embedder']
            cand_emb = sbert.encode(candidates, convert_to_tensor=True)
            ans_emb = sbert.encode([answer], convert_to_tensor=True)
            sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
            
            # اختيار المرشحين الأكثر تشابهاً (ولكن ليس كثيراً)
            selected_indices = []
            for i in sorted(range(len(sims)), key=lambda i: -sims[i]):
                if 0.3 <= sims[i] <= 0.8:
                    selected_indices.append(i)
                if len(selected_indices) >= num:
                    break
            
            if selected_indices:
                return [candidates[i] for i in selected_indices]
        except:
            pass
    
    # إذا فشل الاختيار الدلالي، نعود للاختيار العشوائي
    return candidates[:num]

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """الدالة الرئيسية لتوليد الأسئلة"""
    out = {"questions": []}
    
    # توليد أزواج الأسئلة والإجابات
    qa_pairs = generate_qa_pairs(context, num_questions)
    
    if not qa_pairs:
        st.info("⚠️ لم يتم توليد أي أسئلة. حاول بإدخال نص أطول أو أكثر تفصيلاً.")
        return out
    
    for q, a in qa_pairs:
        trusted_answer = clean_text_generated(a)
        if not trusted_answer:
            continue
        
        # توليد المشتتات
        distractors = generate_distractors_simple(trusted_answer, context, num=3)
        
        # إذا لم يكن هناك مشتتات كافية
        while len(distractors) < 3:
            distractors.append(f"Option {len(distractors)+1}")
        
        # خلط الخيارات
        options = distractors + [trusted_answer]
        random.shuffle(options)
        
        out["questions"].append({
            "question": q,
            "answer": trusted_answer,
            "options": options
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
            st.error("❌ فشل في تحميل النماذج. الرجاء التحقق من السجلات.")
            return
        st.session_state.models = models
        st.success("✅ تم تحميل النماذج بنجاح!")
    
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header("الإعدادات")
        num_questions = st.slider("عدد الأسئلة", 1, 5, 3)
        
        st.header("نصوص مثاليه")
        example_texts = {
            "الساعات الميكانيكية": "The invention of the mechanical clock revolutionized timekeeping. Before clocks, people relied on sundials and water clocks. In the 14th century, European inventors created gears and weights to measure time more accurately.",
            "القراءة": "Reading books is one of the most powerful ways to gain knowledge and expand your imagination.",
            "الاحتباس الحراري": "Global warming is the long-term rise in Earth's average temperature. It is mainly caused by human activities such as burning fossil fuels."
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
            results = generate_mcqs_from_text(input_text, num_questions)
        
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
                for j, option in enumerate(q['options']):
                    is_correct = option == q['answer']
                    emoji = "✅" if is_correct else "⚪"
                    st.markdown(f"{emoji} **{chr(65+j)}.** {option}")
                
                with st.expander("معلومات الإجابة"):
                    st.write(f"**الإجابة الصحيحة:** {q['answer']}")
                
                st.divider()

if __name__ == "__main__":
    main()
