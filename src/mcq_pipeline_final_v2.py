# mcq_pipeline_final_v2.py
import re
import json
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import util
import random
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

# تحميل النماذج بشكل آمن مع Streamlit caching
@st.cache_resource(show_spinner=False)
def load_models():
    """تحميل جميع النماذج مرة واحدة مع معالجة الأخطاء"""
    
    # تحميل spacy model بطريقة آمنة
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model... This may take a few minutes.")
        try:
            # استخدام download مع streamlit
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to load spaCy model: {e}")
            # إنشاء نموذج فارغ كبديل
            nlp = None
    
    # استخدام CPU لتجنب مشاكل الذاكرة
    device = "cpu"
    
    # تحميل نماذج أخرى
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        embedder = None
    
    try:
        qa_pipe = pipeline(
            "question-answering", 
            model="distilbert-base-uncased-distilled-squad", 
            device=-1
        )
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        qa_pipe = None
    
    try:
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1
        )
    except Exception as e:
        st.error(f"Error loading distractor model: {e}")
        distractor_gen = None
    
    return nlp, embedder, qa_pipe, distractor_gen

# -------------------------
# دوال مساعدة مبسطة
# -------------------------

def clean_text_generated(txt: str) -> str:
    """تنظيف النص الناتج من النماذج"""
    if not txt:
        return ""
    txt = txt.strip()
    # إزالة الرموز الغريبة
    txt = re.sub(r'[^\w\s\.\?\!]', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """استخراج العبارات الرئيسية من النص"""
    phrases = []
    
    # استخراج الكيانات المسماة
    if 'nlp' in globals() and globals()['nlp'] is not None:
        try:
            doc = globals()['nlp'](text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME']:
                    phrases.append(clean_text_generated(ent.text))
            
            # استخراج noun chunks
            for chunk in doc.noun_chunks:
                if 1 <= len(chunk.text.split()) <= 3:
                    phrases.append(clean_text_generated(chunk.text))
        except Exception:
            pass
    
    # استخراج التواريخ والأرقام
    years = re.findall(r'\b\d{4}\b', text)
    phrases.extend(years)
    
    # إزالة التكرارات والفرز حسب الطول
    phrases = list(set(phrases))
    phrases.sort(key=len)
    
    return phrases[:max_phrases]

def generate_questions_simple(context: str, num_questions: int = 5) -> List[tuple]:
    """توليد أسئلة بطريقة مبسطة"""
    questions_answers = []
    
    try:
        # استخراج العبارات الرئيسية
        key_phrases = extract_key_phrases(context, num_questions * 3)
        
        for phrase in key_phrases:
            if len(phrase.split()) > 3:
                continue
                
            # إنشاء أنواع مختلفة من الأسئلة
            question_templates = [
                f"What is {phrase}?",
                f"Who is {phrase}?",
                f"When was {phrase}?",
                f"Where is {phrase}?",
            ]
            
            for template in question_templates:
                # التحقق من أن الإجابة موجودة في النص
                if phrase.lower() in context.lower():
                    questions_answers.append((template, phrase))
                    break
            
            if len(questions_answers) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"Error in question generation: {e}")
    
    return questions_answers

def generate_distractors_simple(answer: str, context: str, num_distractors: int = 3) -> List[str]:
    """توليد مشتتات بطريقة مبسطة"""
    distractors = []
    
    try:
        # استخراج عبارات أخرى من النص
        all_phrases = extract_key_phrases(context, 20)
        
        # إزالة الإجابة الصحيحة
        other_phrases = [p for p in all_phrases if p.lower() != answer.lower()]
        
        # اختيار مشتتات عشوائية
        distractors = random.sample(other_phrases, min(num_distractors, len(other_phrases)))
        
    except Exception:
        # إنشاء مشتتات افتراضية إذا فشل الاستخراج
        default_distractors = {
            "person": ["John Smith", "Mary Johnson", "David Brown"],
            "place": ["Paris", "London", "Tokyo"],
            "time": ["2020", "2015", "2010"],
            "thing": ["computer", "book", "car"]
        }
        
        # تحديد نوع الإجابة
        answer_type = "thing"
        if re.search(r'\b(19|20)\d{2}\b', answer):
            answer_type = "time"
        elif any(word in answer.lower() for word in ['city', 'country', 'place', 'location']):
            answer_type = "place"
        elif any(word in answer.lower() for word in ['mr', 'mrs', 'dr', 'professor']):
            answer_type = "person"
            
        distractors = default_distractors.get(answer_type, ["option A", "option B", "option C"])[:num_distractors]
    
    return distractors

def is_valid_question_answer(question: str, answer: str, context: str) -> bool:
    """التحقق من صحة السؤال والإجابة"""
    if not question or not answer:
        return False
    
    if len(question.split()) < 3:
        return False
    
    if len(answer.split()) > 4:
        return False
    
    # التحقق من وجود الإجابة في النص
    if answer.lower() not in context.lower():
        return False
    
    return True

# -------------------------
# الدالة الرئيسية
# -------------------------

def generate_mcqs_from_text(context: str, num_questions: int = 5, desired_distractors: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """الدالة الرئيسية لتوليد أسئلة الاختيار من متعدد"""
    
    # تحميل النماذج
    nlp, embedder, qa_pipe, distractor_gen = load_models()
    
    out = {
        "source_len": len(context.split()), 
        "questions": [],
        "status": "success"
    }
    
    try:
        # توليد الأسئلة باستخدام الطريقة المبسطة
        qa_pairs = generate_questions_simple(context, num_questions)
        
        if verbose:
            st.write(f"Generated {len(qa_pairs)} question-answer pairs")
        
        for q, a in qa_pairs:
            # التحقق من صحة السؤال والإجابة
            if not is_valid_question_answer(q, a, context):
                continue
            
            # توليد المشتتات
            distractors = generate_distractors_simple(a, context, desired_distractors)
            
            # إنشاء الخيارات
            options = distractors + [a]
            random.shuffle(options)
            
            # تحديد نوع السؤال
            qtype = "OTHER"
            if 'who' in q.lower():
                qtype = "PERSON"
            elif 'where' in q.lower():
                qtype = "PLACE"
            elif 'when' in q.lower():
                qtype = "TIME"
            
            # إضافة السؤال إلى النتائج
            out["questions"].append({
                "question": q,
                "answer": a,
                "options": options,
                "qtype": qtype,
                "meta": {
                    "distractors_generated": len(distractors),
                    "method": "simple"
                }
            })
            
            if len(out["questions"]) >= num_questions:
                break
                
    except Exception as e:
        out["status"] = f"error: {str(e)}"
        st.error(f"Error in MCQ generation: {e}")
    
    return out

# -------------------------
# مثال للاختبار
# -------------------------

if __name__ == "__main__":
    # نص تجريبي
    test_text = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair, it was initially 
    criticized by some of France's leading artists and intellectuals for its design, but it has 
    become a global cultural icon of France and one of the most recognizable structures in the world.
    """
    
    results = generate_mcqs_from_text(test_text, num_questions=3, verbose=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
