# mcq_pipeline_simple.py
import re
import random
import streamlit as st
from typing import List, Dict, Any

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """تحميل spaCy model بطريقة آمنة"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        try:
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            st.error(f"Failed to load spaCy: {e}")
            return None

def extract_entities(text: str) -> List[str]:
    """استخراج الكيانات من النص"""
    nlp = load_spacy_model()
    if nlp is None:
        return []
    
    try:
        doc = nlp(text)
        entities = []
        
        # الكيانات المسماة
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME']:
                entities.append(ent.text.strip())
        
        # العبارات الاسمية
        for chunk in doc.noun_chunks:
            if 1 <= len(chunk.text.split()) <= 3:
                entities.append(chunk.text.strip())
        
        return list(set(entities))  # إزالة التكرارات
    except Exception:
        return []

def generate_questions_from_text(text: str, num_questions: int = 3) -> List[tuple]:
    """توليد أسئلة من النص"""
    entities = extract_entities(text)
    questions = []
    
    question_templates = [
        ("What is {entity}?", "PERSON"),
        ("Who is {entity}?", "PERSON"), 
        ("Where is {entity} located?", "GPE"),
        ("When was {entity} founded?", "DATE"),
        ("What organization is {entity} part of?", "ORG"),
        ("Which year was {entity} established?", "DATE")
    ]
    
    for entity in entities[:num_questions * 2]:
        for template, expected_type in question_templates:
            if len(questions) >= num_questions:
                break
                
            question = template.format(entity=entity)
            
            # التحقق من أن الإجابة موجودة في النص
            if entity.lower() in text.lower():
                questions.append((question, entity))
                break
    
    return questions[:num_questions]

def generate_distractors(answer: str, all_entities: List[str], num_distractors: int = 3) -> List[str]:
    """توليد مشتتات من الكيانات الأخرى"""
    # إزالة الإجابة الصحيحة
    other_entities = [e for e in all_entities if e.lower() != answer.lower()]
    
    # إذا لم يكن هناك كيانات كافية، استخدم مشتتات افتراضية
    if len(other_entities) < num_distractors:
        default_distractors = {
            "person": ["Albert Einstein", "Marie Curie", "Charles Darwin"],
            "place": ["Paris, France", "London, UK", "Tokyo, Japan"],
            "date": ["1990", "1985", "2000"],
            "org": ["United Nations", "World Bank", "Red Cross"]
        }
        
        # تحديد نوع الكيان
        entity_type = "person"
        if any(word in answer.lower() for word in ['city', 'country', 'state']):
            entity_type = "place"
        elif re.search(r'\b(19|20)\d{2}\b', answer):
            entity_type = "date"
        elif any(word in answer.lower() for word in ['company', 'organization', 'corporation']):
            entity_type = "org"
            
        other_entities.extend(default_distractors.get(entity_type, ["Option A", "Option B", "Option C"]))
    
    # اختيار عشوائي
    return random.sample(other_entities, min(num_distractors, len(other_entities)))

def clean_text(text: str) -> str:
    """تنظيف النص"""
    if not text:
        return ""
    text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_mcqs_from_text(context: str, num_questions: int = 3, desired_distractors: int = 3) -> Dict[str, Any]:
    """الدالة الرئيسية لتوليد الأسئلة"""
    
    # تنظيف النص
    context = clean_text(context)
    
    if len(context.split()) < 10:
        return {
            "source_len": len(context.split()),
            "questions": [],
            "status": "error",
            "message": "Text too short. Please provide at least 10 words."
        }
    
    # استخراج الكيانات
    all_entities = extract_entities(context)
    
    if not all_entities:
        return {
            "source_len": len(context.split()),
            "questions": [],
            "status": "error", 
            "message": "No entities found in text. Try more descriptive text."
        }
    
    # توليد الأسئلة
    qa_pairs = generate_questions_from_text(context, num_questions)
    
    questions = []
    for question, answer in qa_pairs:
        # توليد المشتتات
        distractors = generate_distractors(answer, all_entities, desired_distractors)
        
        # إنشاء الخيارات
        options = distractors + [answer]
        random.shuffle(options)
        
        # تحديد نوع السؤال
        qtype = "FACT"
        if 'who' in question.lower():
            qtype = "PERSON"
        elif 'where' in question.lower():
            qtype = "PLACE" 
        elif 'when' in question.lower():
            qtype = "TIME"
        
        questions.append({
            "question": question,
            "answer": answer,
            "options": options,
            "qtype": qtype,
            "meta": {
                "entities_found": len(all_entities),
                "distractors": len(distractors)
            }
        })
    
    return {
        "source_len": len(context.split()),
        "questions": questions,
        "status": "success",
        "entities_found": len(all_entities)
    }
