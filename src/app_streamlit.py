# app_streamlit.py
import streamlit as st
import re
import random
import subprocess
import sys
from typing import List, Dict, Any

# إعدادات الصفحة
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="❓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """تحميل spaCy model بطريقة آمنة"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        try:
            st.info("📥 Downloading language model... (first time only)")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            st.error(f"❌ Failed to load language model: {e}")
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
        ("Which year was {entity} established?", "DATE"),
        ("Who discovered {entity}?", "PERSON"),
        ("Where can you find {entity}?", "GPE")
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
            "person": ["Albert Einstein", "Marie Curie", "Charles Darwin", "Isaac Newton"],
            "place": ["Paris, France", "London, UK", "Tokyo, Japan", "New York, USA"],
            "date": ["1990", "1985", "2000", "2015"],
            "org": ["United Nations", "World Bank", "Red Cross", "World Health Organization"],
            "default": ["Option A", "Option B", "Option C", "Option D"]
        }
        
        # تحديد نوع الكيان
        entity_type = "default"
        if any(word in answer.lower() for word in ['city', 'country', 'state', 'river', 'mountain']):
            entity_type = "place"
        elif re.search(r'\b(19|20)\d{2}\b', answer):
            entity_type = "date"
        elif any(word in answer.lower() for word in ['company', 'organization', 'corporation', 'university']):
            entity_type = "org"
        elif any(word in answer.lower() for word in ['mr.', 'mrs.', 'dr.', 'professor', 'president']):
            entity_type = "person"
            
        other_entities.extend(default_distractors.get(entity_type, default_distractors["default"]))
    
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
            "message": "No entities found in text. Try more descriptive text with names, dates, or places."
        }
    
    # توليد الأسئلة
    qa_pairs = generate_questions_from_text(context, num_questions)
    
    if not qa_pairs:
        return {
            "source_len": len(context.split()),
            "questions": [],
            "status": "error",
            "message": "Could not generate questions. Try text with more clear facts and entities."
        }
    
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

# واجهة المستخدم
st.title("🧠 MCQ Generator")
st.markdown("Generate multiple-choice questions from any text using AI")

# نموذج الإدخال
with st.form("input_form"):
    text_input = st.text_area(
        "**Enter your text:**",
        height=150,
        placeholder="Paste your text here...\n\nExample: The Eiffel Tower is in Paris, France. It was built in 1889 and is named after engineer Gustave Eiffel.",
        help="For best results, use text with names, dates, places, and facts."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("**Number of questions:**", 1, 5, 3)
    with col2:
        num_options = st.slider("**Options per question:**", 2, 4, 3)
    
    generate_btn = st.form_submit_button("🎯 Generate Questions", type="primary")

# معالجة النتائج
if generate_btn:
    if not text_input.strip():
        st.error("❌ Please enter some text!")
    else:
        with st.spinner("🔄 Analyzing text and generating questions..."):
            try:
                results = generate_mcqs_from_text(
                    context=text_input,
                    num_questions=num_questions,
                    desired_distractors=num_options - 1  # -1 لأن الإجابة الصحيحة تحسب
                )
                
                # عرض النتائج
                if results["status"] == "success" and results["questions"]:
                    st.success(f"✅ Generated {len(results['questions'])} questions!")
                    st.info(f"📊 Found {results['entities_found']} entities in your text")
                    
                    for i, q in enumerate(results["questions"], 1):
                        with st.container():
                            st.markdown(f"### ❓ Question {i}")
                            st.markdown(f"**{q['question']}**")
                            
                            # عرض الخيارات
                            st.markdown("**Options:**")
                            for j, option in enumerate(q["options"]):
                                emoji = "✅" if option == q["answer"] else "⚪"
                                st.markdown(f"{emoji} **{chr(65+j)}.** {option}")
                            
                            st.caption(f"🔹 Type: {q['qtype']} • 📝 Entities: {q['meta']['entities_found']}")
                            st.divider()
                
                elif results["status"] == "error":
                    st.warning(f"⚠️ {results['message']}")
                    
                else:
                    st.info("🤔 No questions could be generated. Try using more descriptive text with clear facts.")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Please try again with different text.")

# قسم الأمثلة والنصائح
with st.expander("💡 **Tips & Examples**", expanded=True):
    st.markdown("""
    **🎯 For best results, include:**
    - 👤 **People names** (Albert Einstein, Marie Curie)
    - 🗺️ **Places** (Paris, France, Amazon River)  
    - 📅 **Dates & years** (2020, 1995, 19th century)
    - 🏛️ **Organizations** (United Nations, Google)
    - 📚 **Facts & definitions**
    
    **📝 Example text that works well:**
    ```
    The Amazon River is the largest river by discharge volume in the world. 
    It flows through South America and has a length of approximately 6,400 km. 
    The river system originates in the Andes Mountains of Peru and was first 
    explored by Francisco de Orellana in 1542. It empties into the Atlantic Ocean.
    ```
    
    **⚡ How it works:**
    1. Extracts key entities (names, places, dates) from your text
    2. Generates questions based on these entities
    3. Creates distractors from other entities in the text
    4. Shuffles options randomly
    """)

# زر تحميل مثال
if not generate_btn:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎯 Load Example", use_container_width=True):
            example_text = """
            The Great Pyramid of Giza is the oldest and largest of the three pyramids in Egypt. 
            It was built as a tomb for the Pharaoh Khufu around 2560 BC over a 20-year period. 
            The pyramid was constructed by thousands of workers using limestone blocks. 
            It originally stood at 146.6 meters tall and was the tallest structure for 3,800 years.
            The architect was Hemiunu, a relative of Pharaoh Khufu. The pyramid is located on the Giza Plateau.
            """
            st.session_state.example_text = example_text
            st.rerun()

# إذا كان هناك مثال محمل، عرضه
if 'example_text' in st.session_state:
    text_input = st.session_state.example_text

# تذييل الصفحة
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and spaCy • Simple MCQ Generator")
