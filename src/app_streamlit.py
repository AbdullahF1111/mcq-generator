import streamlit as st
import re
import random
import json
from typing import List, Dict, Any
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="Enhanced MCQ Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load models for intelligent question generation"""
    
    with st.spinner("Loading enhanced models..."):
        try:
            qa_pipe = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1
            )
            
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            return {
                'qa_pipe': qa_pipe, 
                'embedder': embedder
            }
            
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return None

def analyze_text_structure(text: str) -> Dict[str, Any]:
    """Analyze text to identify key concepts, processes, and relationships"""
    analysis = {
        'processes': [],
        'definitions': [],
        'locations': [],
        'requirements': [],
        'products': [],
        'key_entities': []
    }
    
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 5:
            continue
            
        sentence_lower = sentence.lower()
        
        # Identify processes
        if any(word in sentence_lower for word in ['process', 'mechanism', 'system', 'method']):
            process = extract_process_name(sentence)
            if process:
                analysis['processes'].append(process)
        
        # Identify definitions (X is Y)
        if re.search(r'\b(is|are|was|were|means|refers to)\b', sentence_lower):
            definition = extract_definition(sentence)
            if definition:
                analysis['definitions'].append(definition)
        
        # Identify locations
        if any(word in sentence_lower for word in [' occurs in', ' happens in', ' takes place in', ' located in', ' found in']):
            location = extract_location(sentence)
            if location:
                analysis['locations'].append(location)
        
        # Identify requirements
        if any(word in sentence_lower for word in [' requires ', ' needs ', ' uses ', ' depends on ']):
            requirement = extract_requirement(sentence)
            if requirement:
                analysis['requirements'].append(requirement)
        
        # Identify products
        if any(word in sentence_lower for word in [' produces ', ' creates ', ' generates ', ' results in ']):
            product = extract_product(sentence)
            if product:
                analysis['products'].append(product)
    
    # Extract key entities
    analysis['key_entities'] = extract_key_entities(text)
    
    return analysis

def extract_process_name(sentence: str) -> str:
    """Extract process names from sentences"""
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:process|mechanism|system)\b',
        r'\b(?:process|mechanism)\s+of\s+([^.,!?]+)',
        r'\b([A-Z][a-z]+\s+[a-z]+)\s+process\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return clean_entity(match.group(1))
    
    return None

def extract_definition(sentence: str) -> tuple:
    """Extract definition pairs (concept, definition)"""
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+([^.,!?]+)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:means|refers to)\s+([^.,!?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            concept = clean_entity(match.group(1))
            definition = clean_entity(match.group(2))
            if concept and definition:
                return (concept, definition)
    
    return None

def extract_location(sentence: str) -> tuple:
    """Extract location information"""
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:occurs|happens|takes place)\s+in\s+([^.,!?]+)',
        r'\b(?:found|located)\s+in\s+([^.,!?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            process = extract_process_name(sentence) or "This process"
            location = clean_entity(match.group(1) if len(match.groups()) == 1 else match.group(2))
            return (process, location)
    
    return None

def extract_requirement(sentence: str) -> tuple:
    """Extract requirement information"""
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:requires|needs)\s+([^.,!?]+)',
        r'\b(?:process|mechanism)\s+requires\s+([^.,!?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            process = extract_process_name(sentence) or "This process"
            requirement = clean_entity(match.group(1) if len(match.groups()) == 1 else match.group(2))
            return (process, requirement)
    
    return None

def extract_product(sentence: str) -> tuple:
    """Extract product information"""
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:produces|creates|generates)\s+([^.,!?]+)',
        r'\b(?:process|mechanism)\s+produces\s+([^.,!?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            process = extract_process_name(sentence) or "This process"
            product = clean_entity(match.group(1) if len(match.groups()) == 1 else match.group(2))
            return (process, product)
    
    return None

def extract_key_entities(text: str) -> List[str]:
    """Extract important entities from text"""
    entities = set()
    
    # Proper nouns
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(proper_nouns)
    
    # Technical terms
    technical_terms = re.findall(r'\b([A-Z][a-z]+\s+[a-z]+\s+[a-z]+|[a-z]+\s+[A-Z][a-z]+)\b', text)
    entities.update(technical_terms)
    
    # Processes and systems
    processes = re.findall(r'\b([A-Z][a-z]+\s+(?:process|system|mechanism|cycle))\b', text)
    entities.update(processes)
    
    return [clean_entity(e) for e in entities if 1 <= len(e.split()) <= 3]

def clean_entity(entity: str) -> str:
    """Clean and format entity"""
    if not entity:
        return ""
    
    entity = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', entity.strip())
    entity = re.sub(r'^(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)
    
    words = entity.split()
    if len(words) > 1:
        capitalized_words = []
        for word in words:
            if word.lower() in ['and', 'or', 'the', 'of', 'in', 'on', 'for']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        entity = ' '.join(capitalized_words)
    else:
        entity = entity.capitalize()
    
    return entity

def generate_high_quality_distractors(correct_answer: str, question: str, context: str) -> List[str]:
    """Generate high-quality, contextually relevant distractors"""
    
    # قاعدة معرفية للمواضيع الشائعة
    knowledge_base = {
        'photosynthesis': ['Cellular Respiration', 'Digestion', 'Fermentation', 'Transpiration'],
        'respiration': ['Photosynthesis', 'Digestion', 'Circulation', 'Excretion'],
        'chloroplast': ['Mitochondria', 'Nucleus', 'Ribosomes', 'Golgi Apparatus'],
        'mitochondria': ['Chloroplasts', 'Nucleus', 'Endoplasmic Reticulum', 'Lysosomes'],
        'glucose': ['Fructose', 'Sucrose', 'Starch', 'Cellulose'],
        'oxygen': ['Nitrogen', 'Carbon Dioxide', 'Hydrogen', 'Helium'],
        'water': ['Carbon Dioxide', 'Oxygen', 'Hydrogen', 'Nitrogen'],
        'carbon dioxide': ['Oxygen', 'Nitrogen', 'Hydrogen', 'Methane']
    }
    
    # ابحث في قاعدة المعرفة أولاً
    answer_lower = correct_answer.lower()
    for key, distractors in knowledge_base.items():
        if key in answer_lower:
            return distractors[:3]
    
    # استخراج كيانات من النص
    entities = extract_key_entities(context)
    entities = [e for e in entities if e.lower() != answer_lower]
    
    # مشتتات ذكية بناءً على نوع السؤال
    question_lower = question.lower()
    
    if 'what is' in question_lower and 'require' not in question_lower:
        # لأسئلة التعريف - استخدام مفاهيم مرتبطة
        return get_related_concepts(correct_answer, entities)
    
    elif 'where' in question_lower:
        # لأسئلة المكان - استخدام أماكن بيولوجية
        bio_locations = ['Mitochondria', 'Nucleus', 'Cytoplasm', 'Cell Membrane', 'Ribosomes']
        return [loc for loc in bio_locations if loc.lower() != answer_lower][:3]
    
    elif 'require' in question_lower:
        # لأسئلة المتطلبات - استخدام مدخلات أخرى
        requirements = ['Oxygen', 'Nitrogen', 'Proteins', 'Minerals', 'Sunlight', 'Water']
        return [req for req in requirements if req.lower() != answer_lower][:3]
    
    elif 'produce' in question_lower or 'create' in question_lower:
        # لأسئلة المنتجات - استخدام منتجات أخرى
        products = ['Carbon Dioxide', 'Water', 'Proteins', 'Energy', 'Heat', 'Oxygen']
        return [prod for prod in products if prod.lower() != answer_lower][:3]
    
    else:
        # مشتتات عامة عالية الجودة
        return get_related_concepts(correct_answer, entities)

def get_related_concepts(correct_answer: str, entities: List[str]) -> List[str]:
    """Get semantically related concepts for better distractors"""
    
    # خريطة المفاهيم المرتبطة
    concept_map = {
        'photosynthesis': ['Cellular Respiration', 'Chemosynthesis', 'Transpiration'],
        'respiration': ['Photosynthesis', 'Fermentation', 'Breathing'],
        'chloroplast': ['Mitochondrion', 'Chromoplast', 'Leucoplast'],
        'mitochondria': ['Chloroplast', 'Ribosome', 'Lysosome'],
        'glucose': ['Fructose', 'Galactose', 'Sucrose'],
        'oxygen': ['Carbon Dioxide', 'Nitrogen', 'Hydrogen'],
        'water': ['Carbon Dioxide', 'Oxygen', 'Hydrogen Peroxide'],
        'energy': ['Matter', 'Force', 'Power'],
        'cell': ['Tissue', 'Organ', 'Organism'],
        'plant': ['Animal', 'Fungus', 'Bacteria']
    }
    
    # ابحث في خريطة المفاهيم
    for key, concepts in concept_map.items():
        if key in correct_answer.lower():
            return concepts
    
    # استخدام الكيانات المستخرجة أو العودة إلى مشتتات عامة جيدة
    if entities:
        return entities[:3]
    else:
        return ['Cellular Process', 'Biological System', 'Metabolic Pathway']

def generate_intelligent_questions(analysis: Dict[str, Any], context: str, num_questions: int = 3) -> List[tuple]:
    """Generate intelligent, specific questions based on text analysis"""
    questions = []
    
    # Generate definition questions
    for concept, definition in analysis['definitions'][:2]:
        question = f"What is {concept}?"
        answer = definition
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    # Generate location questions
    for process, location in analysis['locations'][:2]:
        question = f"Where does {process} occur?"
        answer = location
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    # Generate requirement questions
    for process, requirement in analysis['requirements'][:2]:
        question = f"What does {process} require?"
        answer = requirement
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    # Generate product questions
    for process, product in analysis['products'][:2]:
        question = f"What does {process} produce?"
        answer = product
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    # Generate process identification questions
    for process in analysis['processes'][:2]:
        question = f"What is the name of the process that {get_process_description(process, context)}?"
        answer = process
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    return questions[:num_questions]

def get_process_description(process: str, context: str) -> str:
    """Get a description of what the process does"""
    sentences = re.split(r'[.!?]+', context)
    for sentence in sentences:
        if process.lower() in sentence.lower():
            if 'convert' in sentence.lower():
                return "converts energy"
            elif 'produce' in sentence.lower():
                return "produces energy"
            elif 'release' in sentence.lower():
                return "releases energy"
            elif 'store' in sentence.lower():
                return "stores energy"
    
    return "occurs in living organisms"

def validate_qa_pair(question: str, answer: str, context: str) -> bool:
    """Validate if QA pair is meaningful"""
    if not question or not answer:
        return False
    
    # Basic quality checks
    if len(question.split()) < 4 or len(answer.split()) > 5:
        return False
    
    # Answer should be in context
    if answer.lower() not in context.lower():
        return False
    
    # Avoid generic questions and answers
    generic_answers = ['process', 'system', 'method', 'this', 'the process']
    if answer.lower() in generic_answers:
        return False
    
    return True

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main MCQ generation function"""
    out = {"questions": []}
    
    # Analyze text structure
    analysis = analyze_text_structure(context)
    
    # Generate intelligent questions
    qa_pairs = generate_intelligent_questions(analysis, context, num_questions)
    
    if not qa_pairs:
        st.info("💡 No intelligent questions generated. Try text with clear processes, definitions, and relationships.")
        return out
    
    # Process each QA pair
    for question, answer in qa_pairs:
        # Generate high-quality distractors
        distractors = generate_high_quality_distractors(answer, question, context)
        
        # Create and shuffle options
        options = distractors + [answer]
        random.shuffle(options)
        
        out["questions"].append({
            "question": question,
            "answer": answer,
            "options": options,
            "distractors_quality": "enhanced"
        })
    
    return out

# Streamlit UI
def main():
    st.title("🎯 Enhanced MCQ Generator")
    st.markdown("Generate high-quality multiple choice questions with smart distractors")
    
    # Initialize models
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            st.error("❌ Failed to load models. Please refresh the page.")
            return
        st.session_state.models = models
        st.success("✅ Enhanced models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        st.header("Enhanced Examples")
        example_texts = {
            "Photosynthesis": """
            Photosynthesis is the process used by plants to convert light energy into chemical energy. 
            This process occurs in the chloroplasts of plant cells. Photosynthesis requires carbon dioxide, water, and sunlight. 
            The products of photosynthesis are glucose and oxygen. Chlorophyll is the green pigment that absorbs sunlight.
            The light-dependent reactions capture solar energy, while the Calvin cycle produces sugars.
            """,
            "Cellular Respiration": """
            Cellular respiration is the metabolic process that releases energy from glucose in cells. 
            This process occurs in the mitochondria. Respiration requires glucose and oxygen, 
            and it produces carbon dioxide, water, and ATP energy. There are three main stages: 
            glycolysis, the Krebs cycle, and the electron transport chain.
            """,
            "Water Cycle": """
            The water cycle is the continuous movement of water on Earth. This process involves evaporation, 
            condensation, precipitation, and collection. Water evaporates from oceans and lakes, 
            condenses into clouds in the atmosphere, falls as precipitation, and collects in bodies of water.
            The sun provides the energy that drives the water cycle.
            """
        }
        
        selected_example = st.selectbox("Choose example", list(example_texts.keys()))
        if st.button("Load Enhanced Example"):
            st.session_state.input_text = example_texts[selected_example]
    
    # Main input
    input_text = st.text_area(
        "Enter your text:",
        height=200,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste text with clear processes, definitions, and relationships for best results..."
    )
    
    # Generate button
    if st.button("🚀 Generate Enhanced MCQs", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text")
            return
        
        with st.spinner("Analyzing text and generating high-quality questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        if results['questions']:
            st.success(f"🎉 Generated {len(results['questions'])} enhanced questions!")
            st.divider()
            
            for i, q in enumerate(results['questions'], 1):
                st.subheader(f"Question {i}")
                st.info(f"**{q['question']}**")
                
                # Display options in columns
                col1, col2 = st.columns(2)
                for j, option in enumerate(q['options']):
                    is_correct = option == q['answer']
                    emoji = "✅" if is_correct else "⚪"
                    
                    with (col1 if j % 2 == 0 else col2):
                        st.markdown(f"{emoji} **{chr(65 + j)}.** {option}")
                
                # Enhanced explanation
                with st.expander("View Enhanced Explanation"):
                    st.success(f"**Correct Answer:** {q['answer']}")
                    st.write("✅ **Smart distractors:** Contextually relevant and educationally valid")
                    st.write("✅ **Question quality:** Specific and meaningful")
                    st.write("✅ **Answer validation:** Supported by text context")
                
                st.divider()
        else:
            st.info("""
            💡 **Tips for better enhanced questions:**
            - Include clear definitions (X is Y)
            - Specify locations (occurs in...)
            - List requirements (requires...)
            - Mention products (produces...)
            - Use proper names for processes and concepts
            """)

if __name__ == "__main__":
    main()
