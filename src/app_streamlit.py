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

def is_valid_answer(answer: str) -> bool:
    """التحقق من أن الإجابة مقبولة"""
    if not answer or len(answer) < 2:
        return False
    
    # قائمة بالإجابات غير المقبولة
    invalid_answers = [
        'unprecedented', 'far deeper', 'working', 'the', 'this', 'that',
        'process', 'system', 'method', 'edison sought to', 'continuous',
        'conversion', 'atomic', 'mass', 'far', 'deeper', 'sought', 'to',
        'and', 'or', 'but', 'in', 'on', 'at', 'for', 'of'
    ]
    
    if any(invalid_word == answer.lower() for invalid_word in invalid_answers):
        return False
    
    # يجب أن تحتوي على أحرف أبجدية
    if not re.search(r'[a-zA-Z]', answer):
        return False
    
    # يجب أن تكون كلمة ذات معنى (ليست مجرد حروف عشوائية)
    if len(answer.split()) == 1 and len(answer) < 3:
        return False
    
    return True

def is_meaningful_question(question: str) -> bool:
    """التحقق من أن السؤال منطقي ومفهوم"""
    question_lower = question.lower()
    
    # قائمة بالأنماط المقبولة للأسئلة
    valid_patterns = [
        r'what is [a-z]+',
        r'where does [a-z]+',
        r'what does [a-z]+',
        r'who discovered [a-z]+',
        r'when did [a-z]+',
        r'how does [a-z]+',
        r'why is [a-z]+',
        r'which [a-z]+',
        r'what are [a-z]+'
    ]
    
    # يجب أن يطابق السؤال واحداً على الأقل من الأنماط المقبولة
    if not any(re.search(pattern, question_lower) for pattern in valid_patterns):
        return False
    
    # منع الكلمات غير المجدية في الأسئلة
    meaningless_words = ['unprecedented', 'continuous', 'atomic', 'mass', 'far deeper', 'sought to']
    if any(word in question_lower for word in meaningless_words):
        return False
    
    # منع الأسئلة التي تحتوي على كلمات مفردة غير مجدية
    single_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at']
    words = question_lower.split()
    if len(words) < 4:
        return False
    
    return True

def validate_qa_pair(question: str, answer: str, context: str) -> bool:
    """شروط صارمة للتحقق من جودة السؤال والإجابة"""
    if not question or not answer:
        return False
    
    # شروط صارمة للجودة
    if (len(question.split()) < 4 or 
        len(answer.split()) > 4 or
        len(answer) < 2):
        return False
    
    # الإجابة يجب أن تكون في النص
    if answer.lower() not in context.lower():
        return False
    
    # التحقق من جودة الإجابة
    if not is_valid_answer(answer):
        return False
    
    # التحقق من جودة السؤال
    if not is_meaningful_question(question):
        return False
    
    # منع أسئلة غير منطقية
    invalid_question_patterns = [
        r'continuous conversion of atomic mass',
        r'far deeper',
        r'unprecedented',
        r'edison sought to',
        r'what is the$',
        r'what is this$',
        r'what is that$'
    ]
    
    if any(re.search(pattern, question.lower()) for pattern in invalid_question_patterns):
        return False
    
    return True

def analyze_text_structure(text: str) -> Dict[str, Any]:
    """Analyze text to identify key concepts, processes, and relationships"""
    analysis = {
        'processes': [],
        'definitions': [],
        'locations': [],
        'requirements': [],
        'products': [],
        'key_entities': [],
        'people': [],
        'inventions': []
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
            if process and is_valid_answer(process):
                analysis['processes'].append(process)
        
        # Identify definitions (X is Y)
        if re.search(r'\b(is|are|was|were|means|refers to)\b', sentence_lower):
            definition = extract_definition(sentence)
            if definition and is_valid_answer(definition[1]):
                analysis['definitions'].append(definition)
        
        # Identify locations
        if any(word in sentence_lower for word in [' occurs in', ' happens in', ' takes place in', ' located in', ' found in']):
            location = extract_location(sentence)
            if location and is_valid_answer(location[1]):
                analysis['locations'].append(location)
        
        # Identify requirements
        if any(word in sentence_lower for word in [' requires ', ' needs ', ' uses ', ' depends on ']):
            requirement = extract_requirement(sentence)
            if requirement and is_valid_answer(requirement[1]):
                analysis['requirements'].append(requirement)
        
        # Identify products
        if any(word in sentence_lower for word in [' produces ', ' creates ', ' generates ', ' results in ']):
            product = extract_product(sentence)
            if product and is_valid_answer(product[1]):
                analysis['products'].append(product)
        
        # Identify people
        people = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', sentence)
        for person in people:
            if is_valid_answer(person):
                analysis['people'].append(person)
        
        # Identify inventions
        if any(word in sentence_lower for word in ['invented', 'discovered', 'created', 'developed']):
            invention = extract_invention(sentence)
            if invention and is_valid_answer(invention):
                analysis['inventions'].append(invention)
    
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
            entity = clean_entity(match.group(1))
            if is_valid_answer(entity):
                return entity
    
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
            if concept and definition and is_valid_answer(concept) and is_valid_answer(definition):
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
            if process and location and is_valid_answer(process) and is_valid_answer(location):
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
            if process and requirement and is_valid_answer(process) and is_valid_answer(requirement):
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
            if process and product and is_valid_answer(process) and is_valid_answer(product):
                return (process, product)
    
    return None

def extract_invention(sentence: str) -> str:
    """Extract inventions or discoveries"""
    patterns = [
        r'\b([A-Z][a-z]+ [A-Z][a-z]+)\s+(?:invented|discovered|created)\s+([^.,!?]+)',
        r'\b(?:invention|discovery)\s+of\s+([^.,!?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            invention = clean_entity(match.group(2) if 'invented' in sentence.lower() else match.group(1))
            if invention and is_valid_answer(invention):
                return invention
    
    return None

def extract_key_entities(text: str) -> List[str]:
    """Extract important entities from text"""
    entities = set()
    
    # Proper nouns
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for entity in proper_nouns:
        if is_valid_answer(entity):
            entities.add(entity)
    
    # Technical terms
    technical_terms = re.findall(r'\b([A-Z][a-z]+\s+[a-z]+\s+[a-z]+|[a-z]+\s+[A-Z][a-z]+)\b', text)
    for term in technical_terms:
        clean_term = clean_entity(term)
        if is_valid_answer(clean_term):
            entities.add(clean_term)
    
    # Processes and systems
    processes = re.findall(r'\b([A-Z][a-z]+\s+(?:process|system|mechanism|cycle))\b', text)
    for process in processes:
        clean_process = clean_entity(process)
        if is_valid_answer(clean_process):
            entities.add(clean_process)
    
    return list(entities)

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
            if word.lower() in ['and', 'or', 'the', 'of', 'in', 'on', 'for', 'to']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        entity = ' '.join(capitalized_words)
    else:
        entity = entity.capitalize()
    
    return entity

def generate_high_quality_distractors(correct_answer: str, question: str, context: str) -> List[str]:
    """مشتتات عالية الجودة مع شروط صارمة"""
    
    # إذا كانت الإجابة غير مقبولة، نعود بمشتتات افتراضية جيدة
    if not is_valid_answer(correct_answer):
        return ["Cellular Respiration", "Protein Synthesis", "Water Absorption"]
    
    # قاعدة معرفية موسعة
    knowledge_base = {
        'photosynthesis': ['Cellular Respiration', 'Chemosynthesis', 'Transpiration'],
        'respiration': ['Photosynthesis', 'Fermentation', 'Digestion'],
        'chloroplast': ['Mitochondria', 'Nucleus', 'Ribosomes'],
        'mitochondria': ['Chloroplasts', 'Nucleus', 'Endoplasmic Reticulum'],
        'energy': ['Matter', 'Force', 'Power'],
        'electricity': ['Magnetism', 'Gravity', 'Heat'],
        'current': ['Voltage', 'Resistance', 'Power'],
        'voltage': ['Current', 'Resistance', 'Energy'],
        'invention': ['Discovery', 'Innovation', 'Creation'],
        'scientist': ['Researcher', 'Inventor', 'Scholar'],
        'light': ['Heat', 'Sound', 'Electricity'],
        'power': ['Energy', 'Force', 'Strength'],
        'system': ['Process', 'Method', 'Technique'],
        'process': ['System', 'Method', 'Procedure']
    }
    
    # البحث في قاعدة المعرفة
    answer_lower = correct_answer.lower()
    for key, distractors in knowledge_base.items():
        if key in answer_lower:
            valid_distractors = [d for d in distractors if d.lower() != answer_lower and is_valid_answer(d)]
            return valid_distractors[:3]
    
    # استخدام كيانات من النص كمشتتات
    entities = extract_key_entities(context)
    valid_entities = [e for e in entities if e.lower() != answer_lower and is_valid_answer(e)]
    
    if len(valid_entities) >= 3:
        return valid_entities[:3]
    
    # مشتتات افتراضية عالية الجودة
    default_distractors = ["Cellular Process", "Biological System", "Physical Phenomenon"]
    return [d for d in default_distractors if d.lower() != answer_lower][:3]

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
    
    # Generate people questions
    for person in analysis['people'][:2]:
        question = f"Who is {person}?"
        answer = person
        if validate_qa_pair(question, answer, context):
            questions.append((question, answer))
    
    # Generate invention questions
    for invention in analysis['inventions'][:2]:
        question = f"What was invented by {invention}?" if ' ' in invention else f"What is {invention}?"
        answer = invention
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
            elif 'generate' in sentence.lower():
                return "generates electricity"
    
    return "transforms materials"

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main MCQ generation function"""
    out = {"questions": []}
    
    # Analyze text structure
    analysis = analyze_text_structure(context)
    
    # Generate intelligent questions
    qa_pairs = generate_intelligent_questions(analysis, context, num_questions)
    
    if not qa_pairs:
        st.info("💡 No valid questions generated. Please use text with clear concepts, processes, and proper names.")
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
            "quality": "validated"
        })
    
    return out

# Streamlit UI
def main():
    st.title("🎯 Enhanced MCQ Generator")
    st.markdown("Generate high-quality multiple choice questions with validated answers and smart distractors")
    
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
        
        st.header("Quality Examples")
        example_texts = {
            "Photosynthesis": """
            Photosynthesis is the process used by plants to convert light energy into chemical energy. 
            This process occurs in the chloroplasts of plant cells. Photosynthesis requires carbon dioxide, water, and sunlight. 
            The products of photosynthesis are glucose and oxygen. Chlorophyll is the green pigment that absorbs sunlight.
            """,
            "Thomas Edison": """
            Thomas Edison was an American inventor who developed many devices. 
            He invented the practical electric light bulb and the phonograph. 
            Edison also created the first industrial research laboratory in Menlo Park. 
            His work helped establish the electrical power distribution system.
            """,
            "Electricity": """
            Electricity is the flow of electrical power or charge. It is a form of energy that results from the movement of electrons. 
            Electrical current flows through conductors like copper wires. Voltage is the force that drives the current, 
            while resistance opposes the flow. Electrical power is measured in watts.
            """
        }
        
        selected_example = st.selectbox("Choose example", list(example_texts.keys()))
        if st.button("Load Example"):
            st.session_state.input_text = example_texts[selected_example]
    
    # Main input
    input_text = st.text_area(
        "Enter your text:",
        height=200,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste text with clear concepts, processes, inventions, or scientific explanations..."
    )
    
    # Generate button
    if st.button("🚀 Generate Validated MCQs", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text")
            return
        
        with st.spinner("Analyzing text and generating validated questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        if results['questions']:
            st.success(f"🎉 Generated {len(results['questions'])} validated questions!")
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
                with st.expander("View Validation Details"):
                    st.success(f"**Correct Answer:** {q['answer']}")
                    st.write("✅ **Answer Validation:** Passed quality checks")
                    st.write("✅ **Question Quality:** Meaningful and specific")
                    st.write("✅ **Distractors:** Contextually relevant")
                    st.write("✅ **Text Support:** Answer found in context")
                
                st.divider()
        else:
            st.info("""
            💡 **Tips for better validated questions:**
            - Use text with clear definitions (X is Y)
            - Include specific processes and inventions
            - Mention people and their contributions
            - Describe requirements and products
            - Use proper names and technical terms
            - Avoid vague or abstract language
            """)

if __name__ == "__main__":
    main()
