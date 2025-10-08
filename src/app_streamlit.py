import streamlit as st
import re
import random
import json
from typing import List, Dict, Any
import torch
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="Smart MCQ Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load optimized models"""
    
    with st.spinner("Loading smart models..."):
        try:
            # Use a better model for question generation
            qa_pipe = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1
            )
            
            # Embedding model for semantic similarity
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            # Use T5 for better text generation
            qg_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            qg_pipe = pipeline(
                "text2text-generation",
                model=qg_model,
                tokenizer=qg_tokenizer,
                device=-1
            )
            
            return {
                'qg_pipe': qg_pipe, 
                'qa_pipe': qa_pipe, 
                'embedder': embedder
            }
            
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return None

def create_proper_questions(context: str, num_questions: int = 3) -> List[tuple]:
    """Create proper, well-formed questions using key sentences"""
    qa_pairs = []
    
    # Split into meaningful sentences
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 6]
    
    for sentence in sentences[:num_questions * 2]:
        # Extract key entities to form better questions
        entities = extract_key_entities(sentence)
        
        if entities:
            # Create question based on sentence structure
            question = generate_proper_question(sentence, entities)
            answer = find_best_answer(question, context, entities)
            
            if question and answer and validate_qa_pair(question, answer, context):
                qa_pairs.append((question, answer))
        
        if len(qa_pairs) >= num_questions:
            break
    
    return qa_pairs

def extract_key_entities(text: str) -> List[str]:
    """Extract meaningful entities for questions and answers"""
    entities = set()
    
    # Proper nouns and technical terms
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(proper_nouns)
    
    # Key processes and concepts
    processes = re.findall(r'\b([a-z]+\s+(?:process|system|method|theory|mechanism|reaction))\b', text, re.IGNORECASE)
    entities.update([p.title() for p in processes])
    
    # Important verbs and their objects
    verb_patterns = [
        r'\b(?:is|are|was|were)\s+([^.,!?]+)',
        r'\b(?:called|known as|termed)\s+([^.,!?]+)',
        r'\b(?:requires|needs|uses)\s+([^.,!?]+)',
        r'\b(?:produces|creates|generates)\s+([^.,!?]+)'
    ]
    
    for pattern in verb_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = clean_entity(match)
            if clean_match:
                entities.add(clean_match)
    
    return [e for e in entities if 1 <= len(e.split()) <= 4 and len(e) > 3]

def generate_proper_question(sentence: str, entities: List[str]) -> str:
    """Generate proper, grammatical questions"""
    
    sentence_lower = sentence.lower()
    
    # Pattern-based question generation
    if any(word in sentence_lower for word in [' is ', ' are ', ' was ', ' were ']):
        # For definition/description sentences
        subject = entities[0] if entities else "this process"
        return f"What is {subject}?"
    
    elif any(word in sentence_lower for word in [' requires ', ' needs ', ' uses ']):
        # For requirement sentences
        return "What is required for this process?"
    
    elif any(word in sentence_lower for word in [' produces ', ' creates ', ' generates ']):
        # For production sentences
        return "What is produced by this process?"
    
    elif any(word in sentence_lower for word in [' occurs in ', ' happens in ', ' takes place in ']):
        # For location sentences
        return "Where does this process occur?"
    
    elif any(word in sentence_lower for word in [' discovered ', ' found ', ' invented ']):
        # For discovery sentences
        return "Who made this discovery?"
    
    else:
        # Default question
        if entities:
            return f"What is the role of {entities[0]}?"
        else:
            return "What is the main idea of this text?"

def find_best_answer(question: str, context: str, entities: List[str]) -> str:
    """Find the best answer using QA model with fallbacks"""
    
    try:
        # Try QA model first
        qa_result = st.session_state.models['qa_pipe'](
            question=question,
            context=context,
            top_k=3
        )
        
        for ans in qa_result:
            answer_text = clean_entity(ans['answer'].strip())
            score = ans['score']
            
            if (score > 0.2 and 
                answer_text and 
                1 <= len(answer_text.split()) <= 4 and
                answer_text.lower() in context.lower()):
                return answer_text
    except:
        pass
    
    # Fallback: use entity matching
    question_lower = question.lower()
    
    if 'what is' in question_lower and 'required' in question_lower:
        # Look for requirements in context
        requirements = re.findall(r'(?:requires|needs)\s+([^.,!?]+)', context, re.IGNORECASE)
        if requirements:
            return clean_entity(requirements[0])
    
    elif 'what is produced' in question_lower:
        # Look for products in context
        products = re.findall(r'(?:produces|creates)\s+([^.,!?]+)', context, re.IGNORECASE)
        if products:
            return clean_entity(products[0])
    
    # Return most relevant entity
    if entities:
        return entities[0]
    
    return None

def clean_entity(entity: str) -> str:
    """Clean and format entity"""
    if not entity:
        return ""
    
    entity = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', entity.strip())
    entity = re.sub(r'^(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)
    
    # Capitalize properly
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

def validate_qa_pair(question: str, answer: str, context: str) -> bool:
    """Validate if QA pair makes sense"""
    if not question or not answer:
        return False
    
    # Basic checks
    if len(question.split()) < 3 or len(answer.split()) > 5:
        return False
    
    # Answer should be in context
    if answer.lower() not in context.lower():
        return False
    
    # Question should be properly formatted
    if not question.endswith('?') or not question[0].isupper():
        return False
    
    return True

def generate_smart_distractors(correct_answer: str, context: str, question: str, num_distractors: int = 3) -> List[str]:
    """Generate contextually relevant distractors"""
    
    # Get all entities from context
    all_entities = extract_key_entities(context)
    
    # Remove correct answer
    candidates = [e for e in all_entities if e.lower() != correct_answer.lower()]
    
    # Categorize question type for better distractor selection
    question_lower = question.lower()
    
    distractors = []
    
    # Semantic similarity selection
    if candidates:
        try:
            sbert = st.session_state.models['embedder']
            cand_emb = sbert.encode(candidates, convert_to_tensor=True)
            ans_emb = sbert.encode([correct_answer], convert_to_tensor=True)
            sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
            
            for i, sim in enumerate(sims):
                if 0.3 <= sim <= 0.7:  # Good similarity range
                    distractors.append(candidates[i])
                if len(distractors) >= num_distractors * 2:
                    break
        except:
            distractors = candidates[:num_distractors * 2]
    
    # Question-type specific distractors
    if 'what is required' in question_lower:
        additional = ["Water", "Sunlight", "Oxygen", "Nitrogen"]
        distractors.extend(additional)
    
    elif 'what is produced' in question_lower:
        additional = ["Carbon Dioxide", "Water", "Energy", "Proteins"]
        distractors.extend(additional)
    
    elif 'where does' in question_lower:
        additional = ["Mitochondria", "Nucleus", "Cytoplasm", "Cell Membrane"]
        distractors.extend(additional)
    
    # Filter and select final distractors
    filtered = []
    for distractor in distractors:
        if (distractor.lower() != correct_answer.lower() and
            len(distractor.split()) <= 3 and
            len(distractor) > 2 and
            distractor not in filtered):
            filtered.append(distractor)
    
    # Ensure we have enough distractors
    while len(filtered) < num_distractors:
        fallbacks = ["Alternative Process", "Different Method", "Other Component"]
        for fb in fallbacks:
            if fb not in filtered and len(filtered) < num_distractors:
                filtered.append(fb)
    
    return filtered[:num_distractors]

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main MCQ generation function"""
    out = {"questions": []}
    
    # Generate proper QA pairs
    qa_pairs = create_proper_questions(context, num_questions)
    
    if not qa_pairs:
        st.info("💡 No valid questions generated. Try more detailed text with clear concepts.")
        return out
    
    # Process each QA pair
    for question, answer in qa_pairs:
        # Generate smart distractors
        distractors = generate_smart_distractors(answer, context, question, 3)
        
        # Create and shuffle options
        options = distractors + [answer]
        random.shuffle(options)
        
        out["questions"].append({
            "question": question,
            "answer": answer,
            "options": options
        })
    
    return out

# Streamlit UI
def main():
    st.title("🎯 Smart MCQ Generator")
    st.markdown("Generate high-quality multiple choice questions with proper grammar and relevant options")
    
    # Initialize models
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            st.error("❌ Failed to load models. Please refresh the page.")
            return
        st.session_state.models = models
        st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        st.header("Quality Examples")
        example_texts = {
            "Photosynthesis": """
            Photosynthesis is the process used by plants to convert light energy into chemical energy. 
            This process occurs in the chloroplasts of plant cells. Photosynthesis requires carbon dioxide, water, and sunlight. 
            The products of this process are glucose and oxygen. The light-dependent reactions capture energy from sunlight, 
            while the Calvin cycle uses that energy to produce sugars. Chlorophyll is the green pigment that absorbs sunlight.
            """,
            "Cellular Respiration": """
            Cellular respiration is the process that releases energy from glucose in cells. 
            This process occurs in the mitochondria. Respiration requires glucose and oxygen, 
            and it produces carbon dioxide, water, and ATP energy. There are three main stages: 
            glycolysis, the Krebs cycle, and the electron transport chain.
            """,
            "Industrial Revolution": """
            The Industrial Revolution began in Britain in the late 18th century. 
            James Watt improved the steam engine, which became a key invention. 
            The revolution transformed society from agricultural to industrial. 
            Factories developed and cities grew rapidly during this period.
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
        placeholder="Paste detailed text with clear concepts, processes, and proper names..."
    )
    
    # Generate button
    if st.button("🎯 Generate Smart MCQs", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text")
            return
        
        with st.spinner("Generating high-quality questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        if results['questions']:
            st.success(f"✨ Generated {len(results['questions'])} quality questions!")
            st.divider()
            
            for i, q in enumerate(results['questions'], 1):
                st.subheader(f"Question {i}")
                st.info(f"**{q['question']}**")
                
                # Display options
                col1, col2 = st.columns(2)
                for j, option in enumerate(q['options']):
                    is_correct = option == q['answer']
                    emoji = "✅" if is_correct else "⚪"
                    
                    with (col1 if j % 2 == 0 else col2):
                        st.markdown(f"{emoji} **{chr(65 + j)}.** {option}")
                
                # Answer section
                with st.expander("View Explanation"):
                    st.success(f"**Correct Answer:** {q['answer']}")
                    st.write("This answer is directly supported by the text context.")
                
                st.divider()
        else:
            st.info("""
            🔍 Tips for better questions:
            - Use text with clear processes and definitions
            - Include proper names and technical terms
            - Ensure sentences are complete and well-structured
            - Provide context with cause-effect relationships
            """)

if __name__ == "__main__":
    main()
