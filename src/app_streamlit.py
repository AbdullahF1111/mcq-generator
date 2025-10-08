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
    page_title="MCQ Generator Pro",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load verified models that work on Hugging Face"""
    
    with st.spinner("Loading verified models..."):
        try:
            # Option 1: Use a simple T5 model for question generation
            # We'll use a basic T5 model and craft better prompts
            qg_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            qg_pipe = pipeline(
                "text2text-generation",
                model=qg_model,
                tokenizer=qg_tokenizer,
                device=-1
            )
            
            # Verified QA model
            qa_pipe = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",  # Verified working
                device=-1
            )
            
            # Embedding model
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            # Simple distractor generator using same T5 model
            distractor_gen = pipeline(
                "text2text-generation",
                model=qg_model,
                tokenizer=qg_tokenizer,
                device=-1
            )
            
            return {
                'qg_pipe': qg_pipe, 
                'qa_pipe': qa_pipe, 
                'embedder': embedder, 
                'distractor_gen': distractor_gen
            }
            
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            # Fallback to minimal models
            try:
                st.info("Trying fallback models...")
                qa_pipe = pipeline("question-answering", device=-1)
                embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                return {
                    'qg_pipe': None,
                    'qa_pipe': qa_pipe,
                    'embedder': embedder,
                    'distractor_gen': None
                }
            except:
                return None

def clean_text_generated(txt: str) -> str:
    """Clean generated text"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'^[\'"]|[\'"]$', '', txt)
    return txt

def generate_question_from_sentence(sentence: str) -> str:
    """Generate question using T5 with better prompting"""
    if st.session_state.models['qg_pipe'] is None:
        # Fallback: create simple questions
        words = sentence.split()
        if len(words) > 5:
            key_terms = [word for word in words if word.istitle() and len(word) > 3]
            if key_terms:
                return f"What is {key_terms[0]}?"
        return f"What is described in this text?"
    
    try:
        # Better prompt for T5
        prompt = f"generate question: {sentence}"
        
        result = st.session_state.models['qg_pipe'](
            prompt,
            max_length=64,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        
        question = clean_text_generated(result[0]['generated_text'])
        
        # Ensure it's a proper question
        if not question.endswith('?'):
            question += '?'
        if not question[0].isupper():
            question = question[0].upper() + question[1:]
            
        return question
        
    except Exception:
        # Fallback question
        return f"What is the main idea of this text?"

def extract_entities_from_text(text: str) -> List[str]:
    """Extract potential answer candidates from text"""
    entities = set()
    
    # Extract proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(proper_nouns)
    
    # Extract key phrases
    key_phrases = re.findall(r'\b([A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+)\b', text)
    entities.update(key_phrases)
    
    # Extract technical terms (longer words that might be important)
    words = re.findall(r'\b[a-zA-Z]{6,}\b', text)
    entities.update([word.title() for word in words])
    
    # Filter and clean
    clean_entities = []
    for entity in entities:
        clean_entity = clean_text_generated(entity)
        if (clean_entity and 
            1 <= len(clean_entity.split()) <= 3 and
            len(clean_entity) > 3):
            clean_entities.append(clean_entity)
    
    return clean_entities

def generate_distractors_smart(correct_answer: str, context: str, num_distractors: int = 3) -> List[str]:
    """Generate smart distractors using multiple strategies"""
    
    # Get all entities from context
    all_entities = extract_entities_from_text(context)
    
    # Remove correct answer
    candidates = [e for e in all_entities if e.lower() != correct_answer.lower()]
    
    # Strategy 1: Semantic similarity
    semantic_candidates = []
    if candidates:
        try:
            sbert = st.session_state.models['embedder']
            cand_emb = sbert.encode(candidates, convert_to_tensor=True)
            ans_emb = sbert.encode([correct_answer], convert_to_tensor=True)
            sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
            
            for i, sim in enumerate(sims):
                if 0.3 <= sim <= 0.7:  # Good similarity range
                    semantic_candidates.append(candidates[i])
        except:
            semantic_candidates = candidates[:num_distractors * 2]
    
    # Strategy 2: Pattern-based distractors
    pattern_distractors = []
    
    # For scientific texts
    scientific_terms = ["Chemical Process", "Biological Mechanism", "Physical Reaction", 
                       "Molecular Structure", "Atomic Process"]
    pattern_distractors.extend(scientific_terms)
    
    # For historical texts  
    historical_terms = ["Earlier Period", "Later Development", "Alternative Theory",
                       "Previous Method", "Subsequent Event"]
    pattern_distractors.extend(historical_terms)
    
    # Combine strategies
    all_candidates = list(set(semantic_candidates + pattern_distractors + candidates))
    
    # Filter out poor candidates
    filtered = []
    for candidate in all_candidates:
        if (candidate.lower() != correct_answer.lower() and
            len(candidate) > 2 and
            len(candidate.split()) <= 3):
            filtered.append(candidate)
    
    # Select final distractors
    final_distractors = []
    for candidate in filtered:
        if candidate not in final_distractors:
            final_distractors.append(candidate)
        if len(final_distractors) >= num_distractors:
            break
    
    # Fill remaining slots if needed
    while len(final_distractors) < num_distractors:
        fallback = f"Option {len(final_distractors) + 1}"
        if fallback not in final_distractors:
            final_distractors.append(fallback)
    
    return final_distractors[:num_distractors]

def generate_qa_pairs_robust(context: str, num_questions: int = 3) -> List[tuple]:
    """Robust QA pair generation with fallbacks"""
    qa_pairs = []
    
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 8]  # Longer sentences
        
        for sentence in sentences[:num_questions * 2]:
            # Generate question
            question = generate_question_from_sentence(sentence)
            
            if question:
                # Find answer using QA model
                try:
                    qa_result = st.session_state.models['qa_pipe'](
                        question=question,
                        context=context,
                        top_k=2
                    )
                    
                    # Try to find a good answer
                    best_answer = None
                    for ans in qa_result:
                        answer_text = ans['answer'].strip()
                        score = ans['score']
                        
                        if (score > 0.1 and 
                            answer_text and 
                            len(answer_text.split()) <= 4 and
                            answer_text.lower() in context.lower()):
                            best_answer = clean_text_generated(answer_text)
                            break
                    
                    if best_answer:
                        qa_pairs.append((question, best_answer))
                        
                except Exception:
                    # Fallback: use entity extraction for answer
                    entities = extract_entities_from_text(sentence)
                    if entities:
                        qa_pairs.append((question, entities[0]))
            
            if len(qa_pairs) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"Error in QA generation: {e}")
    
    return qa_pairs

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main MCQ generation function"""
    out = {"questions": []}
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs_robust(context, num_questions)
    
    if not qa_pairs:
        st.info("💡 No questions generated. Try using more detailed text with clear concepts.")
        return out
    
    # Process each QA pair
    for question, answer in qa_pairs:
        # Generate distractors
        distractors = generate_distractors_smart(answer, context, 3)
        
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
    st.title("🎯 MCQ Generator Pro")
    st.markdown("Generate multiple choice questions from any text")
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.models = None
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models... This may take a minute"):
            models = load_models()
            if models is None:
                st.error("""
                ❌ Failed to load models. Possible solutions:
                1. Check your internet connection
                2. Try refreshing the page
                3. The models might be temporarily unavailable
                """)
                return
            
            st.session_state.models = models
            st.session_state.models_loaded = True
        st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        st.header("Example Texts")
        example_texts = {
            "Science - Photosynthesis": """
            Photosynthesis is the process used by plants to convert light energy into chemical energy. 
            This process occurs in the chloroplasts of plant cells and requires carbon dioxide, water, and sunlight. 
            The products of photosynthesis are glucose and oxygen. The light-dependent reactions capture energy from sunlight, 
            while the Calvin cycle uses that energy to produce sugars.
            """,
            "History - Industrial Revolution": """
            The Industrial Revolution began in Great Britain in the late 18th century. 
            Major inventions included the steam engine, spinning jenny, and power loom. 
            This period saw a shift from agrarian societies to industrialized urban centers. 
            Factory systems developed and transportation improved with railways and canals.
            """,
            "Technology - Internet": """
            The Internet is a global network of interconnected computers that communicate using standardized protocols. 
            It originated from ARPANET, a project funded by the U.S. Department of Defense. 
            Tim Berners-Lee developed the World Wide Web in 1989, creating the first web browser and server. 
            The Internet has revolutionized communication, commerce, and information sharing.
            """
        }
        
        selected_example = st.selectbox("Choose an example", list(example_texts.keys()))
        if st.button("Load Example"):
            st.session_state.input_text = example_texts[selected_example]
    
    # Main input area
    input_text = st.text_area(
        "Enter your text below:",
        height=200,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste your text here... (For best results, use detailed text with clear concepts and proper names)"
    )
    
    # Generate button
    if st.button("🎯 Generate MCQs", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text first")
            return
        
        if len(input_text.split()) < 20:
            st.warning("📝 For better results, use longer text (at least 50 words)")
        
        # Generate MCQs
        with st.spinner("Generating questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        if results['questions']:
            st.success(f"✨ Generated {len(results['questions'])} questions!")
            st.divider()
            
            for i, q in enumerate(results['questions'], 1):
                st.subheader(f"Question {i}")
                st.markdown(f"**{q['question']}**")
                
                # Display options in a clean format
                col1, col2 = st.columns(2)
                options_displayed = 0
                
                for j, option in enumerate(q['options']):
                    is_correct = option == q['answer']
                    emoji = "✅" if is_correct else "⚪"
                    
                    if options_displayed % 2 == 0:
                        with col1:
                            st.markdown(f"{emoji} **{chr(65 + j)}.** {option}")
                    else:
                        with col2:
                            st.markdown(f"{emoji} **{chr(65 + j)}.** {option}")
                    
                    options_displayed += 1
                
                # Answer reveal
                with st.expander("Show Answer"):
                    st.success(f"**Correct answer: {q['answer']}**")
                
                st.divider()
        else:
            st.info("""
            🔍 No questions could be generated. Try:
            - Using more detailed text
            - Including proper names and specific concepts
            - Ensuring the text has complete sentences
            - Adding technical or specialized vocabulary
            """)

if __name__ == "__main__":
    main()
