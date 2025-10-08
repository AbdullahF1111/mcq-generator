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
    page_title="Fast MCQ Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load lightweight models for Streamlit Cloud"""
    device = "cpu"  # Force CPU for compatibility
    
    with st.spinner("Loading optimized models..."):
        try:
            # Lightweight question generation
            qg_pipe = pipeline(
                "text2text-generation",
                model="mrm8488/t5-small-finetuned-question-generation-ap",  # Smaller model
                device=-1,  # CPU only
                max_length=64
            )
            
            # Lightweight QA model
            qa_pipe = pipeline(
                "question-answering",
                model="mrm8488/bert-tiny-finetuned-squadv2",  # Tiny model
                device=-1
            )
            
            # Embedding model (already lightweight)
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
            # Lightweight distractor generation
            distractor_gen = pipeline(
                "text2text-generation",
                model="mrm8488/t5-small-finetuned-qa-boolq",  # Small model
                device=-1,
                max_length=50
            )
            
            return {
                'qg_pipe': qg_pipe, 
                'qa_pipe': qa_pipe, 
                'embedder': embedder, 
                'distractor_gen': distractor_gen,
                'device': device
            }
            
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None

def clean_text_generated(txt: str) -> str:
    """Clean generated text"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'^[\'"]|[\'"]$', '', txt)
    return txt

def grammar_correct_question(question: str) -> str:
    """Fix common grammar issues"""
    corrections = {
        r'\bhe\s+found\b': 'he found that',
        r'\bwhat\s+did\s+(\w+)\s+found\b': r'what did \1 find',
        r'\bhow\s+does\s+(\w+)\s+found\b': r'how does \1 find',
    }
    
    corrected = question
    for pattern, replacement in corrections.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    corrected = corrected.strip()
    if corrected and not corrected[0].isupper():
        corrected = corrected[0].upper() + corrected[1:]
    if not corrected.endswith('?'):
        corrected += '?'
    
    return corrected

def extract_key_entities(text: str) -> List[str]:
    """Extract meaningful entities using regex patterns"""
    entities = set()
    
    # Proper nouns and technical terms
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(proper_nouns)
    
    # Key phrases after specific patterns
    patterns = [
        r'(?:called|known as|termed)\s+([^.,;!?]+)',
        r'(?:process|concept|phenomenon)\s+of\s+([^.,;!?]+)',
        r'\b([A-Z][a-z]+\s+[a-z]+\s+[a-z]+)\b'  # Multi-word terms
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = clean_entity(match)
            if clean_match:
                entities.add(clean_match)
    
    # Filter for quality
    return [e for e in entities if 1 <= len(e.split()) <= 3 and len(e) > 3]

def clean_entity(entity: str) -> str:
    """Clean entity formatting"""
    if not entity:
        return ""
    
    entity = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', entity.strip())
    entity = re.sub(r'^(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)
    
    words = entity.split()
    if len(words) > 1:
        capitalized_words = []
        for word in words:
            if word.lower() in ['and', 'or', 'the', 'of', 'in', 'on']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        entity = ' '.join(capitalized_words)
    else:
        entity = entity.capitalize()
    
    return entity

def generate_smart_distractors(question: str, correct_answer: str, context: str, num: int = 3) -> List[str]:
    """Generate quality distractors efficiently"""
    
    # Get entities from context
    entities = extract_key_entities(context)
    entities = [e for e in entities if e.lower() != correct_answer.lower()]
    
    # Use semantic similarity for selection
    semantic_distractors = get_semantic_distractors(correct_answer, entities, num * 2)
    
    # Add pattern-based distractors
    pattern_distractors = get_pattern_based_distractors(question, correct_answer, context)
    
    # Combine and filter
    all_distractors = list(set(semantic_distractors + pattern_distractors))
    filtered = filter_distractors(all_distractors, correct_answer, question)
    
    return select_final_distractors(filtered, num)

def get_semantic_distractors(answer: str, candidates: List[str], num: int) -> List[str]:
    """Select distractors based on semantic similarity"""
    if not candidates:
        return []
    
    try:
        sbert = st.session_state.models['embedder']
        cand_emb = sbert.encode(candidates, convert_to_tensor=True)
        ans_emb = sbert.encode([answer], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
        
        # Select candidates with moderate similarity
        selected = []
        for i, sim in enumerate(sims):
            if 0.3 <= sim <= 0.7:
                selected.append(candidates[i])
            if len(selected) >= num:
                break
        
        return selected
    except:
        return candidates[:num]

def get_pattern_based_distractors(question: str, answer: str, context: str) -> List[str]:
    """Generate distractors based on question patterns"""
    distractors = []
    q_lower = question.lower()
    
    # For discovery questions
    if re.search(r'\b(discover|find|identify)\b', q_lower):
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
        other_entities = [pn for pn in proper_nouns if pn.lower() != answer.lower()]
        distractors.extend(other_entities[:2])
    
    # For process/questions
    elif re.search(r'\b(process|phenomenon|mechanism)\b', q_lower):
        process_terms = ["Chemical Process", "Biological Mechanism", "Physical Reaction"]
        distractors.extend(process_terms)
    
    return distractors

def filter_distractors(distractors: List[str], answer: str, question: str) -> List[str]:
    """Filter out poor quality distractors"""
    filtered = []
    question_words = set(question.lower().split())
    
    for distractor in distractors:
        d_lower = distractor.lower()
        
        if (d_lower == answer.lower() or
            any(q_word in d_lower for q_word in question_words if len(q_word) > 3) or
            len(distractor) < 2):
            continue
            
        filtered.append(distractor)
    
    return filtered

def select_final_distractors(distractors: List[str], num: int) -> List[str]:
    """Select final distractors ensuring diversity"""
    if len(distractors) <= num:
        return distractors
    
    # Simple diversity selection - take first N unique ones
    final = []
    for distractor in distractors:
        if distractor not in final:
            final.append(distractor)
        if len(final) >= num:
            break
    
    # Fill with fallbacks if needed
    fallbacks = ["Alternative Method", "Different Process", "Other Approach"]
    while len(final) < num:
        for fb in fallbacks:
            if fb not in final and len(final) < num:
                final.append(fb)
    
    return final[:num]

def generate_qa_pairs_fast(context: str, num_questions: int = 3) -> List[tuple]:
    """Fast QA pair generation"""
    qa_pairs = []
    
    try:
        # Split into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 6]
        
        for sentence in sentences[:num_questions * 2]:
            # Generate question
            prompt = f"question: {sentence}"
            
            result = st.session_state.models['qg_pipe'](
                prompt,
                max_length=64,
                num_return_sequences=1,
                temperature=0.7
            )
            
            raw_question = clean_text_generated(result[0]['generated_text'])
            question = grammar_correct_question(raw_question)
            
            if question and '?' in question:
                # Get answer
                try:
                    qa_result = st.session_state.models['qa_pipe'](
                        question=question,
                        context=context
                    )
                    
                    answer_text = qa_result.get('answer', '').strip()
                    score = qa_result.get('score', 0)
                    
                    if (answer_text and score > 0.2 and
                        len(answer_text.split()) <= 4 and
                        answer_text.lower() in context.lower()):
                        
                        qa_pairs.append((question, clean_entity(answer_text)))
                        
                except Exception:
                    continue
            
            if len(qa_pairs) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"Question generation error: {e}")
    
    return qa_pairs

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main function with fast processing"""
    out = {"questions": []}
    
    qa_pairs = generate_qa_pairs_fast(context, num_questions)
    
    if not qa_pairs:
        st.info("💡 No questions generated. Try more detailed text.")
        return out
    
    for q, a in qa_pairs:
        # Generate distractors
        distractors = generate_smart_distractors(q, a, context, 3)
        
        # Shuffle options
        options = distractors + [a]
        random.shuffle(options)
        
        out["questions"].append({
            "question": q,
            "answer": a,
            "options": options
        })
    
    return out

# Streamlit UI
def main():
    st.title("⚡ Fast MCQ Generator")
    st.markdown("Generate multiple choice questions quickly with optimized models")
    
    # Load models
    if 'models' not in st.session_state:
        with st.spinner("Loading lightweight models for fast performance..."):
            models = load_models()
            if models is None:
                st.error("❌ Failed to load models. The app cannot continue.")
                return
            st.session_state.models = models
        st.success("✅ Lightweight models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        st.header("Quick Examples")
        example_texts = {
            "Science": """Photosynthesis converts light energy to chemical energy. Plants use chlorophyll in chloroplasts. The process needs carbon dioxide and water. It produces glucose and oxygen. Light reactions make ATP and NADPH. The Calvin cycle makes sugars.""",
            "History": """The Industrial Revolution started in Britain. Inventions included the steam engine and spinning jenny. Factories grew and cities expanded. Transportation improved with railways."""
        }
        
        selected = st.selectbox("Choose Example", list(example_texts.keys()))
        if st.button("Use Example"):
            st.session_state.input_text = example_texts[selected]
    
    # Main input
    input_text = st.text_area(
        "Enter your text:",
        height=150,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste your text here... (Keep it concise for faster processing)"
    )
    
    # Generate button
    if st.button("🚀 Generate MCQs", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text")
            return
        
        with st.spinner("Generating questions quickly..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        if results['questions']:
            st.success(f"✨ Generated {len(results['questions'])} questions!")
            
            for i, q in enumerate(results['questions'], 1):
                st.markdown(f"**{i}. {q['question']}**")
                
                # Display options in columns
                col1, col2 = st.columns(2)
                for j, opt in enumerate(q['options']):
                    with col1 if j % 2 == 0 else col2:
                        is_correct = opt == q['answer']
                        emoji = "✅" if is_correct else "⚪"
                        st.markdown(f"{emoji} **{chr(65+j)}.** {opt}")
                
                st.divider()
        else:
            st.info("🔍 No questions generated. Try different text.")

if __name__ == "__main__":
    main()
