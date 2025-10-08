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
    page_title="Smart MCQ Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load models with better error handling"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Loading enhanced models..."):
        # More reliable question generation model
        qg_pipe = pipeline(
            "text2text-generation",
            model="valhalla/t5-base-qa-qg-hl",  # Better for QG
            device=0 if device == "cuda" else -1
        )
        
        # More accurate QA model
        qa_pipe = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",  # More reliable
            device=0 if device == "cuda" else -1
        )
        
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if device == "cuda" else -1
        )
    
    return {
        'qg_pipe': qg_pipe, 'qa_pipe': qa_pipe, 
        'embedder': embedder, 'distractor_gen': distractor_gen
    }

def validate_qa_pair(question: str, answer: str, context: str) -> bool:
    """Validate if the QA pair makes sense"""
    if not question or not answer:
        return False
    
    # Basic quality checks
    if len(question.split()) < 4 or len(answer.split()) > 5:
        return False
    
    # Answer should be in context
    if answer.lower() not in context.lower():
        return False
    
    # Question should make sense (not be too generic)
    generic_patterns = [
        r'what is the', r'what are the', r'who is the', 
        r'what was the', r'what were the'
    ]
    
    question_lower = question.lower()
    if any(re.search(pattern, question_lower) for pattern in generic_patterns):
        # For generic questions, ensure answer is specific
        if len(answer.split()) == 1 and answer.isalpha():
            return False
    
    return True

def extract_entities_and_concepts(text: str) -> List[str]:
    """Extract meaningful entities using multiple strategies"""
    entities = set()
    
    # Proper nouns (capitalized phrases)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(proper_nouns)
    
    # Technical terms (words that appear multiple times)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    from collections import Counter
    word_freq = Counter(words)
    important_terms = [word.title() for word, count in word_freq.items() 
                      if count > 1 and len(word) > 3]
    entities.update(important_terms)
    
    # Key phrases after "called", "known as", "termed"
    patterns = [
        r'(?:called|known as|termed|named)\s+([^.,;!?]+)',
        r'(?:process|concept|phenomenon|theory)\s+of\s+([^.,;!?]+)',
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # Multi-word capitalized terms
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            clean_match = re.sub(r'^(the|a|an)\s+', '', match.strip())
            if len(clean_match.split()) <= 3:
                entities.add(clean_match.title())
    
    return [e for e in entities if 1 <= len(e.split()) <= 3]

def generate_high_quality_distractors(question: str, correct_answer: str, context: str, num: int = 3) -> List[str]:
    """Generate much better distractors"""
    
    # Get all potential entities from context
    all_entities = extract_entities_and_concepts(context)
    
    # Remove correct answer
    candidates = [e for e in all_entities if e.lower() != correct_answer.lower()]
    
    # Categorize question type for better distractor selection
    question_lower = question.lower()
    
    if re.search(r'\b(what|which)\b.*\b(called|named|termed)\b', question_lower):
        # For "what is X called" questions - use other named entities
        distractors = [c for c in candidates if c != correct_answer][:num*2]
    
    elif re.search(r'\b(who)\b', question_lower):
        # For "who" questions - use other person names
        distractors = [c for c in candidates if len(c.split()) >= 2][:num*2]
    
    elif re.search(r'\b(when)\b', question_lower):
        # For "when" questions
        distractors = ["19th century", "20th century", "recently", "in the past"]
    
    else:
        # General case - use semantic similarity
        distractors = get_semantic_distractors(correct_answer, candidates, num*2)
    
    # Use LM for final refinement if needed
    if len(distractors) < num:
        lm_distractors = generate_lm_distractors_enhanced(question, correct_answer, context)
        distractors.extend(lm_distractors)
    
    # Final filtering and selection
    return select_best_distractors_enhanced(distractors, correct_answer, question, num)

def generate_lm_distractors_enhanced(question: str, answer: str, context: str) -> List[str]:
    """Enhanced LM distractor generation with better prompting"""
    try:
        prompt = f"""Based on this context: "{context[:500]}"

For the question: "{question}"
The correct answer is: "{answer}"

Generate 3 plausible but incorrect alternatives that:
1. Are related to the topic but clearly wrong
2. Are 1-3 words each
3. Sound realistic but are factually incorrect
4. Are different from each other

Distractors:"""
        
        result = st.session_state.models['distractor_gen'](
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        generated = result[0]['generated_text']
        # Improved extraction
        lines = [line.strip() for line in generated.split('\n') if line.strip()]
        distractors = []
        
        for line in lines:
            # Remove numbering and bullets
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            clean_line = re.sub(r'^[\-\*]\s*', '', clean_line)
            clean_line = clean_text_generated(clean_line)
            
            if (clean_line and 
                1 <= len(clean_line.split()) <= 3 and
                clean_line.lower() != answer.lower() and
                len(clean_line) > 2):
                distractors.append(clean_line)
        
        return distractors[:3]
    except:
        return []

def get_semantic_distractors(answer: str, candidates: List[str], num: int) -> List[str]:
    """Get semantically appropriate distractors"""
    if not candidates:
        return []
    
    try:
        sbert = st.session_state.models['embedder']
        cand_emb = sbert.encode(candidates, convert_to_tensor=True)
        ans_emb = sbert.encode([answer], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
        
        # Select candidates with good semantic relationship
        selected = []
        for i, sim in enumerate(sims):
            if 0.4 <= sim <= 0.8:  # Tighter similarity range
                selected.append((candidates[i], sim))
        
        # Sort by similarity and take best ones
        selected.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in selected[:num]]
    
    except:
        return candidates[:num]

def select_best_distractors_enhanced(distractors: List[str], answer: str, question: str, num: int) -> List[str]:
    """Enhanced distractor selection"""
    if not distractors:
        return ["Alternative method", "Different approach", "Other technique"][:num]
    
    # Remove poor quality distractors
    filtered = []
    question_words = set(question.lower().split())
    
    for distractor in distractors:
        d_lower = distractor.lower()
        
        # Skip if too similar to answer or question
        if (d_lower == answer.lower() or 
            any(qw in d_lower for qw in question_words if len(qw) > 3) or
            len(distractor) < 2):
            continue
        
        # Skip generic words
        generic_terms = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        if all(word in generic_terms for word in d_lower.split()):
            continue
            
        filtered.append(distractor)
    
    # Ensure diversity
    final = []
    for distractor in filtered:
        if not any(util in distractor.lower() for util in final):
            final.append(distractor)
        if len(final) >= num:
            break
    
    # Fill remaining slots if needed
    while len(final) < num:
        fallback = f"Option {len(final) + 1}"
        if fallback not in final:
            final.append(fallback)
    
    return final[:num]

def clean_text_generated(txt: str) -> str:
    """Clean generated text"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'^[\'"]|[\'"]$', '', txt)
    return txt

def generate_qa_pairs_enhanced(context: str, num_questions: int = 3) -> List[tuple]:
    """Enhanced QA pair generation with validation"""
    qa_pairs = []
    
    try:
        # Use highlight-based question generation
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 8]
        
        for sentence in sentences[:num_questions * 3]:  # Try more sentences
            # Enhanced prompt for better questions
            prompt = f"Generate a specific question about: {sentence}"
            
            result = st.session_state.models['qg_pipe'](
                prompt,
                max_length=80,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )
            
            question = clean_text_generated(result[0]['generated_text'])
            
            if question and '?' in question and len(question.split()) >= 4:
                # Get answer with validation
                try:
                    qa_result = st.session_state.models['qa_pipe'](
                        question=question,
                        context=context,
                        top_k=3  # Get multiple possible answers
                    )
                    
                    # Try to find the best answer
                    best_answer = None
                    for ans in qa_result:
                        answer_text = ans['answer'].strip()
                        score = ans['score']
                        
                        if (score > 0.3 and  # Higher confidence threshold
                            validate_qa_pair(question, answer_text, context)):
                            best_answer = answer_text
                            break
                    
                    if best_answer:
                        qa_pairs.append((question, best_answer))
                        
                except Exception as e:
                    continue
            
            if len(qa_pairs) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"Error in QA generation: {e}")
    
    return qa_pairs

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main function with enhanced validation"""
    out = {"questions": []}
    
    qa_pairs = generate_qa_pairs_enhanced(context, num_questions)
    
    if not qa_pairs:
        st.info("🔍 No valid questions generated. Try more detailed text.")
        return out
    
    for q, a in qa_pairs:
        # Generate high-quality distractors
        distractors = generate_high_quality_distractors(q, a, context, 3)
        
        # Shuffle options
        options = distractors + [a]
        random.shuffle(options)
        
        out["questions"].append({
            "question": q,
            "answer": a,
            "options": options,
            "validated": True
        })
    
    return out

# Streamlit UI
def main():
    st.title("🧠 Smart MCQ Generator")
    st.markdown("Generate high-quality multiple choice questions with validated answers")
    
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            return
        st.session_state.models = models
        st.success("✅ Enhanced models loaded!")
    
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Questions", 1, 5, 3)
        
        st.header("Better Examples")
        example_texts = {
            "Radioactivity": """The discovery of radioactivity began with Henri Becquerel in 1896. He found that uranium salts emitted rays that could darken photographic plates. Marie Curie later coined the term "radioactivity" and discovered the elements polonium and radium. The process involves unstable atomic nuclei releasing radiation and heat as they decay. There are three main types of radiation: alpha, beta, and gamma rays.""",
            "Photosynthesis": """Photosynthesis is the biological process that converts light energy into chemical energy. Plants use chlorophyll in their chloroplasts to capture sunlight. The process requires carbon dioxide and water, producing glucose and oxygen as byproducts. The light-dependent reactions occur in the thylakoid membranes, while the Calvin cycle takes place in the stroma."""
        }
        
        selected = st.selectbox("Examples", list(example_texts.keys()))
        if st.button("Load Example"):
            st.session_state.input_text = example_texts[selected]
    
    input_text = st.text_area(
        "Enter detailed text:",
        height=200,
        value=st.session_state.get('input_text', '')
    )
    
    if st.button("🎯 Generate Smart MCQs"):
        if not input_text.strip():
            st.warning("Please enter text")
            return
        
        with st.spinner("Generating validated questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        st.subheader(f"Validated Questions ({len(results['questions'])})")
        
        for i, q in enumerate(results['questions'], 1):
            st.markdown(f"{i}. {q['question']}")
            
            for j, opt in enumerate(q['options']):
                is_correct = opt == q['answer']
                emoji = "✅" if is_correct else "⚪️"
                st.write(f"{emoji} {chr(65+j)}. {opt}")
            
            st.divider()
if name == "main":
   main()
