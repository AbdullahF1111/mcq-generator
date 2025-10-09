
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
    page_title="MCQ Generator",
    page_icon="❓",
    layout="wide"
)

# Cache models
@st.cache_resource
def load_models():
    """Load all required models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Loading models... This may take a few minutes"):
        # Question generation model
        qg_pipe = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-question-generation-ap",
            device=0 if device == "cuda" else -1
        )
        
        # QA model
        qa_pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if device == "cuda" else -1
        )
        
        # Embedding model
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Distractor generation model
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if device == "cuda" else -1
        )
    
    return {
        'qg_pipe': qg_pipe,
        'qa_pipe': qa_pipe,
        'embedder': embedder,
        'distractor_gen': distractor_gen,
        'device': device
    }

# Helper functions
def clean_text_generated(txt: str) -> str:
    """Clean generated text"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    # Remove quotes and other artifacts
    txt = re.sub(r'^[\'"]|[\'"]$', '', txt)
    return txt

def extract_key_phrases(context: str) -> List[str]:
    """Extract meaningful phrases from context using multiple strategies"""
    phrases = set()
    
    # Strategy 1: Extract capitalized phrases (proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
    phrases.update(proper_nouns)
    
    # Strategy 2: Extract noun phrases after articles
    noun_phrases = re.findall(r'\b(?:the|a|an)\s+([a-z]+\s+[a-z]+|[a-z]+)', context.lower())
    noun_phrases = [np.title() for np in noun_phrases]
    phrases.update(noun_phrases)
    
    # Strategy 3: Extract years and numbers
    years = re.findall(r'\b\d{4}\b', context)
    phrases.update(years)
    
    # Strategy 4: Extract phrases in parentheses or quotes
    quoted = re.findall(r'["\']([^"\']+)["\']', context)
    parenthetical = re.findall(r'\(([^)]+)\)', context)
    phrases.update(quoted)
    phrases.update(parenthetical)
    
    # Strategy 5: Extract key technical terms (words that appear multiple times)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', context.lower())
    from collections import Counter
    word_freq = Counter(words)
    important_words = [word.title() for word, count in word_freq.items() 
                      if count > 1 and len(word) > 3]
    phrases.update(important_words)
    
    # Filter and clean
    filtered_phrases = []
    for phrase in phrases:
        clean_phrase = clean_text_generated(phrase)
        if (clean_phrase and 
            1 <= len(clean_phrase.split()) <= 3 and
            len(clean_phrase) < 30 and
            not clean_phrase.isdigit()):
            filtered_phrases.append(clean_phrase)
    
    return list(set(filtered_phrases))

def generate_smart_distractors(question: str, answer: str, context: str, num_distractors: int = 3) -> List[str]:
    """Generate high-quality distractors using multiple strategies"""
    
    # Get candidate phrases from context
    candidates = extract_key_phrases(context)
    
    # Remove the correct answer
    candidates = [c for c in candidates if c.lower() != answer.lower()]
    
    # Strategy 1: Semantic similarity
    semantic_distractors = get_semantic_distractors(answer, candidates, num_distractors)
    
    # Strategy 2: Pattern-based distractors
    pattern_distractors = get_pattern_based_distractors(question, answer, context)
    
    # Strategy 3: LM-generated distractors
    lm_distractors = get_lm_distractors(question, answer)
    
    # Combine all strategies
    all_distractors = list(set(semantic_distractors + pattern_distractors + lm_distractors))
    
    # Remove duplicates and poor quality distractors
    filtered_distractors = filter_distractors(all_distractors, answer, question)
    
    # Final selection
    final_distractors = select_best_distractors(filtered_distractors, answer, num_distractors)
    
    return final_distractors

def get_semantic_distractors(answer: str, candidates: List[str], num: int) -> List[str]:
    """Select distractors based on semantic similarity"""
    if not candidates:
        return []
    
    try:
        sbert = st.session_state.models['embedder']
        cand_emb = sbert.encode(candidates, convert_to_tensor=True)
        ans_emb = sbert.encode([answer], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
        
        # Select candidates with moderate similarity (0.3-0.7)
        selected = []
        for i, sim in enumerate(sims):
            if 0.3 <= sim <= 0.7:
                selected.append(candidates[i])
            if len(selected) >= num * 2:  # Get extra for filtering
                break
        
        return selected
    except:
        return candidates[:num]

def get_pattern_based_distractors(question: str, answer: str, context: str) -> List[str]:
    """Generate distractors based on question patterns"""
    distractors = []
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # For "what was X" questions, find other entities of similar type
    if re.search(r'what (was|is|are)\s+', q_lower):
        # Extract other proper nouns from context
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
        other_entities = [pn for pn in proper_nouns if pn.lower() != a_lower]
        distractors.extend(other_entities[:2])
    
    # For "when" questions, generate different time periods
    if re.search(r'\bwhen\b', q_lower):
        time_related = ["19th century", "20th century", "last year", "next decade", "recently"]
        distractors.extend(time_related)
    
    # For "who" questions, find other person names
    if re.search(r'\bwho\b', q_lower):
        # Simple pattern for person names (capitalized words that might be names)
        words = re.findall(r'\b[A-Z][a-z]+\b', context)
        potential_names = [w for w in words if len(w) > 2 and w.lower() != a_lower]
        distractors.extend(potential_names[:2])
    
    return distractors[:3]

def get_lm_distractors(question: str, answer: str) -> List[str]:
    """Generate distractors using language model"""
    try:
        prompt = f"""Generate 3 plausible but incorrect alternatives for this question.
Question: {question}
Correct answer: {answer}

Requirements:
- Each alternative should be 1-3 words
- Make them semantically related but clearly wrong
- Return only a comma-separated list

Distractors:"""
        
        result = st.session_state.models['distractor_gen'](
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        
        generated = result[0]['generated_text']
        # Extract distractors from the generated text
        parts = [p.strip() for p in re.split(r'[,\n\.]+', generated) if p.strip()]
        distractors = []
        for part in parts:
            # Clean and validate each distractor
            clean_part = clean_text_generated(part)
            if (clean_part and 
                1 <= len(clean_part.split()) <= 3 and
                clean_part.lower() != answer.lower()):
                distractors.append(clean_part)
        
        return distractors[:3]
    except:
        return []

def filter_distractors(distractors: List[str], answer: str, question: str) -> List[str]:
    """Filter out poor quality distractors"""
    filtered = []
    a_lower = answer.lower()
    q_words = set(question.lower().split())
    
    for distractor in distractors:
        d_lower = distractor.lower()
        
        # Skip if too similar to answer
        if d_lower == a_lower or a_lower in d_lower or d_lower in a_lower:
            continue
            
        # Skip if contains question words
        if any(q_word in d_lower for q_word in q_words if len(q_word) > 3):
            continue
            
        # Skip if too generic
        generic_terms = {"thing", "stuff", "item", "object", "element", "factor"}
        if any(term in d_lower for term in generic_terms):
            continue
            
        # Skip if too long or too short
        if len(distractor) < 2 or len(distractor) > 30:
            continue
            
        filtered.append(distractor)
    
    return filtered

def select_best_distractors(distractors: List[str], answer: str, num: int) -> List[str]:
    """Select the best distractors ensuring diversity"""
    if len(distractors) <= num:
        return distractors
    
    # Prioritize distractors that are semantically diverse
    try:
        sbert = st.session_state.models['embedder']
        dist_emb = sbert.encode(distractors, convert_to_tensor=True)
        
        selected = []
        remaining = list(range(len(distractors)))
        
        while len(selected) < num and remaining:
            # If no selections yet, pick the first one
            if not selected:
                selected.append(remaining.pop(0))
                continue
                
            # Find the most different distractor from already selected ones
            max_min_sim = -1
            best_idx = -1
            
            for i in remaining:
                min_sim = min([util.pytorch_cos_sim(dist_emb[i], dist_emb[j]).item() 
                             for j in selected])
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    best_idx = i
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return [distractors[i] for i in selected[:num]]
    
    except:
        return distractors[:num]

def generate_qa_pairs(context: str, num_questions: int = 3) -> List[tuple]:
    """Generate question-answer pairs"""
    qa_pairs = []
    
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 6]
        
        for sentence in sentences[:num_questions * 2]:
            # Generate question
            prompt = f"generate question: {sentence}"
            result = st.session_state.models['qg_pipe'](
                prompt,
                max_length=64,
                num_return_sequences=1,
                temperature=0.7
            )
            
            question = clean_text_generated(result[0]['generated_text'])
            
            if question and len(question.split()) >= 3 and '?' in question:
                # Find answer using QA model
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
        st.error(f"Error generating questions: {e}")
    
    return qa_pairs

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main function to generate MCQs"""
    out = {"questions": []}
    
    # Generate question-answer pairs
    qa_pairs = generate_qa_pairs(context, num_questions)
    
    if not qa_pairs:
        st.info("⚠️ No questions generated. Try with longer or more detailed text.")
        return out
    
    for q, a in qa_pairs:
        trusted_answer = clean_text_generated(a)
        if not trusted_answer:
            continue
        
        # Generate high-quality distractors
        distractors = generate_smart_distractors(q, trusted_answer, context, num_distractors=3)
        
        # Ensure we have exactly 3 distractors
        while len(distractors) < 3:
            # Add fallback distractors
            fallbacks = ["Alternative method", "Different approach", "Other technique"]
            for fb in fallbacks:
                if fb not in distractors and len(distractors) < 3:
                    distractors.append(fb)
        
        # Shuffle options
        options = distractors + [trusted_answer]
        random.shuffle(options)
        
        out["questions"].append({
            "question": q,
            "answer": trusted_answer,
            "options": options,
            "distractors_used": len(distractors)
        })
    
    return out

# Streamlit UI
def main():
    st.title("🧠 MCQ Generator")
    st.markdown("Enter text to automatically generate high-quality multiple choice questions")
    
    # Load models
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            st.error("❌ Failed to load models. Please check the logs.")
            return
        st.session_state.models = models
        st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of questions", 1, 5, 3)
        
        st.header("Example Texts")
        example_texts = {
            "Technology": "The invention of the dynamo revolutionized electricity generation. Michael Faraday discovered electromagnetic induction in 1831. Thomas Edison later refined the dynamo to create the first commercial electric utility. Nikola Tesla's alternating current system eventually became the standard for power distribution.",
            "Science": "Photosynthesis is the process by which plants convert sunlight into chemical energy. This process occurs in chloroplasts and requires carbon dioxide and water. The light-dependent reactions produce ATP and NADPH, while the Calvin cycle uses these to synthesize glucose.",
            "History": "The Industrial Revolution began in Britain in the late 18th century. Key inventions included the steam engine, spinning jenny, and power loom. This period saw massive urbanization and the growth of factory systems."
        }
        
        selected_example = st.selectbox("Choose example", list(example_texts.keys()))
        if st.button("Use Example"):
            st.session_state.input_text = example_texts[selected_example]
    
    # Text input
    input_text = st.text_area(
        "Enter your text here:",
        height=200,
        value=st.session_state.get('input_text', '')
    )
    
    # Generate button
    if st.button("🎯 Generate MCQs", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text first")
            return
        
        with st.spinner("Generating high-quality questions... This may take a few seconds"):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        # Display results
        st.subheader(f"Generated Questions ({len(results['questions'])} questions)")
        
        if not results['questions']:
            st.info("⚠️ No questions generated. Try with longer or more detailed text.")
            return
        
        for i, q in enumerate(results['questions'], 1):
            with st.container():
                st.markdown(f"### Question {i}")
                st.markdown(f"{q['question']}")
                
                # Display options
                cols = st.columns(2)
                for j, option in enumerate(q['options']):
                    with cols[j % 2]:
                        is_correct = option == q['answer']
                        emoji = "✅" if is_correct else "⚪️"
                        st.markdown(f"{emoji} {chr(65+j)}. {option}")
                
                with st.expander("Answer Details"):
                    st.write(f"Correct Answer: {q['answer']}")
                    st.write(f"Distractors Generated: {q['distractors_used']}")
                
                st.divider()

if __name__ == "__main__":
    main()
