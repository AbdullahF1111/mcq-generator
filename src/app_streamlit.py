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
    page_title="Professional MCQ Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load optimized models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Loading professional-grade models..."):
        qg_pipe = pipeline(
            "text2text-generation",
            model="valhalla/t5-base-qa-qg-hl",
            device=0 if device == "cuda" else -1
        )
        
        qa_pipe = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if device == "cuda" else -1
        )
        
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Use a larger model for better distractor generation
        distractor_gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",  # Upgraded for better quality
            device=0 if device == "cuda" else -1
        )
    
    return {
        'qg_pipe': qg_pipe, 'qa_pipe': qa_pipe, 
        'embedder': embedder, 'distractor_gen': distractor_gen
    }

def grammar_correct_question(question: str) -> str:
    """Fix common grammar issues in generated questions"""
    corrections = {
        r'\bhe\s+found\b': 'he found that',
        r'\bthey\s+found\b': 'they found that',
        r'\bwhat\s+did\s+(\w+)\s+found\b': r'what did \1 find',
        r'\bwhat\s+does\s+(\w+)\s+found\b': r'what does \1 find',
        r'\bhow\s+does\s+(\w+)\s+found\b': r'how does \1 find',
    }
    
    corrected = question
    for pattern, replacement in corrections.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    # Capitalize first letter and ensure it ends with ?
    corrected = corrected.strip()
    if corrected and not corrected[0].isupper():
        corrected = corrected[0].upper() + corrected[1:]
    if not corrected.endswith('?'):
        corrected += '?'
    
    return corrected

def extract_high_quality_entities(text: str) -> List[str]:
    """Extract meaningful, well-formatted entities"""
    entities = set()
    
    # Multi-word capitalized terms (most reliable)
    proper_phrases = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    entities.update(proper_phrases)
    
    # Key technical terms in context
    technical_patterns = [
        r'(?:term|concept|process|phenomenon)\s+(?:of|called)\s+["\']?([^"\',.!?]+)',
        r'(?:known as|called|termed)\s+["\']?([^"\',.!?]+)',
        r'\b([A-Z][a-z]+\s+(?:radiation|energy|process|element|theory))\b',
    ]
    
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = clean_entity(match)
            if clean_match:
                entities.add(clean_match)
    
    # Filter for quality
    quality_entities = []
    for entity in entities:
        if (2 <= len(entity.split()) <= 4 and
            len(entity) >= 4 and
            not entity.isupper() and
            not any(word in entity.lower() for word in ['the', 'a', 'an', 'and', 'or'])):
            quality_entities.append(entity)
    
    return quality_entities

def clean_entity(entity: str) -> str:
    """Clean and format entities properly"""
    if not entity:
        return ""
    
    # Remove leading/trailing junk
    entity = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', entity.strip())
    
    # Remove common prefixes
    entity = re.sub(r'^(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)
    
    # Proper title case for multi-word entities
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

def generate_professional_distractors(question: str, correct_answer: str, context: str, num: int = 3) -> List[str]:
    """Generate professional-quality distractors"""
    
    # Get high-quality entities
    entities = extract_high_quality_entities(context)
    entities = [e for e in entities if e.lower() != correct_answer.lower()]
    
    # Analyze question type for targeted distractor generation
    question_lower = question.lower()
    
    if re.search(r'\b(discover|find|identify)\b', question_lower):
        # For discovery questions - use other discoveries/findings
        distractors = generate_discovery_distractors(correct_answer, entities, context)
    
    elif re.search(r'\b(call|term|name)\b', question_lower):
        # For naming questions - use other named concepts
        distractors = generate_naming_distractors(correct_answer, entities)
    
    elif re.search(r'\b(process|phenomenon|mechanism)\b', question_lower):
        # For process questions - use other processes
        distractors = generate_process_distractors(correct_answer, entities)
    
    else:
        # General case - semantic similarity
        distractors = get_semantic_distractors_pro(correct_answer, entities, num * 2)
    
    # Enhance with LM if needed
    if len(distractors) < num:
        lm_distractors = generate_lm_distractors_pro(question, correct_answer, context)
        distractors.extend([d for d in lm_distractors if d not in distractors])
    
    # Final quality control
    return apply_quality_control(distractors, correct_answer, question, num)

def generate_discovery_distractors(correct_answer: str, entities: List[str], context: str) -> List[str]:
    """Generate distractors for discovery/finding questions"""
    distractors = []
    
    # Look for other discoveries in context
    discovery_indicators = ['discovered', 'found', 'identified', 'observed', 'detected']
    sentences = re.split(r'[.!?]+', context)
    
    for sentence in sentences:
        for indicator in discovery_indicators:
            if indicator in sentence.lower():
                # Extract potential discoveries after the indicator
                pattern = f"{indicator}\\s+([^.,!?]+)"
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    clean_match = clean_entity(match)
                    if (clean_match and 
                        clean_match.lower() != correct_answer.lower() and
                        clean_match not in distractors):
                        distractors.append(clean_match)
    
    # Add relevant entities as fallback
    for entity in entities:
        if (entity not in distractors and 
            len(distractors) < 6 and
            entity.lower() != correct_answer.lower()):
            distractors.append(entity)
    
    return distractors[:4]

def generate_naming_distractors(correct_answer: str, entities: List[str]) -> List[str]:
    """Generate distractors for naming questions"""
    # Use other named entities from context
    named_entities = [e for e in entities if len(e.split()) >= 2]
    
    # Add common scientific terms if needed
    scientific_terms = ["Chemical Process", "Biological Mechanism", "Physical Phenomenon", "Atomic Reaction"]
    
    distractors = named_entities + scientific_terms
    distractors = [d for d in distractors if d.lower() != correct_answer.lower()]
    
    return list(dict.fromkeys(distractors))[:4]  # Remove duplicates

def generate_process_distractors(correct_answer: str, entities: List[str]) -> List[str]:
    """Generate distractors for process questions"""
    process_terms = [
        "Nuclear Fission", "Chemical Synthesis", "Biological Degradation",
        "Energy Transfer", "Molecular Binding", "Cell Division"
    ]
    
    # Combine context entities with general process terms
    all_terms = entities + process_terms
    distractors = [term for term in all_terms if term.lower() != correct_answer.lower()]
    
    return list(dict.fromkeys(distractors))[:4]

def get_semantic_distractors_pro(answer: str, candidates: List[str], num: int) -> List[str]:
    """Professional semantic distractor selection"""
    if not candidates:
        return []
    
    try:
        sbert = st.session_state.models['embedder']
        cand_emb = sbert.encode(candidates, convert_to_tensor=True)
        ans_emb = sbert.encode([answer], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(cand_emb, ans_emb).squeeze(1).cpu().numpy()
        
        # Select candidates with optimal similarity (0.4-0.7)
        scored_candidates = []
        for i, sim in enumerate(sims):
            if 0.4 <= sim <= 0.7:
                scored_candidates.append((candidates[i], sim))
        
        # Sort by similarity and take best ones
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_candidates[:num]]
    
    except:
        return candidates[:num]

def generate_lm_distractors_pro(question: str, answer: str, context: str) -> List[str]:
    """Professional LM distractor generation"""
    try:
        prompt = f"""Context: "{context[:400]}"

Question: "{question}"
Correct Answer: "{answer}"

Generate 3 professional, plausible distractors that:
- Are scientifically/technically relevant
- Are 2-4 words each
- Sound realistic but are factually incorrect
- Are grammatically correct and well-formatted
- Cover different aspects of the topic

Examples of good distractors:
- "Chemical synthesis" instead of "photosynthesis"
- "Nuclear fusion" instead of "nuclear fission"
- "Alpha particles" instead of "beta particles"

Distractors:"""
        
        result = st.session_state.models['distractor_gen'](
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        generated = result[0]['generated_text']
        
        # Sophisticated extraction
        distractors = []
        lines = [line.strip() for line in generated.split('\n') if line.strip()]
        
        for line in lines:
            # Remove numbering, bullets, and quotes
            clean_line = re.sub(r'^[\d\-•\*"\'\)\.]\s*', '', line)
            clean_line = clean_entity(clean_line)
            
            if (clean_line and 
                2 <= len(clean_line.split()) <= 4 and
                clean_line.lower() != answer.lower() and
                len(clean_line) >= 6 and
                clean_line not in distractors):
                distractors.append(clean_line)
        
        return distractors[:3]
        
    except Exception as e:
        return []

def apply_quality_control(distractors: List[str], answer: str, question: str, num: int) -> List[str]:
    """Apply final quality control"""
    if not distractors:
        return ["Molecular Synthesis", "Biological Process", "Chemical Reaction"][:num]
    
    filtered = []
    question_words = set(question.lower().split())
    
    for distractor in distractors:
        d_lower = distractor.lower()
        
        # Skip if too similar to answer or question
        if (d_lower == answer.lower() or
            answer.lower() in d_lower or
            d_lower in answer.lower() or
            any(q_word in d_lower for q_word in question_words if len(q_word) > 4)):
            continue
        
        # Skip if too generic or poorly formatted
        if (len(distractor) < 4 or
            distractor.isupper() or
            distractor.islower() or
            any(word in d_lower for word in ['option', 'choice', 'answer'])):
            continue
            
        filtered.append(distractor)
    
    # Ensure diversity and professional quality
    final = []
    for distractor in filtered:
        # Check if this distractor adds diversity
        if not any(similar_distractor(distractor, existing) for existing in final):
            final.append(distractor)
        if len(final) >= num:
            break
    
    # Professional fallbacks if needed
    professional_fallbacks = {
        'science': ["Molecular Synthesis", "Chemical Process", "Biological Mechanism"],
        'history': ["Earlier Discovery", "Related Finding", "Alternative Theory"],
        'general': ["Primary Method", "Secondary Process", "Alternative Approach"]
    }
    
    while len(final) < num:
        for fb in professional_fallbacks['science']:
            if fb not in final and len(final) < num:
                final.append(fb)
    
    return final[:num]

def similar_distractor(d1: str, d2: str) -> bool:
    """Check if two distractors are too similar"""
    words1 = set(d1.lower().split())
    words2 = set(d2.lower().split())
    common_words = words1.intersection(words2)
    return len(common_words) >= 2  # Too similar if they share 2+ words

def generate_qa_pairs_pro(context: str, num_questions: int = 3) -> List[tuple]:
    """Professional QA pair generation"""
    qa_pairs = []
    
    try:
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 10]  # Longer sentences
        
        for sentence in sentences[:num_questions * 3]:
            # Enhanced prompt for professional questions
            prompt = f"Generate a clear, grammatically correct question about this: {sentence}"
            
            result = st.session_state.models['qg_pipe'](
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
            
            raw_question = clean_text_generated(result[0]['generated_text'])
            question = grammar_correct_question(raw_question)
            
            if question and '?' in question and len(question.split()) >= 5:
                try:
                    qa_result = st.session_state.models['qa_pipe'](
                        question=question,
                        context=context,
                        top_k=3,
                        max_answer_len=50
                    )
                    
                    best_answer = None
                    for ans in qa_result:
                        answer_text = clean_entity(ans['answer'].strip())
                        score = ans['score']
                        
                        if (score > 0.4 and  # Higher threshold
                            answer_text and
                            1 <= len(answer_text.split()) <= 4 and
                            answer_text.lower() in context.lower()):
                            best_answer = answer_text
                            break
                    
                    if best_answer:
                        qa_pairs.append((question, best_answer))
                        
                except Exception:
                    continue
            
            if len(qa_pairs) >= num_questions:
                break
                
    except Exception as e:
        st.error(f"Generation error: {e}")
    
    return qa_pairs

def clean_text_generated(txt: str) -> str:
    """Clean generated text"""
    if not txt:
        return ""
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'^[\'"]|[\'"]$', '', txt)
    return txt

def generate_mcqs_from_text(context: str, num_questions: int = 3) -> Dict[str, Any]:
    """Main function with professional standards"""
    out = {"questions": []}
    
    qa_pairs = generate_qa_pairs_pro(context, num_questions)
    
    if not qa_pairs:
        st.info("🎯 Try providing more detailed, technical text for better questions.")
        return out
    
    for q, a in qa_pairs:
        # Generate professional distractors
        distractors = generate_professional_distractors(q, a, context, 3)
        
        # Format answer consistently
        formatted_answer = clean_entity(a)
        
        # Shuffle options
        options = distractors + [formatted_answer]
        random.shuffle(options)
        
        out["questions"].append({
            "question": q,
            "answer": formatted_answer,
            "options": options,
            "quality": "professional"
        })
    
    return out

# Streamlit UI
def main():
    st.title("🎓 Professional MCQ Generator")
    st.markdown("Generate high-quality, grammatically correct multiple choice questions")
    
    if 'models' not in st.session_state:
        models = load_models()
        if models is None:
            return
        st.session_state.models = models
        st.success("✅ Professional models loaded!")
    
    with st.sidebar:
        st.header("Settings")
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        st.header("Professional Examples")
        example_texts = {
            "Radioactivity": """The discovery of radioactivity began in 1896 when Henri Becquerel found that uranium salts emitted rays that could darken photographic plates. Marie Curie later coined the term "radioactivity" and discovered the radioactive elements polonium and radium. This process involves unstable atomic nuclei releasing radiation and heat as they decay. There are three main types of radiation: alpha particles, beta particles, and gamma rays. Radioactive decay occurs at predictable rates measured by half-lives.""",
            
            "Photosynthesis": """Photosynthesis is the biological process that converts light energy into chemical energy. Plants utilize chlorophyll pigments in chloroplasts to capture sunlight. The process requires carbon dioxide and water as inputs, producing glucose and oxygen as outputs. The light-dependent reactions occur in thylakoid membranes and generate ATP and NADPH. The Calvin cycle in the stroma uses these products to synthesize organic compounds."""
        }
        
        selected = st.selectbox("Choose Example", list(example_texts.keys()))
        if st.button("Load Professional Example"):
            st.session_state.input_text = example_texts[selected]
    
    input_text = st.text_area(
        "Enter detailed, technical text:",
        height=200,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste detailed scientific, historical, or technical text here..."
    )
    
    if st.button("🎯 Generate Professional MCQs"):
        if not input_text.strip():
            st.warning("Please enter detailed text for best results")
            return
        
        with st.spinner("Generating professional-quality questions..."):
            results = generate_mcqs_from_text(input_text, num_questions)
        
        if results['questions']:
            st.success(f"🎉 Generated {len(results['questions'])} professional questions!")
        else:
            st.info("💡 Tip: Use more detailed text with clear concepts and proper names")
        
        for i, q in enumerate(results['questions'], 1):
            st.markdown(f"**{i}. {q['question']}**")
            
            cols = st.columns(2)
            for j, opt in enumerate(q['options']):
                with cols[j % 2]:
                    is_correct = opt == q['answer']
                    emoji = "✅" if is_correct else "⚪"
                    st.markdown(f"{emoji} **{chr(65+j)}.** {opt}")
            
            st.divider()

if __name__ == "__main__":
    main()
