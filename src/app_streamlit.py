# app_streamlit.py
import streamlit as st
import json
from mcq_pipeline_final_v2 import generate_mcqs_from_text

st.set_page_config(page_title="MCQ Generator", page_icon="❓", layout="wide")

st.title("🧠 MCQ Generator from Text")
st.markdown("Generate multiple-choice questions from any text input")

# إدخال النص
text_input = st.text_area(
    "Enter your text here:",
    height=200,
    placeholder="Paste your text here to generate multiple-choice questions..."
)

# الإعدادات
col1, col2 = st.columns(2)
with col1:
    num_questions = st.slider("Number of questions", 1, 10, 5)
with col2:
    num_distractors = st.slider("Distractors per question", 2, 5, 3)

if st.button("Generate MCQs", type="primary"):
    if not text_input.strip():
        st.error("Please enter some text first!")
    else:
        with st.spinner("Generating questions... This may take a few moments."):
            try:
                results = generate_mcqs_from_text(
                    context=text_input,
                    num_questions=num_questions,
                    desired_distractors=num_distractors,
                    verbose=False
                )
                
                if results["questions"]:
                    st.success(f"Generated {len(results['questions'])} questions!")
                    
                    for i, q in enumerate(results["questions"], 1):
                        with st.expander(f"Question {i}: {q['question']}", expanded=True):
                            st.markdown(f"**Question:** {q['question']}")
                            st.markdown("**Options:**")
                            
                            for j, option in enumerate(q["options"]):
                                if option == q["answer"]:
                                    st.markdown(f"✅ **{chr(65+j)}. {option}** (Correct Answer)")
                                else:
                                    st.markdown(f"{chr(65+j)}. {option}")
                            
                            st.markdown(f"**Type:** {q['qtype']}")
                else:
                    st.warning("No questions could be generated from this text. Try a longer or more detailed text.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with different text or settings.")
