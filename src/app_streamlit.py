import json
import os
import streamlit as st
from src.mcq_pipeline_final_v2 import generate_mcqs_from_text
import spacy
from spacy.cli import download

# ----------------------------
# Load spaCy model safely
# ----------------------------
try:
    spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")

# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="MCQ Generator", layout="wide")
st.title("🧠 Automatic MCQ Generator")
st.markdown(
    "Paste any English text below and click **Generate MCQs** "
    "to create multiple-choice questions automatically using NLP models."
)

# ----------------------------
# User Input
# ----------------------------
user_text = st.text_area("✍️ Enter your text here:", height=200)
num_questions = st.slider("Number of questions to generate:", 1, 10, 5)

# ----------------------------
# Generate MCQs
# ----------------------------
if st.button("🚀 Generate MCQs"):
    if not user_text.strip():
        st.warning("Please enter some text before generating questions.")
    else:
        with st.spinner("Generating questions... please wait ⏳"):
            results = generate_mcqs_from_text(
                user_text, num_questions=num_questions, desired_distractors=3
            )

        if not results.get("questions"):
            st.error("❌ No questions could be generated. Try a longer or more informative text.")
        else:
            st.success("✅ MCQs generated successfully!")

            # Display questions nicely
            for i, q in enumerate(results["questions"], start=1):
                st.markdown(f"### Q{i}. {q['question']}")
                for opt in q["options"]:
                    st.write(f"- {opt}")
                st.write(f"**Answer:** ✅ {q['answer']}")
                st.write("---")
