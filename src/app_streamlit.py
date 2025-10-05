import streamlit as st
import json
from mcq_pipeline_final_v2 import generate_mcqs_from_text

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="MCQ Generator", layout="wide")
st.title("🧠 Automatic MCQ Generator")
st.markdown("Paste any English text below and click **Generate MCQs** to create automatic questions.")

# User text input
user_text = st.text_area("✍️ Enter your text here:", height=200)

num_questions = st.slider("Number of questions to generate:", 1, 10, 5)

if st.button("🚀 Generate MCQs"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        st.info("Generating questions... please wait ⏳")
        results = generate_mcqs_from_text(user_text, num_questions=num_questions, desired_distractors=3)

        st.success("✅ MCQs generated successfully!")

        # Display the questions
        for i, q in enumerate(results["questions"], 1):
            st.markdown(f"### Q{i}. {q['question']}")
            for j, opt in enumerate(q["options"]):
                st.markdown(f"- {chr(65+j)}. {opt}")
            st.markdown(f"**Correct Answer:** ✅ {q['answer']}")
            st.markdown("---")

        # Save output file
        output_path = "outputs/mcq_output_user.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        st.success(f"Results saved to `{output_path}`")

        st.download_button(
            label="⬇️ Download Generated MCQs (JSON)",
            data=json.dumps(results, ensure_ascii=False, indent=2),
            file_name="mcq_output_user.json",
            mime="application/json"
        )
