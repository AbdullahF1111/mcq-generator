# app_streamlit.py
import streamlit as st
import json
from mcq_pipeline_final_v2 import generate_mcqs_from_text

st.set_page_config(
    page_title="MCQ Generator", 
    page_icon="❓", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Simple MCQ Generator")
st.markdown("Generate multiple-choice questions from any text")

# نموذج الإدخال
with st.form("mcq_form"):
    text_input = st.text_area(
        "Enter your text:",
        height=150,
        placeholder="Paste your text here...",
        help="The text should be at least 2-3 sentences long for best results."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("Questions to generate", 1, 5, 2)
    with col2:
        num_distractors = st.slider("Options per question", 2, 4, 3)
    
    submitted = st.form_submit_button("Generate Questions", type="primary")

if submitted:
    if not text_input.strip():
        st.error("❌ Please enter some text first!")
    elif len(text_input.split()) < 20:
        st.warning("⚠️ For better results, please provide longer text (at least 20 words).")
    
    with st.spinner("🔄 Generating questions... This may take a moment."):
        try:
            results = generate_mcqs_from_text(
                context=text_input,
                num_questions=num_questions,
                desired_distractors=num_distractors,
                verbose=False
            )
            
            if results["questions"]:
                st.success(f"✅ Generated {len(results['questions'])} questions!")
                st.divider()
                
                for i, q in enumerate(results["questions"], 1):
                    with st.container():
                        st.subheader(f"Question {i}")
                        st.markdown(f"**{q['question']}**")
                        
                        # عرض الخيارات
                        for j, option in enumerate(q["options"]):
                            col1, col2 = st.columns([1, 20])
                            with col1:
                                st.write(f"**{chr(65+j)}.**")
                            with col2:
                                if option == q["answer"]:
                                    st.success(f"{option} ✓")
                                else:
                                    st.write(option)
                        
                        st.caption(f"Type: {q['qtype']}")
                        st.divider()
            else:
                st.warning("""
                🤔 No questions could be generated. This could be because:
                - The text is too short or simple
                - No clear entities or facts were found
                - Try using longer, more descriptive text
                """)
                        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please try again with different text.")

# قسم المساعدة
with st.expander("💡 Tips for better results"):
    st.markdown("""
    - **Use descriptive text** with clear facts and entities
    - **Include names, dates, places** for better question generation  
    - **Minimum 3-4 sentences** works best
    - **Avoid very technical or complex** language
    - Example good text:
        *"The Amazon River is the largest river by discharge volume of water in the world. 
        It flows through South America and has a length of approximately 6,400 km. 
        The river system originates in the Andes Mountains of Peru and empties into 
        the Atlantic Ocean near Brazil."*
    """)

# نص تجريبي للاختبار السريع
if not submitted:
    st.divider()
    if st.button("🎯 Load Example Text"):
        example_text = """
        The Great Pyramid of Giza is the oldest and largest of the three pyramids in the Giza pyramid complex in Egypt. 
        It was built as a tomb for the Fourth Dynasty pharaoh Khufu over a 20-year period concluding around 2560 BC. 
        The pyramid is made of limestone blocks and originally stood at 146.6 meters tall. 
        It was the tallest man-made structure in the world for over 3,800 years.
        """
        st.session_state.example_text = example_text
        st.rerun()
