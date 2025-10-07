# app_streamlit.py
import streamlit as st
import json
from mcq_pipeline_simple import generate_mcqs_from_text

# إعدادات الصفحة
st.set_page_config(
    page_title="Simple MCQ Generator",
    page_icon="❓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# العنوان والوصف
st.title("🧠 MCQ Generator")
st.markdown("Generate multiple-choice questions from any text using AI")

# نموذج الإدخال
with st.form("input_form"):
    text_input = st.text_area(
        "**Enter your text:**",
        height=150,
        placeholder="Paste your text here...\n\nExample: The Eiffel Tower is in Paris, France. It was built in 1889 and is named after engineer Gustave Eiffel.",
        help="For best results, use text with names, dates, places, and facts."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("**Number of questions:**", 1, 5, 2)
    with col2:
        num_options = st.slider("**Options per question:**", 2, 4, 3)
    
    generate_btn = st.form_submit_button("🎯 Generate Questions", type="primary")

# معالجة النتائج
if generate_btn:
    if not text_input.strip():
        st.error("❌ Please enter some text!")
    else:
        with st.spinner("🔄 Analyzing text and generating questions..."):
            try:
                results = generate_mcqs_from_text(
                    context=text_input,
                    num_questions=num_questions,
                    desired_distractors=num_options - 1  # -1 لأن الإجابة الصحيحة تحسب
                )
                
                # عرض النتائج
                if results["status"] == "success" and results["questions"]:
                    st.success(f"✅ Generated {len(results['questions'])} questions from your text!")
                    
                    for i, q in enumerate(results["questions"], 1):
                        with st.container():
                            st.markdown(f"### Question {i}")
                            st.markdown(f"**{q['question']}**")
                            
                            # عرض الخيارات
                            st.markdown("**Options:**")
                            for j, option in enumerate(q["options"]):
                                emoji = "✅" if option == q["answer"] else "⚪"
                                st.markdown(f"{emoji} **{chr(65+j)}.** {option}")
                            
                            st.caption(f"Type: {q['qtype']} • Entities found: {q['meta']['entities_found']}")
                            st.divider()
                
                elif results["status"] == "error":
                    st.warning(f"⚠️ {results['message']}")
                    
                else:
                    st.info("🤔 No questions could be generated. Try using more descriptive text with clear facts.")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Please try again with different text.")

# قسم الأمثلة والنصائح
with st.expander("💡 **Tips & Examples**", expanded=True):
    st.markdown("""
    **For best results, include:**
    - 👤 **People names** (Albert Einstein, Marie Curie)
    - 🗺️ **Places** (Paris, France, Amazon River)  
    - 📅 **Dates & years** (2020, 1995, 19th century)
    - 🏛️ **Organizations** (United Nations, Google)
    - 📚 **Facts & definitions**
    
    **Example text that works well:**
    ```
    The Amazon River is the largest river by discharge volume in the world. 
    It flows through South America and has a length of approximately 6,400 km. 
    The river system originates in the Andes Mountains of Peru and was first 
    explored by Francisco de Orellana in 1542. It empties into the Atlantic Ocean.
    ```
    """)

# زر تحميل مثال
if not generate_btn:
    if st.button("🎯 Load Example Text"):
        example_text = """
        The Great Pyramid of Giza is the oldest and largest of the three pyramids in Egypt. 
        It was built as a tomb for the Pharaoh Khufu around 2560 BC over a 20-year period. 
        The pyramid was constructed by thousands of workers using limestone blocks. 
        It originally stood at 146.6 meters tall and was the tallest structure for 3,800 years.
        The architect was Hemiunu, a relative of Pharaoh Khufu.
        """
        st.session_state.example_text = example_text
        st.rerun()

# إذا كان هناك مثال محمل، عرضه
if 'example_text' in st.session_state:
    text_input = st.session_state.example_text
