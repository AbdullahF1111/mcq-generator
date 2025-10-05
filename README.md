# MCQ Generator Project
This repository contains the source code for the NLP-based MCQ Generator with Semantic Distractor Filtering.
# 🧠 MCQ Generator using Transformers

This project automatically generates **Multiple-Choice Questions (MCQs)** from text using **NLP and transformer-based models**.  
It combines question generation, answer extraction, and distractor creation into one smart pipeline.

---

## 🚀 Features
- Question Generation with `lmqg/t5-base-squad-qg`
- Distractor Generation using `google/flan-t5-base`
- Semantic filtering using `SentenceTransformer`
- Automatic QA validation with `deepset/roberta-base-squad2`

---

## 📁 Project Structure
mcq-generator/
│
├── data/ # Example input texts
│ ├── sample_books.txt
│ ├── sample_climate.txt
│ ├── sample_clocks.txt
│ └── sample_liam.txt
│
├── src/ # Core Python logic
│ └── mcq_pipeline_final_v2.py
│
├── notebooks/ # (Optional) Jupyter/Colab demos
│ └── demo_colab.ipynb
│
├── models/ # Saved tokenizer / QG model
│
├── requirements.txt # Python dependencies
├── .gitignore # Ignore unnecessary files
└── README.md # Project documentation

## 🧠 Model Setup
The required models (`lmqg/t5-base-squad-qg`, `google/flan-t5-base`, and `all-MiniLM-L6-v2`)
are automatically downloaded when you run the notebook or pipeline.

---

## 🧩 Example Usage
```python
from src.mcq_pipeline_final_v2 import generate_mcqs_from_text

context = open("data/sample_climate.txt").read()
results = generate_mcqs_from_text(context, num_questions=3, verbose=True)

for q in results["questions"]:
    print(q["question"])
    for opt in q["options"]:
        print("-", opt)
    print("Correct:", q["answer"])
⚙️ Installation
bash
نسخ الكود
pip install -r requirements.txt
python -m spacy download en_core_web_sm
🧠 Author
Abdullah Fahlo
Data Science & AI Student
📍 Aleppo, Syria
📧 abdullahfahlo.com@gmail.com
🌐 GitHub: AbdullahF1111
