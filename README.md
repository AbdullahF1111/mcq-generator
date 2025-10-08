# 🧠 Automatic MCQ Generator (NLP + Streamlit)

A **research-driven, end-to-end NLP project** that automatically generates **multiple-choice questions (MCQs)** from any English text using transformer-based models such as **T5**, **RoBERTa**, and **Sentence-BERT**.

It demonstrates a **complete natural language processing pipeline** — from text understanding and question generation to distractor creation and semantic filtering — all wrapped in an **interactive Streamlit web app**.

👉 **Live Demo:** [Streamlit App](https://mcq-generator-ubnjdcuymvze6drflrwtvy.streamlit.app/)  
📂 **Repository:** [GitHub Repo](https://github.com/abdullahf1111/mcq-generator)

---

## 🚀 Project Overview

| Stage | Description |
|--------|--------------|
| **1️⃣ Data Input** | User provides a paragraph or educational text. |
| **2️⃣ Question Generation (QG)** | Model (`lmqg/t5-base-squad-qg`) generates question–answer pairs. |
| **3️⃣ Answer Validation** | QA model (`deepset/roberta-base-squad2`) validates answers using context. |
| **4️⃣ Distractor Generation** | `Sentence-BERT` + `Flan-T5` generate similar but incorrect options. |
| **5️⃣ Streamlit Interface** | Interactive app for input, generation, and visualization. |

---

## 🧠 Author
- Abdullah Fahlo
- Data Science & AI Student
- 📍 Aleppo, Syria
- 📧 abdullahfahlo.com@gmail.com
- 🌐 GitHub: AbdullahF1111
