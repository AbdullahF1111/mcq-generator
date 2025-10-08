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

---

## ⚙️ Key Components

| Component | Model / Library | Purpose |
|------------|----------------|----------|
| **Question Generation** | `lmqg/t5-base-squad-qg` | Generates question–answer pairs |
| **Answer Extraction** | `deepset/roberta-base-squad2` | Validates and extracts accurate answer spans |
| **Distractor Generation** | `google/flan-t5-base`, `Sentence-BERT` | Produces contextually similar but incorrect options |
| **Semantic Filtering** | `spacy`, cosine similarity | Ensures diversity and avoids repetition |
| **Web App** | `Streamlit` | Interactive user interface for testing & demo |

---

## 📸 Streamlit Interface

### ✍️ User Input
Paste or upload any English paragraph to generate questions.

### ⚙️ Generated MCQs
Each question is displayed with one correct answer and three distractors.

#### 🧪 Example Output
Q1. Who invented the mechanical clock?
A) Galileo
B) European inventors ✅
C) Newton
D) Pythagoras

> 💡 *Note: Distractors are generated heuristically and may vary in quality (research prototype).*

---

## 🧠 Model Capabilities

- Automatic **question and answer generation** from raw text.
- Semantic **distractor creation** using embeddings and LM suggestions.
- **Answer validation** using contextual QA models.
- Export generated questions in JSON format for reuse.

---


---

## 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/abdullahf1111/mcq-generator.git
cd mcq-generator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run src/app_streamlit.py
```
🌐 Deployment (Optional)

The app is already deployed at:

👉 https://mcq-generator-ubnjdcuymvze6drflrwtvy.streamlit.app/

To deploy your own:

Push your repo to GitHub.

Go to streamlit.io/cloud
.

Choose your repo and select src/app_streamlit.py as the entry file.

🔬 Limitations & Future Work

Distractors are heuristically generated — can be generic or low-quality.

Plan to fine-tune distractor generation using larger QG datasets.

Add multilingual support with mT5 or flan-t5-xl.

Evaluate question quality using BLEU/ROUGE metrics.

Expand to domain-specific MCQs (e.g., medicine, education, history).

👤 Author

Abdullah Fahlo
🎓 B.Sc. in Informatics Engineering — University of Aleppo
📍 Aleppo, Syria
📧 abdullahfahlo.com@gmail.com

💼 LinkedIn

💻 GitHub
