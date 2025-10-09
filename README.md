# 🧠 Automatic MCQ Generator (NLP + Streamlit)

A **research-driven, end-to-end NLP project** that automatically generates **multiple-choice questions (MCQs)** from any English text using transformer-based models such as **T5**, **RoBERTa**, and **Sentence-BERT**.

It demonstrates a **complete natural language processing pipeline** — from text understanding and question generation to distractor creation and semantic filtering — all wrapped in an **interactive Streamlit web app**.

- 👉  **Live Demo:** [Streamlit App](https://mcq-generator-ubnjdcuymvze6drflrwtvy.streamlit.app/) 
- 📂 **Repository:** [GitHub Repo](https://github.com/abdullahf1111/mcq-generator)

---

## 🚀 Project Overview

| Stage | Description |
|--------|--------------|
| **1️⃣ Data Input** | User provides a paragraph or educational text. |
| **2️⃣ Question Generation (QG)** | Transformer models (LMQG or T5-based) generate question–answer pairs. |
| **3️⃣ Answer Validation** | A QA model (`deepset/roberta-base-squad2`) validates answers using contextual evidence. |
| **4️⃣ Distractor Generation** | `Sentence-BERT` + `Flan-T5` generate plausible but incorrect options. |
| **5️⃣ Semantic Filtering** | Filters ensure the distractors are meaningful, short, and diverse. |
| **6️⃣ Streamlit Interface** | Interactive app for text input, generation, and visualization of MCQs. |

---

## ⚙️ Key Components

| Component | Model / Library | Purpose |
|------------|----------------|----------|
| **Question Generation** | `lmqg/t5-base-squad-qg` (research) / `mrm8488/t5-base-finetuned-question-generation-ap` (app) | Generates question–answer pairs |
| **Answer Extraction** | `deepset/roberta-base-squad2` | Validates and extracts accurate answer spans |
| **Distractor Generation** | `google/flan-t5-base`, `Sentence-BERT` | Produces contextually similar but incorrect options |
| **Semantic Filtering** | `spacy`, cosine similarity | Ensures diversity, relevance, and non-overlap |
| **Web Interface** | `Streamlit` | User-friendly interface for interactive demos |

---

## 🧩 Architecture Overview

This project includes **two complementary pipelines**:

| Version | Description | Target |
|----------|--------------|--------|
| **🔬 Research Pipeline (LMQG)** | Uses `lmqg/t5-base-squad-qg` for joint Question–Answer generation and semantic distractor creation. High-quality but computationally heavy. | Research / Colab (GPU) |
| **⚡ Streamlit App Pipeline** | Uses smaller modular models (`T5`, `RoBERTa`, `Flan-T5`) for QG, QA, and distractor generation. Optimized for CPU and deployment on Streamlit Cloud. | Real-time Web App |

---

## 💡 Why Two Pipelines?

Deploying large NLP models like **LMQG** is challenging on limited environments (e.g., Streamlit Cloud).  
Therefore, the app uses a **lightweight modular version** that combines smaller transformer models and heuristic methods to ensure:
- ✅ Fast inference on CPU  
- ✅ Low memory usage  
- ✅ Smooth user experience  

Meanwhile, the **LMQG pipeline** remains as a **research-grade backbone**, demonstrating your capability to build, optimize, and validate complex transformer architectures.

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Core Libraries:** Transformers, Sentence-Transformers, spaCy, Streamlit  
- **ML Models:** T5, RoBERTa, Flan-T5, LMQG, MiniLM  
- **Deployment:** Streamlit Cloud / Local Execution  

---




## 📸 Streamlit Interface

### 🧠 Example Output

**Input:**
> "The invention of the mechanical clock revolutionized timekeeping. Before clocks, people relied on sundials and water clocks. In the 14th century, European inventors created gears and weights to measure time more accurately."

**Generated MCQ:**
- **Question:** When did the invention of the mechanical clock revolutionize timekeeping?  
- **Options:**
  - 14th century ✅  
  - 16th century  
  - 18th century  
  - 12th century

### ✍️ User Input
Paste or upload any English paragraph to generate questions.

### ⚙️ Generated MCQs
Each question is displayed with one correct answer and three distractors.

> 💡 *Note: Distractors are generated heuristically and may vary in quality (research prototype).*

---

## 🧠 Model Capabilities

- Automatic **question and answer generation** from raw text.
- Semantic **distractor creation** using embeddings and LM suggestions.
- **Answer validation** using contextual QA models.
- Export generated questions in JSON format for reuse.

---

### ⚙️ Architecture Comparison — Research Pipeline vs. Streamlit App

This project was developed in **two stages**, showcasing both *research-grade NLP modeling* and *production-ready deployment design*.

---

### 🧠 1. Main Research Pipeline (LMQG-based)
**File:** `notebooks/mcq_generator.ipynb`  
**Environment:** Colab / Local (GPU recommended)

- **Core Model:** `lmqg/t5-base-squad-qg`  
  → An end-to-end **Question Generation (QG)** and **Answer Extraction (AE)** model built on T5.  
  It learns to jointly predict both the *question* and *answer* from the input passage.

- **Other Components:**
  - `deepset/roberta-base-squad2` — QA validation & span refinement.
  - `google/flan-t5-base` — fallback for generating distractors.
  - `sentence-transformers/all-MiniLM-L6-v2` — used for semantic similarity scoring.
  - `spaCy` — linguistic feature extraction and noun-phrase filtering.
  - `keyBERT` & `pdfplumber` — used for optional key-term extraction and text input processing.

- **Pipeline Logic:**
  1. LMQG generates initial Question–Answer pairs.
  2. QA model validates and corrects answer spans.
  3. Noun phrases and named entities are extracted using spaCy.
  4. Semantic filtering and LM-based generation produce distractors.
  5. MCQs are finalized and exported in structured JSON format.

- **Key Strengths:**
  - Produces *coherent and contextually deep questions.*
  - High linguistic accuracy and answer alignment.
  - Demonstrates research-level NLP pipeline engineering.

- **Limitations:**
  - Large model weights (~1.2 GB for LMQG alone).
  - Requires GPU and high RAM → **not suitable for Streamlit Cloud**.
  - Slower inference, but excellent for offline or research use.

---

#### ⚙️ 2. Streamlit App Pipeline (Lightweight Modular Version)
**File:** `src/app_streamlit.py`  
**Environment:** Streamlit Cloud (CPU-friendly, < 500 MB total)

- **Components:**
  - **Question Generation:** `mrm8488/t5-base-finetuned-question-generation-ap`
  - **Answer Extraction:** `deepset/roberta-base-squad2`
  - **Distractor Generation:** `google/flan-t5-base`
  - **Semantic Filtering:** `sentence-transformers/all-MiniLM-L6-v2`

- **Pipeline Logic:**
  1. Text is split into sentences → questions are generated with a smaller T5 model.
  2. The QA model extracts short, relevant answers.
  3. Distractors are generated using a hybrid approach:
     - Context phrase extraction (regex, capitalization, repetition).
     - LM-generated distractors.
     - Semantic similarity scoring for diversity.
  4. The app displays questions interactively with multiple-choice options.

- **Key Strengths:**
  - Modular and optimized for **real-time deployment**.
  - Fast and memory-efficient (runs on CPU).
  - Great for demos, testing, and user interaction.

- **Trade-offs:**
  - Simpler linguistic structure.
  - Slightly lower question quality compared to LMQG.
  - Distractors are more heuristic and pattern-based rather than semantically learned.

---

#### 📊 Summary of Key Differences

| Feature | **LMQG Research Pipeline** | **Streamlit Cloud Pipeline** |
|----------|-----------------------------|-------------------------------|
| Framework | LMQG (end-to-end) | Transformers (modular) |
| QG Model | `lmqg/t5-base-squad-qg` | `mrm8488/t5-base-finetuned-question-generation-ap` |
| Architecture | Joint QG + QA | Separate QG → QA → Distractors |
| Model Size | ~1.2 GB | ~400 MB |
| Context Understanding | Deep semantic reasoning | Sentence-level generation |
| Distractor Generation | LM + semantic filtering | Regex + semantic + LM mix |
| Deployment Target | Research / Colab | Streamlit Cloud |
| Performance | High accuracy, slower | Fast, lightweight |
| Use Case | NLP experimentation & evaluation | Interactive web demo |

---

#### 💡 Design Insight

This two-tier design reflects a **real-world MLOps trade-off**:
- The **LMQG pipeline** shows your ability to build *research-grade* NLP systems with deep context modeling.  
- The **Streamlit pipeline** shows your *engineering adaptability* — creating an efficient, modular version that can run anywhere.

Together, they highlight your strengths as both a **data scientist** and a **practical ML engineer**.


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

### 🌐 Deployment (Optional)

The app is already deployed at:

- 👉 [https://mcq-generator-ubnjdcuymvze6drflrwtvy.streamlit.app/](https://mcq-generator-ubnjdcuymvze6drflrwtvy.streamlit.app/)

To deploy your own:

- Push your repo to GitHub.

- Go to streamlit.io/cloud
.

Choose your repo and select src/app_streamlit.py as the entry file.

### 🔬 Limitations & Future Work

- Distractors are heuristically generated — can be generic or low-quality.
- Plan to fine-tune distractor generation using larger QG datasets & Improve it.
- Add multilingual support with mT5 or flan-t5-xl.
- Evaluate question quality using BLEU/ROUGE metrics.
- Expand to domain-specific MCQs (e.g., medicine, education, history)
- Integrate ranking models for question quality scoring.  
  

👤 Author

- Abdullah Fahlo
- 🎓 B.Sc. in Informatics Engineering — University of Aleppo
- 📍 Aleppo, Syria
- 📧 abdullahfahlo.com@gmail.com
- 💼 LinkedIn
- 💻 GitHub
