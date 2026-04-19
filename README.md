
# 🧬 BioMedRag – Advanced Medical Q&A System

BioMedRag is a state-of-the-art **Retrieval-Augmented Generation (RAG)** system designed specifically for the medical and biomedical domain. It leverages multiple retrieval strategies and high-performance LLMs to provide accurate, cited answers to complex medical queries using a vast corpus of PubMed abstracts.

---

## ✨ Key Enhancements & Features

We have significantly upgraded the core architecture to move from a research prototype to a production-ready application:

- **🎨 Modern interactive UI**: A custom-styled Streamlit dashboard featuring dark mode, session history, and real-time performance metrics.
- **🤖 Multi-Cloud LLM Engine**: Native support for **Google Gemini**, **Groq (LPU)**, and **OpenAI**, allowing you to switch providers seamlessly via configuration.
- **🔍 Advanced Retrieval Suite**:
  - **BM25**: Ultra-fast lexical keyword searching.
  - **DPR (Dense Passage Retrieval)**: Semantic vector search using **FAISS**.
  - **Hybrid**: Optimized two-stage retrieval with neural reranking.
- **🔖 Automatic Citations**: Every answer includes a list of used **PMIDs (PubMed IDs)**, ensuring all medical claims are verifiable.
- **🧬 Medical-Grade Encoders**: Integration with **MedCPT** and **BioBERT** for superior understanding of biomedical terminology.

---

## 🏗️ Project Structure

```plaintext
├── streamlit_app.py           # Main entry point (Modern Web UI)
├── rag_system                 # Core RAG Orchestrator
│   ├── med_rag.py             # Main pipeline logic
│   ├── groq_chat.py           # Groq LPU integration
│   ├── gemini_chat.py         # Google Gemini integration
│   ├── openAI_chat.py         # OpenAI integration
│   ├── bm25_retriever.py      # Lexical search (BM25)
│   ├── dpr_retriever.py       # Semantic search (DPR)
│   └── hybrid_retriever.py    # Hybrid reranking logic
├── information_retrieval      # Data Infrastructure
│   ├── elastic_container      # Elasticsearch setup & ingestion
│   ├── faiss_container        # FAISS vector DB setup
│   └── document_encoding      # medCPT/bioBERT encoding scripts
├── evaluation                 # Benchmarking & QA Evaluation
├── sample_data                # Medical dataset subsets
├── smoke_test_groq.py         # Provider verification script
├── .env.example               # Configuration template
└── requirements.txt           # Python dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Daksha10/BioMedRag.git
cd BioMedRag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` to select your preferred provider:
- `LLM_PROVIDER`: Set to `groq`, `gemini`, or `openai`.
- Add the corresponding API key (`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`).

### 3. Start Infrastructure

Use the provided scripts to start Elasticsearch (required for BM25 and Hybrid search):

```bash
cd information_retrieval/elastic_container
bash start_elasticsearch.sh
```

### 4. Run the Application

Launch the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

---

## 🔍 Retrieval Methods

| Method | Component | Logic | Ideal For |
| :--- | :--- | :--- | :--- |
| **BM25** | `bm25_retriever` | Key-term frequency matching via Elasticsearch. | Fast, keyword-heavy queries. |
| **DPR** | `dpr_retriever` | Semantic vector similarity via FAISS. | Concept-based, nuanced queries. |
| **Hybrid** | `hybrid_retriever` | Combined BM25 + Cross-Encoder Reranking. | Maximum accuracy & relevance. |

---

## Data

- **Dataset**: We use a subset of **2.4M PubMed abstracts**.

---