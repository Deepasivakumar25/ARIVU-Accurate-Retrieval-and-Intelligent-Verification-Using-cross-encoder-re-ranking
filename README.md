# ARIVU — Accurate Retrieval and Intelligent Verification Using Cross-Encoder Re-Ranking

ARIVU (Accurate Retrieval and Intelligent Verification Using Cross-Encoder Re-Ranking) is an AI-based retrieval system designed to improve the relevance and accuracy of retrieved information. It uses semantic retrieval followed by keyword-based candidate combination and cross-encoder re-ranking to identify the most relevant context for a given query.

## Features

- PDF document ingestion and text extraction
- Text chunking
- Embedding-based semantic retrieval using FAISS
- TF-IDF keyword search
- Hybrid candidate retrieval
- Cross-encoder re-ranking
- Context-aware answer generation using an LLM

## Retrieval Workflow

```text
PDF Document
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Semantic Search + Keyword Search
   ↓
Candidate Combination
   ↓
Cross-Encoder Re-Ranking
   ↓
Relevant Context
   ↓
LLM
   ↓
Generated Answer
```

## Repository Contents

This **ARIVU** repository contains the project in both **Jupyter Notebook (`.ipynb`) and Python (`.py`) formats**.

- **`hybrid_rag_cross_encoder_reranking.ipynb`** — The complete workflow in Jupyter/Google Colab notebook format for step-by-step experimentation and learning.
- **Python files** — The notebook workflow is separated into modular Python files corresponding to the major stages of the implementation.

Both formats are maintained in the repository so the project can be explored interactively through the notebook or used through modular Python code.

## Python Modules

The Python implementation follows the workflow of the notebook:

```text
src/
├── 01_installation.py
├── 02_pdf_upload.py
├── 03_pdf_extraction.py
├── 04_text_chunking.py
├── 05_llm_setup.py
├── 06_semantic_search.py
├── 07_keyword_search.py
├── 08_cross_encoder_reranking.py
├── 09_rag_generation.py
└── 10_hybrid_rag_pipeline.py
```

## Technologies

- Python
- Sentence Transformers
- FAISS
- TF-IDF
- Scikit-learn
- Cross-Encoder
- Hugging Face Transformers
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

## Purpose

The project demonstrates how semantic search and keyword search can be combined to retrieve candidate information, followed by cross-encoder re-ranking to improve retrieval relevance before passing the best context to an LLM.

## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Deepasivakumar25/ARIVU-Accurate-Retrieval-and-Intelligent-Verification-Using-cross-encoder-re-ranking.git
cd ARIVU-Accurate-Retrieval-and-Intelligent-Verification-Using-cross-encoder-re-ranking
pip install -r requirements.txt
```

The notebook can be opened in Google Colab or Jupyter, while the Python modules can be used individually or as part of the complete pipeline.

## Project Status

This repository is part of a hands-on learning project focused on semantic retrieval, hybrid search, cross-encoder re-ranking, and Retrieval-Augmented Generation.

## License

This project is intended for educational and learning purposes.
