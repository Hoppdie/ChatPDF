# Contributing to ChatPDF

Thank you for your interest in contributing! This guide explains how to get started.

## Development Setup

```bash
git clone https://github.com/Hoppdie/ChatPDF.git
cd ChatPDF
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Architecture Overview

ChatPDF uses a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Document Ingestion** — PDFs are loaded via PyPDF2, split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
2. **Embedding & Indexing** — Each chunk is embedded via OpenAI Embeddings and stored in a FAISS vector store
3. **Query Pipeline** — User questions trigger a similarity search; top-k chunks are retrieved and passed as context to the LLM
4. **Response Generation** — GPT-3.5/4 generates a grounded answer with source attribution

## Adding Features

- **New document types**: Extend the loader in `htmlTemplates.py` or add a new ingestion function
- **Different vector stores**: Swap FAISS for Pinecone/Chroma by changing the vector store initialization
- **Alternative LLMs**: Replace the ChatOpenAI call with any LangChain-compatible LLM

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Add docstrings to new functions
- Test with at least 2-3 different PDFs before submitting
- Update `README.md` if you change user-facing behaviour

## Reporting Issues

Open a GitHub Issue with:
- Python version and OS
- The PDF that caused the problem (or a minimal reproduction)
- Full error traceback
