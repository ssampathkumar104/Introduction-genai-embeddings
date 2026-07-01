# Introduction-genai-embeddings

Gen AI Data Ingestion, Transformations & Embeddings for OpenAI, Ollama and HuggingFace

## Introduction

This repository provides hands-on, notebook-driven examples for ingesting data, transforming it, splitting it, and creating vector embeddings using multiple GenAI providers — OpenAI, Ollama, and Hugging Face — and persisting/searching them with vector stores such as ChromaDB and FAISS. It's aimed at engineers and researchers who want practical demonstrations of the end-to-end pipeline: from raw documents to searchable embeddings and retrieval-augmented workflows.

Key capabilities covered by the notebooks:
- Data ingestion from PDFs, HTML, XML, and web sources
- Text splitting strategies (character, JSON, HTML-aware, and custom splitters)
- Creating embeddings with OpenAI, Hugging Face, and Ollama models
- Storing and querying embeddings with ChromaDB and FAISS
- Minimal LangChain integration examples to wire LLMs and vector stores together

## Notebooks included

- GettingStarted.ipynb — basic environment setup and LLM sanity checks
- DataIngestion.ipynb — examples for reading PDFs, web pages, and other documents
- DataSplitter.ipynb — demonstration of splitting strategies for long documents
- CharacterSplitter.ipynb — character-based splitting
- HTMLSplitter.ipynb — HTML-aware splitting
- JSONSplitter.ipynb — JSON-aware splitting
- OPENAI_Embeddings.ipynb — how to use OpenAI embeddings
- HuggingFace_Embeddings.ipynb — using Hugging Face Sentence Transformers
- Ollamma_Embeddings.ipynb — examples for Ollama-based embeddings
- ChromaDB.ipynb — storing and querying with ChromaDB
- FAISS.ipynb — storing and querying with FAISS

## Quick setup

1. Clone the repository

```bash
git clone https://github.com/ssampathkumar104/Introduction-genai-embeddings.git
cd Introduction-genai-embeddings
```

2. Install the Python requirements (prefer a virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Fill in your API keys in a `.env` file (examples used in notebooks expect OpenAI / LangChain keys):

```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

## Usage example (from GettingStarted.ipynb)

Below is a minimal example adapted from GettingStarted.ipynb that shows how to load environment variables, initialize a LangChain/OpenAI LLM wrapper, and run a simple prompt. This reproduces the core steps demonstrated in the notebook.

```python
import os
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

# Set environment variables so the notebook/llm code picks them up
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Example: initialize a LangChain OpenAI wrapper and run a query
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
print(llm)

# Invoke the model (not all wrappers return plain dicts — notebooks show how to handle responses)
result = llm.invoke("What is generative AI?")
print(result)
# Note: some example notebooks later access content differently depending on the wrapper; inspect the notebook cells for exact usage (e.g., result["content"]) and handle None safely.
```

## Where to find usage for embeddings & vector stores

- OPENAI_Embeddings.ipynb: concrete examples of creating OpenAI embeddings for a set of documents and inserting them into a vector store.
- HuggingFace_Embeddings.ipynb: shows creating SentenceTransformer embeddings locally and using them for search.
- Ollamma_Embeddings.ipynb: demonstrates using Ollama models for embeddings.
- ChromaDB.ipynb and FAISS.ipynb: walk through persisting vectors, performing similarity search, and retrieving metadata.

Each notebook contains runnable cells and inline comments. If you want a short script instead of a notebook cell, copy the relevant cells and adapt them into a .py file.

## Requirements

See `requirements.txt` for the primary Python dependencies used across the notebooks (langchain, chromadb, faiss-cpu, sentence_transformers, etc.).

## Notes & tips

- Notebooks are written for Python 3.12 in the kernel metadata but should work on other 3.x versions with the listed dependencies.
- Some wrappers return structured objects rather than raw strings; handle return values defensively (check for None before subscripting).
- For large datasets, prefer chunked splitting strategies (see DataSplitter.ipynb) and a vector index optimized for your workload (ChromaDB vs FAISS).

## License

(If you want to add a license, add a LICENSE file to the repo.)

---

If you'd like, I can also:
- Extract and paste specific runnable code blocks from any notebook into separate .py example files,
- Add a short scripted quickstart (single Python script) that reproduces a notebook flow,
- Or create a CONTRIBUTING.md with development and testing steps.
