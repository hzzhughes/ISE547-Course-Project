# ISE 547 Course Project

> [Project Report](https://innovai.pro/app/editor/JEdzmHpEhsR3vnMC48Ex)

## Description

This is a course project for ISE547.

This repository contains codes for a small RAG (Retrieval-Augmented Generation) system that ingests Yugioh card and rule data and answers questions using OpenRouter as the LLM backend.

Files added:

- `rag_agent.py` — ingestion, chunking, embeddings (sentence-transformers), FAISS indexing, retrieval, and OpenRouter generation helpers
- `requirements.txt` — Python dependencies
<!-- - `examples/query_example.py` — example script demonstrating build & query flow -->

Environment variables:

- `OPENAI_API_KEY` — required (your OpenRouter API key)
- `OPENAI_BASE_URL` — optional (defaults to `https://openrouter.ai/api/v1`)

Quick start

1. Install dependencies:

```bash
pip install -r assignments/project/requirements.txt
```

2. Update/downloading card data
```bash
git submodule update --init --recursive
```

3. Run rag with a CLI interface (assumes `./data` contains `cards/` and `rules/`):

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
python rag_agent.py
```

<!-- 3. Query example (set your OpenRouter key first). You can customize the generation model and embedding model:

```bash
export OPENROUTER_API_KEY="<your-key>"
# Basic query (uses default models)
python assignments/project/examples/query_example.py --query "What does Monster Reborn do?" --index_dir ../index

# Specify OpenRouter model name and sentence-transformers embedding model
python assignments/project/examples/query_example.py --query "What does Monster Reborn do?" --index_dir ../index --model gpt-4o-mini --embed_model all-MiniLM-L6-v2

# Attempt to use GPU for embeddings/FAISS (best-effort)
python assignments/project/examples/query_example.py --query "What does Monster Reborn do?" --index_dir ../index --use_gpu --embed_model all-MiniLM-L6-v2
``` -->
