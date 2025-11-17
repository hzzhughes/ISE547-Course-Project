#!/usr/bin/env python3
"""Terminal RAG agent based on the notebook `rag.ipynb`.

Usage:
  python rag_agent.py [--rebuild-index] [--index-path INDEX_PATH] [--cards-path CARDS_GLOB]

The script will prompt for an OpenAI-compatible API key if not set in the environment.
It will load or build a FAISS index from `./data/rules` and optionally card JSONs from `./data/cards/en`.
"""

import argparse
import glob
import json
import os
import sys
import getpass
from difflib import SequenceMatcher
from typing import List, Tuple

try:
    # langchain packages used in the notebook
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PDFMinerLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.tools import tool
    from langchain.agents import create_agent
except Exception as e:
    print("Missing dependencies or failed import:", e)
    print("Make sure to run: pip install -r requirements.txt (or install langchain_openai, langchain-community, faiss-cpu, langchain-text-splitters)")
    raise


def ensure_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI-compatible API key: ")


def init_model():
    # notebook used an OpenRouter base; try to respect OPENAI_BASE_URL if set
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        model = ChatOpenAI(model="google/gemini-2.5-flash-lite", openai_api_base=base_url)
    else:
        model = ChatOpenAI()
    return model


def init_embeddings():
    return OpenAIEmbeddings(model="openai/text-embedding-3-small")


def build_or_load_faiss(index_path: str, embeddings, docs_paths: List[str], rebuild: bool = False):
    """Load existing index if present, otherwise build from documents.

    `docs_paths` is a list of file paths or globs to load (PDFs and/or text files).
    """
    if os.path.exists(index_path) and not rebuild:
        print(f"Loading vector store from {index_path}...")
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vs

    print("Building FAISS index from documents...")
    # create an empty index
    import faiss

    # minimal embedding to get dimension
    dummy = embeddings.embed_query("hello world")
    embedding_dim = len(dummy)
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    all_docs = []
    for p in docs_paths:
        # support globs
        for fp in glob.glob(p):
            try:
                if fp.lower().endswith(".pdf"):
                    loader = PDFMinerLoader(file_path=fp)
                    all_docs.extend(loader.load())
                else:
                    # plain text
                    with open(fp, "r", encoding="utf-8") as f:
                        text = f.read()
                    # create a simple document-like object expected by vectorstore
                    from types import SimpleNamespace

                    all_docs.append(SimpleNamespace(page_content=text, metadata={"source": fp}))
            except Exception as e:
                print(f"Warning: failed loading {fp}: {e}")

    if not all_docs:
        print("No documents found to index. You can still use card search.")
    else:
        # split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = splitter.split_documents(all_docs)
        print(f"Indexing {len(splits)} document chunks...")
        vector_store.add_documents(documents=splits)
        vector_store.save_local(index_path)
        print(f"Saved index to {index_path}")

    return vector_store


_cards_cache = None


def load_cards(path: str = "./data/cards/en/*.json") -> List[dict]:
    global _cards_cache
    if _cards_cache is None:
        _cards_cache = []
        for fp in glob.glob(path):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item["_source"] = fp
                            _cards_cache.append(item)
                elif isinstance(data, dict):
                    data["_source"] = fp
                    _cards_cache.append(data)
            except Exception:
                continue
    return _cards_cache


def search_rules(query: str, vector_store=None):
    """Retrieve top documents from the FAISS vector store."""
    if vector_store is None:
        return "No vector store available.", []
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join((f"Source: {getattr(doc, 'metadata', {})}\nContent: {getattr(doc, 'page_content', '')}") for doc in retrieved_docs)
    return serialized or "No matching rulebook passages found.", retrieved_docs


def search_cards(query: str, k: int = 5, cards_path: str = "./data/cards/en/*.json") -> Tuple[str, List[dict]]:
    cards = load_cards(cards_path)
    q = query.lower().strip()
    if not q:
        return "No query provided.", []

    def score_card(card: dict) -> int:
        text_fields = [
            card.get("name", ""),
            card.get("englishAttribute", ""),
            card.get("effectText", ""),
            json.dumps(card),
        ]
        combined = " ".join(str(t) for t in text_fields).lower()
        occurrence_score = combined.count(q) * 100
        ratios = [SequenceMatcher(None, q, str(t).lower()).ratio() for t in text_fields]
        fuzzy_score = int(max(ratios) * 100)
        return occurrence_score + fuzzy_score

    scored = [(score_card(c), c) for c in cards]
    scored = [s for s in scored if s[0] >= 30]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_cards = [c for _, c in scored[:k]]
    serialized = "\n\n".join(f"Source: {c.get('_source')}\nName: {c.get('name')}\nContent: {c}" for c in top_cards) or "No matching cards found."
    return serialized, top_cards


def create_agent_and_tools(model, vector_store, cards_glob):
    # create wrapper functions bound to our vector store and card path
    @tool("search_rules", description="Search rulebook passages (retrieval)", response_format="content_and_artifact")
    def bound_search_rules(query: str):
        """Search the rulebook vector store for relevant passages."""
        return search_rules(query, vector_store=vector_store)

    @tool("search_cards", description="Search card JSONs for matching cards", response_format="content_and_artifact")
    def bound_search_cards(query: str):
        """Search the card JSON dataset for matching cards."""
        return search_cards(query, cards_path=cards_glob)

    tools = [bound_search_rules, bound_search_cards]

    prompt = (
        "You have access to tools that retrieve card and rule information from the Yu-Gi-Oh! database. "
        "Use the tools to help answer user queries. If you can answer directly, do so; otherwise use the tools to fetch facts."
    )
    agent = create_agent(model, tools, system_prompt=prompt)
    return agent


def interactive_loop(agent, vector_store, cards_glob):
    print("\nInteractive RAG Agent — type a question and press Enter. Type 'exit' or Ctrl+C to quit.")
    try:
        while True:
            query = input("\n> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye")
                break

            # try to stream if the agent supports it
            try:
                if hasattr(agent, "stream"):
                    for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
                        # event may contain 'messages' with objects that have pretty_print or .content
                        try:
                            msg = event.get("messages", [])[-1]
                            if hasattr(msg, "pretty_print"):
                                msg.pretty_print()
                            else:
                                # try to print content attribute or dict
                                print(getattr(msg, "content", msg))
                        except Exception:
                            print(event)
                else:
                    resp = agent.run({"messages": [{"role": "user", "content": query}]})
                    print(resp)
            except Exception:
                # fallback: perform retrievals and show them
                print("Agent failed, performing basic retrievals as fallback:")
                sr, rd = search_rules(query, vector_store=vector_store)
                print("\n-- Rulebook retrievals:\n", sr)
                sc, cards = search_cards(query, cards_path=cards_glob)
                print("\n-- Card search results:\n", sc)

    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="index.faiss", help="FAISS index folder path")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the FAISS index from source docs")
    parser.add_argument("--rules-glob", default="./data/rules/*.pdf", help="Glob for rulebook files")
    parser.add_argument("--cards-glob", default="./data/cards/en/*.json", help="Glob for card JSON files")
    args = parser.parse_args()

    ensure_api_key()
    model = init_model()
    embeddings = init_embeddings()

    vector_store = None
    try:
        vector_store = build_or_load_faiss(args.index_path, embeddings, [args.rules_glob], rebuild=args.rebuild_index)
    except Exception as e:
        print(f"Failed to initialize vector store: {e}")

    agent = create_agent_and_tools(model, vector_store, cards_glob=args.cards_glob)

    interactive_loop(agent, vector_store, args.cards_glob)


if __name__ == "__main__":
    main()
