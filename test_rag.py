#!/usr/bin/env python3
"""Run the Yu-Gi-Oh! RAG system against the multiple-choice QA set.

This script loads `data/questions/qa.json`, performs retrievals (rulebook FAISS + card JSONs),
then asks the chat model to answer each question using the retrieved context. It reports per-question
outputs and a simple accuracy metric (detecting letter A/B/C/D in model output).

Usage:
  python test_rag.py [--chat-model MODEL] [--embed-model MODEL] [--index-path PATH] [--rebuild-index]

If you only want to test retrievals without calling the model (offline), use `--no-chat`.
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path


def ensure_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        import getpass

        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI-compatible API key: ")


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def extract_choice_text(choice_map, letter):
    return choice_map.get(letter.upper(), "") if isinstance(choice_map, dict) else ""


def detect_answer_in_text(text: str, choices: dict):
    """Try to detect which choice (A/B/C/D) the model's text corresponds to.

    Returns detected_letter or None.
    """
    if not text:
        return None

    up = text.upper()

    # 1) Explicit single-letter answers (covers 'D.', 'D)', 'D:', ' D ', 'Answer: D', etc.)
    m = re.search(r"\b([A-D])\b", up)
    if m:
        return m.group(1)

    # 1b) Common phrasings: 'ANSWER: D', 'THE ANSWER IS D', 'CHOICE D'
    m = re.search(r"(?:ANSWER|CHOICE|OPTION|THE ANSWER IS)[:\s-]*([A-D])\b", up)
    if m:
        return m.group(1)

    # 2) Fallback: exact (normalized) substring of the choice text in model output
    def _normalize(s: str) -> str:
        return re.sub(r"[^A-Z0-9\s]", "", (s or "").upper())

    norm_up = _normalize(up)
    for letter, txt in (choices or {}).items():
        if not isinstance(letter, str):
            continue
        if not txt:
            continue
        if _normalize(txt) and _normalize(txt) in norm_up:
            return letter

    # 3) Last resort: token-overlap / fuzzy-ish match between model output and each choice
    model_tokens = set(tok for tok in norm_up.split() if tok)
    best = None
    best_count = 0
    for letter, txt in (choices or {}).items():
        txt_norm = _normalize(txt)
        tokens = set(tok for tok in txt_norm.split() if tok)
        if not tokens:
            continue
        common = model_tokens & tokens
        count = len(common)
        if count > best_count:
            best_count = count
            best = letter

    if best:
        # require a reasonable overlap: either >=2 shared tokens or >=50% of choice tokens
        choice_tokens = set(tok for tok in _normalize(choices.get(best, "")).split() if tok)
        if best_count >= 2 or (choice_tokens and (best_count / max(1, len(choice_tokens))) >= 0.5):
            return best

    return None


def call_chat_model(model, user_prompt: str, system_prompt: str | None = None, debug: bool = False) -> str:
    """Call a chat model instance with flexible API handling and return text.

    Tries several common LangChain/OpenAI client call patterns:
    - model.invoke([SystemMessage,...,HumanMessage,...])
    - model.generate([[SystemMessage,...,HumanMessage,...]])
    - model.predict_messages or model.predict
    Returns the generated text or empty string on failure.
    """
    text = ""
    last_error = None

    # Try to import message classes dynamically
    try:
        import importlib
        schema_mod = importlib.import_module("langchain_core.messages")
        SystemMessage = getattr(schema_mod, "SystemMessage", None)
        HumanMessage = getattr(schema_mod, "HumanMessage", None)
    except Exception:
        try:
            import importlib
            schema_mod = importlib.import_module("langchain.schema")
            SystemMessage = getattr(schema_mod, "SystemMessage", None)
            HumanMessage = getattr(schema_mod, "HumanMessage", None)
        except Exception:
            SystemMessage = None
            HumanMessage = None

    # Build messages if classes are available
    messages = None
    if SystemMessage and HumanMessage:
        sys_msg = SystemMessage(content=system_prompt or "You are a helpful assistant.")
        human_msg = HumanMessage(content=user_prompt)
        messages = [sys_msg, human_msg]

    # Try 1: model.invoke (modern LangChain)
    if messages and hasattr(model, "invoke"):
        try:
            if debug:
                print("[DEBUG] Trying model.invoke(messages)...")
            response = model.invoke(messages)
            # Extract content from AIMessage or similar
            text = getattr(response, "content", str(response))
            if text:
                return text
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model.invoke failed: {e}")

    # Try 2: model.generate (older LangChain ChatModel API)
    if messages and hasattr(model, "generate"):
        try:
            if debug:
                print("[DEBUG] Trying model.generate([[messages]])...")
            res = model.generate([messages])
            # Extract from LLMResult.generations
            try:
                text = res.generations[0][0].text
            except Exception:
                try:
                    text = res.generations[0][0].message.content
                except Exception:
                    text = str(res)
            if text:
                return text
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model.generate failed: {e}")

    # Try 3: model.predict_messages
    if messages and hasattr(model, "predict_messages"):
        try:
            if debug:
                print("[DEBUG] Trying model.predict_messages(messages)...")
            msg = model.predict_messages(messages)
            text = getattr(msg, "content", str(msg))
            if text:
                return text
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model.predict_messages failed: {e}")

    # Try 4: model.predict (simple string input)
    if hasattr(model, "predict"):
        try:
            if debug:
                print("[DEBUG] Trying model.predict(user_prompt)...")
            text = model.predict(user_prompt)
            if text:
                return str(text)
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model.predict failed: {e}")

    # Try 5: model.call_as_llm
    if hasattr(model, "call_as_llm"):
        try:
            if debug:
                print("[DEBUG] Trying model.call_as_llm(user_prompt)...")
            text = model.call_as_llm(user_prompt)
            if text:
                return str(text)
        except Exception as e:
            last_error = e
            if debug:
                print(f"[DEBUG] model.call_as_llm failed: {e}")

    if debug and last_error:
        print(f"[DEBUG] All methods failed. Last error: {last_error}")

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-model", default=None, help="Chat model name to use")
    parser.add_argument("--embed-model", default=None, help="Embedding model name to use")
    parser.add_argument("--index-path", default="index.faiss", help="FAISS index path")
    parser.add_argument("--rules-glob", default="./data/rules/*.pdf", help="Glob for rulebook files")
    parser.add_argument("--cards-glob", default="./data/cards/en/*.json", help="Glob for card JSON files")
    parser.add_argument("--qa-path", default="./data/questions/qa.json", help="Path to QA JSON file")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--no-chat", action="store_true", help="Only run retrievals, do not call chat model")
    args = parser.parse_args()

    # import rag components (assumes this script lives in same folder as rag_agent)
    repo_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_dir))

    try:
        import rag_agent as ra
    except Exception as e:
        print("Failed to import rag_agent module:", e)
        raise

    ensure_api_key()

    # initialize model and embeddings
    model = None
    if not args.no_chat:
        model = ra.init_model(chat_model=args.chat_model)

    embeddings = ra.init_embeddings(embed_model=args.embed_model)

    # build or load vector store
    vector_store = ra.build_or_load_faiss(args.index_path, embeddings, [args.rules_glob], rebuild=args.rebuild_index)

    questions = load_questions(args.qa_path)
    if not questions:
        print("No questions found in", args.qa_path)
        return

    correct = 0
    results = []

    for q in questions:
        qid = q.get("id")
        prompt_q = q.get("question")
        choices = q.get("choices", {})
        answer_letter = q.get("answer")

        # retrievals
        rules_text, rules_docs = ra.search_rules(prompt_q, vector_store=vector_store)
        cards_text, cards_docs = ra.search_cards(prompt_q, cards_path=args.cards_glob)

        context = "".join(["Rulebook:\n", rules_text, "\n\nCards:\n", cards_text])

        model_output = None
        detected = None

        if args.no_chat:
            print(f"Q {qid}: {prompt_q}")
            print("-- Retrieved Context --\n", context)
            detected = None
        else:
            # craft a prompt for the chat model
            user_prompt = (
                "You are an expert Yu-Gi-Oh! rules assistant. Use the context below to answer the multiple-choice question. "
                "If the answer corresponds to one of the choices, reply with the letter (A, B, C, or D) and a short justification.\n\n"
                f"Context:\n{context}\n\nQuestion: {prompt_q}\nChoices:\n"
            )
            for let, txt in choices.items():
                user_prompt += f"{let}: {txt}\n"

            # call the chat model using a robust helper that supports multiple client APIs
            text = call_chat_model(model, user_prompt, system_prompt="You are a helpful Yu-Gi-Oh rules assistant.", debug=True)
            if not text:
                print("Failed to call chat model or model returned empty response.")

            model_output = text
            detected = detect_answer_in_text(text, choices)

            print(f"Q {qid}: {prompt_q}")
            print("Model output:\n", model_output)
            print("Detected choice:", detected, "Expected:", answer_letter)

        is_correct = detected == (answer_letter or None)
        if is_correct:
            correct += 1

        results.append({"id": qid, "detected": detected, "expected": answer_letter, "correct": is_correct})

    total = len(questions)
    print("\nSummary:")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")


if __name__ == "__main__":
    main()
