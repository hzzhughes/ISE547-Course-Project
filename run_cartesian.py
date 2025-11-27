#!/usr/bin/env python3
"""Run `test_rag.py` across a Cartesian product of chat and embed models.

Saves per-run logs under `results/` and aggregates a JSON summary at
`results/cartesian_results.json`.

Usage:
  python run_cartesian.py

Edit the `chat_models` and `embed_models` lists below if you want different sets.
"""

import json
import os
import subprocess
from pathlib import Path

# --- User-provided model lists (from user's attachment) ---
chat_models = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-lite-001",
    "qwen/qwen-2.5-72b-instruct",
]
embed_models = [
    "openai/text-embedding-3-small",
    "qwen/qwen3-embedding-8b",
]

HERE = Path(__file__).resolve().parent
TEST_SCRIPT = HERE / "test_rag.py"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# quick dependency check helper
def check_imports():
    mods = [
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "faiss",
        "openai",
        "pdfminer",
    ]
    missing = []
    for m in mods:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    return missing


def run_pair(chat_model, embed_model, extra_args=None):
    env = os.environ.copy()
    # Always rebuild index so embedding model dimension mismatches are avoided
    cmd = ["python", str(TEST_SCRIPT), "--chat-model", chat_model, "--embed-model", embed_model, "--rebuild-index"]
    if extra_args:
        cmd.extend(extra_args)
    logfile = RESULTS_DIR / f"log_chat={chat_model.replace('/',':')}_embed={embed_model.replace('/',':')}.txt"
    print("Running:", " ".join(cmd))
    with open(logfile, "wb") as out:
        try:
            res = subprocess.run(cmd, env=env, cwd=str(HERE), stdout=out, stderr=subprocess.STDOUT, timeout=600)
            return res.returncode, str(logfile)
        except subprocess.TimeoutExpired:
            out.write(b"TIMEOUT\n")
            return -1, str(logfile)
        except Exception as e:
            out.write(str(e).encode() + b"\n")
            return -2, str(logfile)


def main():
    print("Checking imports before running tests...")
    missing = check_imports()
    if missing:
        print("Missing packages detected:", missing)
        print("You should install them before running these tests. Example:")
        print("  pip install langchain-openai langchain-community faiss-cpu openai pdfminer.six")
        print("Proceeding anyway; runs may fail and logs will capture errors.")

    summary = []
    total = len(chat_models) * len(embed_models)
    i = 0
    for c in chat_models:
        for e in embed_models:
            i += 1
            print(f"({i}/{total}) Chat={c} Embed={e}")
            code, logfile = run_pair(c, e)
            # try to extract a simple accuracy line from the log
            acc = None
            try:
                txt = Path(logfile).read_text(encoding="utf-8", errors="ignore")
                for line in txt.splitlines()[::-1]:
                    if line.strip().startswith("Total:") or line.strip().startswith("Summary:"):
                        acc = line.strip()
                        break
            except Exception:
                pass
            summary.append({
                "chat_model": c,
                "embed_model": e,
                "returncode": code,
                "logfile": logfile,
                "summary_line": acc,
            })

    outp = RESULTS_DIR / "cartesian_results.json"
    outp.write_text(json.dumps(summary, indent=2))
    print(f"Done. Results written to {outp}")


if __name__ == '__main__':
    main()
