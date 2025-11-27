#!/usr/bin/env python3
"""Plot Cartesian results from `results/cartesian_results.json`.

Creates `results/cartesian_plot.png` (grouped bar chart: chat models on x, bars per embed model).
"""
import json
from pathlib import Path
import re
import math

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required to run this script. Install with: pip install matplotlib")
    raise

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "cartesian_results.json"
OUT = HERE / "results" / "cartesian_plot.png"

if not RESULTS.exists():
    raise SystemExit(f"Results file not found: {RESULTS}")

with open(RESULTS, "r", encoding="utf-8") as f:
    data = json.load(f)

# parse accuracy from summary_line like 'Total: 20, Correct: 19, Accuracy: 95.00%'
def parse_accuracy(entry):
    s = entry.get("summary_line")
    if not s:
        # try to parse logfile for an Accuracy line
        logfile = entry.get("logfile")
        if logfile:
            try:
                txt = Path(logfile).read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"Accuracy:\s*([0-9.]+)%", txt)
                if m:
                    return float(m.group(1))
            except Exception:
                pass
        return math.nan
    m = re.search(r"Accuracy:\s*([0-9.]+)%", s)
    if m:
        return float(m.group(1))
    return math.nan

# order unique chat and embed models
chat_models = []
embed_models = []
for e in data:
    c = e.get("chat_model")
    em = e.get("embed_model")
    if c not in chat_models:
        chat_models.append(c)
    if em not in embed_models:
        embed_models.append(em)

# build matrix of accuracies
acc_matrix = {em: [] for em in embed_models}
for c in chat_models:
    for em in embed_models:
        # find corresponding entry
        found = None
        for e in data:
            if e.get("chat_model") == c and e.get("embed_model") == em:
                found = e
                break
        acc = parse_accuracy(found) if found else math.nan
        acc_matrix[em].append(acc)

# plotting
x = list(range(len(chat_models)))
width = 0.8 / max(1, len(embed_models))
plt.figure(figsize=(max(8, len(chat_models)*1.5), 6))
for i, em in enumerate(embed_models):
    offsets = [xi + (i - (len(embed_models)-1)/2.0)*width for xi in x]
    plt.bar(offsets, acc_matrix[em], width=width, label=em)

plt.xticks(x, [c.split('/')[-1] for c in chat_models], rotation=30, ha='right')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Cartesian Results: Accuracy by Chat Model and Embed Model')
plt.legend(title='Embed Model')
plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200)
print(f"Saved plot to {OUT}")
