import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(path)

datasets = df["dataset"].unique()
methods = ["zero-shot", "base", "lora"]
colors = {"zero-shot": "#d9534f", "base": "#5bc0de", "lora": "#5cb85c"}

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

for i, method in enumerate(methods):
    values = [
        df[(df["dataset"] == d) & (df["peft_method"] == method)]["exact_match"].values[
            0
        ]
        for d in datasets
    ]
    bars = ax.bar(
        x + i * width, values, width, label=method, color=colors[method], alpha=0.85
    )

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Accuracy by Dataset and Method (llama-3.2-1b-instruct)", fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels(datasets, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.legend(title="Method")
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
print("Saved to results.png")
