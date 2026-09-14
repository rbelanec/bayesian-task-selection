from pandas import read_csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

df=read_csv('figures/merged/merge_summary.csv')
base=df[df.method=="base"].groupby("combo")["mean_acc"].mean().sort_index()
lora=df[df.method=="lora"].groupby("combo")["mean_acc"].mean().sort_index()
combos=sorted(set(base.index)|set(lora.index))
base=base.reindex(combos)
lora=lora.reindex(combos)

base_avg=base.mean()
lora_avg=lora.mean()

xb=np.arange(len(combos)+1)
xl=np.arange(len(combos)+1)+len(combos)+3

fig,ax=plt.subplots(figsize=(13,4))
bars_base = ax.bar(xb[:-1],base.values)
bars_base_avg = ax.bar(xb[-1],base_avg)
bars_lora = ax.bar(xl[:-1],lora.values)
bars_lora_avg = ax.bar(xl[-1],lora_avg)

# Add labels
ax.bar_label(bars_base_avg, fmt="%.3f", padding=2, fontsize=7, label_type="edge", rotation=90)
ax.bar_label(bars_lora_avg, fmt="%.3f", padding=2, fontsize=7, label_type="edge", rotation=90)

labels=combos+["Average"]+combos+["Average"]
ax.set_xticks(np.concatenate([xb,xl]))
ax.set_xticklabels(labels,rotation=90,fontsize=7)
ax.set_ylabel("Mean accuracy")
# ax.set_title("Mean accuracy per task combination with overall average")
ax.text(xb.mean(),1.02,"Base",transform=ax.get_xaxis_transform(),ha="center")
ax.text(xl.mean(),1.02,"LoRA",transform=ax.get_xaxis_transform(),ha="center")
plt.savefig("figures/merged/base_lora_two_groups_with_average.png",dpi=150,bbox_inches="tight")
plt.close()
