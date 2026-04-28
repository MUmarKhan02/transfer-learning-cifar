import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/all_results.csv")
os.makedirs("plots", exist_ok=True)

fig,axes=plt.subplots(2,2,figsize=(12,10))
for i,model in enumerate(["resnet50","efficientnet_b3"]):
    for j,dataset in enumerate(["cifar10","cifar100"]):
        ax = axes[i,j]
        subset = df[(df.model==model)&(df.dataset==dataset) & (df.data_split==1.0)]
        ax.plot(subset.freeze_fraction * 100, subset.best_val_acc, marker="o")
        ax.set_title(f"{model} on {dataset}")
        ax.set_xlabel("Freeze Fraction (%)")
        ax.set_ylabel("Best Val Accuracy")
        ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/freeze_vs_acc.png",dpi=150)
plt.close()

domain_df = df[(df.dataset=="cifar10") & (df.domain_shift_acc.notna())]
fig,ax=plt.subplots(figsize=(8,5))
for model in ["resnet50","efficientnet_b3"]:
    subset = domain_df[(domain_df.model==model) & (domain_df.data_split==1.0)]
    ax.plot(subset.freeze_fraction * 100, subset.domain_shift_acc, marker="s", label=model)

ax.set_title("Domain Shift Accuracy (C10 -> C100)")
ax.set_xlabel("Freeze Fraction (%)")
ax.set_ylabel("Superclass-level Accuracy")
ax.legend()
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/domain_shift_acc.png",dpi=150)
plt.close()

print("Plots saved in results/plots/")