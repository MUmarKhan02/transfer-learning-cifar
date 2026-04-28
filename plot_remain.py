import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/all_results.csv")
os.makedirs("results/plots", exist_ok=True)

# 1. Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, dataset in enumerate(["cifar10", "cifar100"]):
    ax = axes[i]
    for model in ["resnet50", "efficientnet_b3"]:
        subset = df[(df.model == model) & (df.dataset == dataset) & (df.data_split == 1.0)].sort_values("freeze_fraction")
        ax.plot(subset.freeze_fraction * 100, subset.best_val_acc, marker="o", label=model)
    ax.set_title(f"Model comparison — {dataset}")
    ax.set_xlabel("Layers frozen (%)")
    ax.set_ylabel("Best val accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/model_comparison.png", dpi=150)
plt.close()
print("Saved model_comparison.png")

# 2. Data Split Effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, dataset in enumerate(["cifar10", "cifar100"]):
    ax = axes[i]
    for model in ["resnet50", "efficientnet_b3"]:
        subset = df[(df.model == model) & (df.dataset == dataset) & (df.freeze_fraction == 0.0)].sort_values("data_split")
        ax.plot(subset.data_split * 100, subset.best_val_acc, marker="o", label=model)
    ax.set_title(f"Data efficiency — {dataset}")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Best val accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/data_split.png", dpi=150)
plt.close()
print("Saved data_split.png")

# 3. Training Time
fig, ax = plt.subplots(figsize=(8, 5))
for model in ["resnet50", "efficientnet_b3"]:
    subset = df[(df.model == model) & (df.dataset == "cifar10") & (df.data_split == 1.0)].sort_values("freeze_fraction")
    ax.plot(subset.freeze_fraction * 100, subset.wall_time_seconds / 60, marker="o", label=model)
ax.set_title("Training time vs freeze level (CIFAR-10, 100% data)")
ax.set_xlabel("Layers frozen (%)")
ax.set_ylabel("Training time (minutes)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/training_time.png", dpi=150)
plt.close()
print("Saved training_time.png")

print("All done!")