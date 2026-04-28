import os
import csv
import copy
import torch
import config
from data import get_loaders
from model import get_model, count_params
from train import run_training
from evaluate import evaluate_domain_shift


os.makedirs(config.RESULTS_DIR, exist_ok=True)  # ADD THIS LINE


RESULTS_FILE=os.path.join(config.RESULTS_DIR, "all_results.csv")

FIELDNAMES=["model","dataset","freeze_fraction","data_split","best_val_acc","final_train_acc","final_val_acc",
            "wall_time_seconds","trainable_params","total_params","domain_shift_acc"]

def run_all():
    # Load already completed runs to skip them
    completed = set()
    if os.path.exists(RESULTS_FILE):
        import pandas as pd
        existing = pd.read_csv(RESULTS_FILE)
        for _, row in existing.iterrows():
            completed.add((row['model'], row['dataset'], 
                          round(row['freeze_fraction'], 2), 
                          round(row['data_split'], 2)))
        print(f"Found {len(completed)} completed runs, skipping them...")
    else:
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    for model_name in config.MODELS:
        for dataset_name in config.DATASETS:
            for split in config.DATA_SPLITS:
                for freeze in config.FREEZE_LEVELS:
                    # Skip if already done
                    if (model_name, dataset_name, round(freeze, 2), round(split, 2)) in completed:
                        print(f"Skipping {model_name}/{dataset_name}/split{int(split*100)}/freeze{int(freeze*100)} (already done)")
                        continue

                    label = f"{model_name}/{dataset_name}/split{int(split*100)}/freeze{int(freeze*100)}"
                    print(f"\n{'='*60}")
                    print(f"Running experiment: {label}")
                    print(f"{'='*60}\n")

                    model = get_model(model_name, dataset_name, freeze_fraction=freeze)
                    total_p, trainable_p = count_params(model)
                    train_loader, val_loader = get_loaders(dataset_name, split_fraction=split)
                    history = run_training(model, train_loader, val_loader, run_label=label)

                    history_file = os.path.join(config.RESULTS_DIR, f"{model_name}_{dataset_name}_split{int(split*100)}_freeze{int(freeze*100)}_history.csv")
                    with open(history_file, "w", newline="") as f:
                        hist_writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","train_acc","val_acc"])
                        hist_writer.writeheader()
                        for i in range(len(history["train_loss"])):
                            hist_writer.writerow({
                                "epoch": i+1,
                                "train_loss": history["train_loss"][i],
                                "val_loss": history["val_loss"][i],
                                "train_acc": history["train_acc"][i],
                                "val_acc": history["val_acc"][i],
                            })

                    domain_acc = None
                    if dataset_name == "cifar10":
                        domain_acc = evaluate_domain_shift(model)
                        print(f"Domain Shift Accuracy (C10 -> C100): {domain_acc:.4f}")

                    row = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "freeze_fraction": freeze,
                        "data_split": split,
                        "best_val_acc": history["best_val_acc"],
                        "final_train_acc": history["train_acc"][-1],
                        "final_val_acc": history["val_acc"][-1],
                        "wall_time_seconds": history["wall_time_seconds"],
                        "trainable_params": trainable_p,
                        "total_params": total_p,
                        "domain_shift_acc": domain_acc
                    }
                    with open(RESULTS_FILE, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                        writer.writerow(row)

                    print(f"Best val acc: {history['best_val_acc']:.4f}")
                    del model
                    torch.cuda.empty_cache()

    print(f"\nAll experiments completed. Results saved to {RESULTS_FILE}")



# def run_all():
#     with open(RESULTS_FILE,"w",newline="") as f:
#         writer=csv.DictWriter(f,fieldnames=FIELDNAMES)
#         writer.writeheader()
    
#     for model_name in config.MODELS:
#         for dataset_name in config.DATASETS:
#             for split in config.DATA_SPLITS:
#                 for freeze in config.FREEZE_LEVELS:
#                     label = f"{model_name}/{dataset_name}/split{int(split*100)}/freeze{int(freeze*100)}"
#                     print(f"\n{'='*60}")
#                     print(f"Running experiment: {label}")
#                     print(f"{'='*60}\n")
                    
#                     model = get_model(model_name, dataset_name, freeze_fraction=freeze)
                    
#                     total_p, trainable_p = count_params(model)
                    
#                     train_loader,val_loader = get_loaders(dataset_name, split_fraction=split)
#                     history = run_training(model, train_loader, val_loader, run_label=label)
                    
#                     history_file = os.path.join(config.RESULTS_DIR, f"{model_name}_{dataset_name}_split{int(split*100)}_freeze{int(freeze*100)}_history.csv")
#                     with open(history_file, "w", newline="") as f:
#                         hist_writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","train_acc","val_acc"])
#                         hist_writer.writeheader()
#                         for i in range(len(history["train_loss"])):
#                             hist_writer.writerow({
#                                 "epoch": i+1,
#                                 "train_loss": history["train_loss"][i],
#                                 "val_loss": history["val_loss"][i],
#                                 "train_acc": history["train_acc"][i],
#                                 "val_acc": history["val_acc"][i],
#                             })
                    
#                     domain_acc = None
#                     if dataset_name == "cifar10":
#                         domain_acc = evaluate_domain_shift(model)
#                         print(f"Domain Shift Accuracy (C10 -> C100): {domain_acc:.4f}")

#                     row = {
#                         "model": model_name,
#                         "dataset": dataset_name,
#                         "freeze_fraction": freeze,
#                         "data_split": split,
#                         "best_val_acc": history["best_val_acc"],
#                         "final_train_acc": history["train_acc"][-1],
#                         "final_val_acc": history["val_acc"][-1],
#                         "wall_time_seconds": history["wall_time_seconds"],
#                         "trainable_params": trainable_p,
#                         "total_params": total_p,
#                         "domain_shift_acc": domain_acc
#                     }
#                     with open(RESULTS_FILE,"a",newline="") as f:
#                         writer=csv.DictWriter(f,fieldnames=FIELDNAMES)
#                         writer.writerow(row)
                    
#                     print(f"Best val acc: {history['best_val_acc']:.4f}")
                    
#                     del model
#                     torch.cuda.empty_cache()
#     print(f"\nAll experiments completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_all()
    
                