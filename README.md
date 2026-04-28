# Transfer Learning for Image Classification (CIFAR)

A systematic investigation of transfer learning strategies for image classification using pretrained convolutional neural networks on CIFAR-10 and CIFAR-100, including a cross-dataset domain shift evaluation.

## Overview

This project evaluates the effectiveness of different layer freezing strategies, training data sizes, and cross-dataset generalization for image classification tasks.

Two pretrained architectures are compared:
- **ResNet-50** (~25M parameters)
- **EfficientNet-B3** (~12M parameters)

## Experiments

- 2 models × 2 datasets × 3 data splits × 5 freeze levels = **60 total runs**
- Freeze levels: 0%, 25%, 50%, 75%, 100%
- Data splits: 25%, 50%, 100% of training data
- Domain shift evaluation: trained on CIFAR-10, tested on CIFAR-100
- All models pretrained on ImageNet

## Results Summary

| Model | Dataset | 0% Frozen | 25% Frozen | 50% Frozen | 75% Frozen | 100% Frozen |
|---|---|---|---|---|---|---|
| ResNet-50 | CIFAR-10 | 96.68% | 96.61% | 95.98% | 94.50% | 73.79% |
| ResNet-50 | CIFAR-100 | 84.15% | 83.65% | 82.22% | 79.34% | 54.98% |
| EfficientNet-B3 | CIFAR-10 | 96.16% | 95.81% | 95.01% | 89.36% | 62.01% |
| EfficientNet-B3 | CIFAR-100 | 82.86% | 82.13% | 80.62% | 70.21% | 40.37% |

## Domain Shift Results

Models trained on CIFAR-10 evaluated on CIFAR-100 at superclass level:

| Model | 0% Frozen | 25% Frozen | 50% Frozen | 75% Frozen | 100% Frozen |
|---|---|---|---|---|---|
| ResNet-50 | 56.60% | 56.60% | 55.02% | 57.22% | 53.02% |
| EfficientNet-B3 | 56.60% | 59.37% | 57.37% | 57.00% | 56.90% |

## Dataset

CIFAR-10 and CIFAR-100 are automatically downloaded via torchvision when running the code — no manual setup required.

## Installation

**Install dependencies:**
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Usage

**Run all experiments:**
```bash
python experiment.py
```

The experiment runner includes resume/skip logic — if interrupted, it will automatically skip completed runs and continue from where it left off.

**Generate plots:**
```bash
python plot_results.py
python plot_remaining.py
```

Results are saved to `results/all_results.csv` and plots to `results/plots/`.

## Project Structure
```
├── config.py              # Hyperparameters and experiment settings
├── data.py                # Dataset loading and preprocessing
├── model.py               # Model loading and layer freezing logic
├── train.py               # Training and validation loops
├── evaluate.py            # Domain shift evaluation
├── experiment.py          # Main experiment runner
├── plot_results.py        # Result visualization
├── plot_remaining.py      # Additional plots
└── results/
  ├── all_results.csv    # All experiment results
  └── plots/             # Generated figures
```
## Key Findings

- Full fine-tuning (0% frozen) consistently achieves highest accuracy across all conditions
- Freezing up to 25% of backbone layers offers free efficiency gains with negligible accuracy cost
- Performance drops sharply beyond 75% freezing — feature extraction alone is insufficient for high class granularity tasks
- ResNet-50 is more robust to aggressive freezing, outperforming EfficientNet-B3 by up to 14.6 points at 100% freeze on CIFAR-100
- Both models maintain ~56-57% superclass accuracy under domain shift, with EfficientNet-B3 showing more stable cross-dataset generalization
- Transfer learning achieves near-full performance with as little as 25% of training data

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU recommended

## Course

CSC 7760 - Deep Learning | Wayne State University | April 2026
