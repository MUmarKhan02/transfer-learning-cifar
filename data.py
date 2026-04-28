import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import config

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config.IMAGE_SIZE, padding=16),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_dataset(name, train=True):
    transforms = get_transforms(train)
    root = "./data"
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transforms)
    elif name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transforms)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
def get_subset(dataset,fraction):
    if fraction >= 1.0:
        return dataset
    n= int(len(dataset)*fraction)
    targets = np.array(dataset.targets)
    indices=[]
    classes=np.unique(targets)
    per_class=max(1,n//len(classes))
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        selected_indices = np.random.choice(cls_indices, min(per_class, len(cls_indices)), replace=False)
        indices.extend(selected_indices.tolist())
    return Subset(dataset, indices)

def get_loaders(dataset_name,split_fraction=1.0):
    train_dataset = get_dataset(dataset_name, train=True)
    test_dataset = get_dataset(dataset_name, train=False)
    
    train_subset = get_subset(train_dataset, split_fraction)
    
    return(
        DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,pin_memory=True),
        DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,pin_memory=True)
    )