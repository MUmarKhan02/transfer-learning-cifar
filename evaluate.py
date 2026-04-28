import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config

#CIFAR-100 superclass index for each fine class (official mapping)

COARSE_LABELS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

VEHICLE_SUPERCLASSES={8,14,9}
ANIMAL_SUPERCLASSES={0,1,2,5,7,15,19}

def get_cifar100_domain_shift_loader():
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    keep_superclasses = VEHICLE_SUPERCLASSES | ANIMAL_SUPERCLASSES
    indices = [i for i, label in enumerate(dataset.targets) if COARSE_LABELS[label] in keep_superclasses]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

@torch.no_grad()
def evaluate_domain_shift(model):
    
    C10_VEHICLE={0,1,8,9}
    C10_ANIMAL={2,3,4,5,6,7}
    loader = get_cifar100_domain_shift_loader()
    model.eval()
    correct,total=0,0
    
    for images, labels in loader:
        images = images.to(config.DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        
        for pred, label in zip(preds, labels):
            superclass = COARSE_LABELS[label]
            pred_is_vehicle = pred in C10_VEHICLE
            pred_is_animal = pred in C10_ANIMAL
            label_is_vehicle = superclass in VEHICLE_SUPERCLASSES
            label_is_animal = superclass in ANIMAL_SUPERCLASSES
            if (pred_is_vehicle and label_is_vehicle) or (pred_is_animal and label_is_animal):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0
