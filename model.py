import torch
import torch.nn as nn
from torchvision import models
import config

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100
}

def get_model(model_name,dataset_name,freeze_fraction=0.0):
    
    num_classes= NUM_CLASSES[dataset_name]
    
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        backbone=[name for name,_ in model.named_parameters() if not name.startswith("fc")]
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        backbone=[name for name,_ in model.named_parameters() if not name.startswith("classifier")]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    _apply_freeze(model,backbone,freeze_fraction)
    return model.to(config.DEVICE)

def _apply_freeze(model,backbone,freeze_fraction):
    if freeze_fraction == 0.0:
        return
  
    num_to_freeze = int(len(backbone) * freeze_fraction)
    frozen_names=set(backbone[:num_to_freeze])
    for name, param in model.named_parameters():
        if name in frozen_names:
            param.requires_grad = False
            
def count_params(model):
    total=sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total,trainable
