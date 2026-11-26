import torch.nn as nn
from torchvision import models

def build_model(n, arch="resnet50", pretrained=True, dropout=.25):
    if arch=="resnet50":
        m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(m.fc.in_features,n))
    elif arch=="efficientnet_b0":
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1]=nn.Sequential(nn.Dropout(dropout),nn.Linear(m.classifier[1].in_features,n))
    else: raise ValueError("Unknown architecture")
    return m

def freeze_backbone(m, arch):
    for p in m.parameters(): p.requires_grad=False
    head=m.fc if arch=="resnet50" else m.classifier
    for p in head.parameters(): p.requires_grad=True

def unfreeze_all(m):
    for p in m.parameters(): p.requires_grad=True
