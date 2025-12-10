# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision import models


class CNN_ResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN_ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
