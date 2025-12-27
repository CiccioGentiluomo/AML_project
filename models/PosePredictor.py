import torch
import torch.nn as nn
from torchvision import models

class PosePredictor(nn.Module):
    def __init__(self):
        super(PosePredictor, self).__init__()
        # Caricamento Backbone ResNet-50 con pesi ImageNet 
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Estrazione del numero di feature in ingresso all'ultimo strato
        num_features = self.backbone.fc.in_features
        
        # Rimozione del layer di classificazione originale [cite: 175]
        self.backbone.fc = nn.Identity()
        
        # Definizione della Regression Head per i quaternioni (4D) 
        self.regression_head = nn.Linear(num_features, 4)

    def forward(self, x):
        # x: Immagine RGB croppata e ridimensionata [cite: 173]
        features = self.backbone(x)
        quaternions = self.regression_head(features)
        return quaternions