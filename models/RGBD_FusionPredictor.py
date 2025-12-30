import torch
import torch.nn as nn
from torchvision import models

class RGBD_FusionPredictor(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor, self).__init__()
        
        # 1. Ramo RGB: ResNet-50 (Backbone della Fase 3)
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features_rgb = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity() # Rimuove il classificatore
        
        # 2. Ramo DEPTH: Nuova CNN (usiamo ResNet-18 per leggere la geometria)
        # Nota: La profondità è spesso a un canale, ma torchvision si aspetta 3 canali.
        # Possiamo duplicare il canale Depth o modificare il primo layer.
        self.depth_backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_features_depth = self.depth_backbone.fc.in_features
        self.depth_backbone.fc = nn.Identity()
        
        # Dimensione totale dopo la concatenazione (2048 + 512 = 2560)
        combined_features = num_features_rgb + num_features_depth
        
        # 3. Pose Estimator (MLP): Predictor di Traslazione e Rotazione
        # Usiamo strati intermedi per elaborare la fusione delle feature
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output Traslazione: 3 coordinate (X, Y, Z)
        self.translation_head = nn.Linear(512, 3)
        
        # Output Rotazione: Matrice 3x3 (9 valori)
        # (Si può continuare con i quaternioni (4) se preferisci, 
        # ma il PDF suggerisce la matrice 3x3 per l'estensione)
        self.rotation_head = nn.Linear(512, 9)

    def forward(self, rgb_crop, depth_crop):
        """
        rgb_crop: Immagine RGB (B, 3, 224, 224)
        depth_crop: Mappa di profondità (B, 3, 224, 224) 
        """
        # Estrazione feature dai due rami
        f_rgb = self.rgb_backbone(rgb_crop)      # Output: 2048
        f_depth = self.depth_backbone(depth_crop) # Output: 512
        
        # Fusione tramite concatenazione
        fused = torch.cat((f_rgb, f_depth), dim=1) # Output: 2560
        
        # Elaborazione tramite MLP
        shared_features = self.fusion_mlp(fused)
        
        # Predizione dei parametri della posa
        translation = self.translation_head(shared_features)
        rotation = self.rotation_head(shared_features)
        
        return translation, rotation