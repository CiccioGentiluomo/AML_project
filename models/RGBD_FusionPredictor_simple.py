import torch
import torch.nn as nn
import torchvision.models as models

class SimpleDepthCNN(nn.Module):
    def __init__(self, out_features=512):
        super(SimpleDepthCNN, self).__init__()
        # Architettura con 5 layer convoluzionali
        self.features = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            
            # Layer 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            
            # Layer 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            
            # Layer 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), 
            
            # Layer 5 (Nuovo): 14 -> 7 (prima del Pooling globale)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Ora l'input della FC è 512 perché l'ultimo layer ha 512 filtri
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

class RGBD_FusionPredictor_Simple(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor_Simple, self).__init__()
        # Ramo RGB (ResNet-50) [cite: 4]
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.rgb_backbone.fc = nn.Identity() 
        
        # Ramo Depth (Simple 1-ch) 
        self.depth_backbone = SimpleDepthCNN(out_features=512)
        
        # Meta Encoder [cite: 5]
        self.meta_encoder = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        
        # Fusion MLP: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624 [cite: 5]
        self.fusion_mlp = nn.Sequential(nn.Linear(2048 + 512 + 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 9)

    def forward(self, rgb_crop, depth_crop, meta_info):
        f_rgb = self.rgb_backbone(rgb_crop)
        f_depth = self.depth_backbone(depth_crop)
        f_meta = self.meta_encoder(meta_info)
        # Concatenazione feature [cite: 5, 13]
        shared = self.fusion_mlp(torch.cat((f_rgb, f_depth, f_meta), dim=1))
        return self.translation_head(shared), self.rotation_head(shared)