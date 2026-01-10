import torch
import torch.nn as nn
import torchvision.models as models

class RGBD_FusionPredictor(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor, self).__init__()
        
        # 1. Ramo RGB: ResNet-50
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features_rgb = self.rgb_backbone.fc.in_features # 2048
        self.rgb_backbone.fc = nn.Identity() 
        
        # 2. Ramo DEPTH: ResNet-18
        self.depth_backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_features_depth = self.depth_backbone.fc.in_features # 512
        self.depth_backbone.fc = nn.Identity()
        
        # 3. Ramo METADATI (Info Camera e BBox)
        # Il vettore conterrà: [cx, cy, w, h, fx, fy, px, py] -> 8 valori
        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Dimensione totale: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624
        combined_features = num_features_rgb + num_features_depth + 64
        
        # 4. Pose Estimator (MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Output Traslazione: 3 coordinate (X, Y, Z)
        self.translation_head = nn.Linear(256, 3)
        
        # Output Rotazione: Matrice 3x3 (9 valori) [cite: 81]
        self.rotation_head = nn.Linear(256, 9)

    def forward(self, rgb_crop, depth_crop, meta_info):
        """
        rgb_crop: (B, 3, 224, 224)
        depth_crop: (B, 3, 224, 224)
        meta_info: (B, 8) -> [cx, cy, w, h, fx, fy, px, py] normalizzati
        """
        # Estrazione feature visuali e geometriche
        f_rgb = self.rgb_backbone(rgb_crop)      # 2048
        f_depth = self.depth_backbone(depth_crop) # 512
        
        # Codifica dei metadati spaziali
        f_meta = self.meta_encoder(meta_info)     # 64
        
        # Fusione per concatenazione (come richiesto dal PDF) [cite: 80]
        fused = torch.cat((f_rgb, f_depth, f_meta), dim=1) # 2624
        
        # Elaborazione finale
        shared = self.fusion_mlp(fused)
        
        # Predizione posa 6D [cite: 12]
        translation = self.translation_head(shared)
        rotation = self.rotation_head(shared)
        
        return translation, rotation