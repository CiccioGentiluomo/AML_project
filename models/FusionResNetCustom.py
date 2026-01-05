import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- BLOCCO RESIDUALE (SKIP CONNECTION) ---
# Fondamentale per facilitare il flusso del gradiente durante l'addestramento da zero
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Percorso principale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Percorso Shortcut (Skip Connection)
        self.shortcut = nn.Identity()
        # Se le dimensioni cambiano (stride > 1 o canali diversi), adattiamo l'identità con una conv 1x1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Somma della skip connection
        return F.relu(out)

# --- BACKBONE DEPTH: SIMIL RESNET (1-CANALE) ---
# Architettura a 5 strati ottimizzata per dati geometrici
class SimpleResNet1ch(nn.Module):
    def __init__(self, out_features=512):
        super(SimpleResNet1ch, self).__init__()
        
        # Layer 1: Convoluzione iniziale (porta 1 canale a 64)
        self.start = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Layer 2-5: Blocchi Residui per estrarre feature complesse
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 112 -> 56
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 56 -> 28
        self.layer4 = ResidualBlock(256, 512, stride=2)  # 28 -> 14
        self.layer5 = ResidualBlock(512, 512, stride=1)  # Raffinamento finale
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.start(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))

# --- MODELLO DI FUSIONE COMPLETO ---
class RGBD_FusionPredictor_Simple(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor_Simple, self).__init__()
        
        # 1. Ramo RGB: ResNet-50 pre-addestrata (3 canali)
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features_rgb = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity() 
        
        # 2. Ramo DEPTH: SimpleResNet1ch (1 canale, addestrata da zero)
        self.depth_backbone = SimpleResNet1ch(out_features=512)
        
        # 3. Ramo METADATI (BBox e Camera Info)
        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Dimensione totale: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624
        combined_features = num_features_rgb + 512 + 64
        
        # 4. Pose Estimator (MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Heads per Traslazione (3) e Rotazione (9)
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 9)

    def forward(self, rgb_crop, depth_crop, meta_info):
        # Estrazione feature dai rami paralleli
        f_rgb = self.rgb_backbone(rgb_crop)      # (B, 2048)
        f_depth = self.depth_backbone(depth_crop) # (B, 512)
        f_meta = self.meta_encoder(meta_info)    # (B, 64)
        
        # Fusione per concatenazione
        fused = torch.cat((f_rgb, f_depth, f_meta), dim=1)
        shared = self.fusion_mlp(fused)
        
        # Predizione posa 6D
        translation = self.translation_head(shared)
        rotation = self.rotation_head(shared)
        
        return translation, rotation