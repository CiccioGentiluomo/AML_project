import torch
from torch.utils.data import Dataset
import os
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class LineModDataset(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, img_size=(224, 224)):
        self.dataset_root = dataset_root
        self.samples = samples      # Lista di (obj_id, img_id)
        self.gt_cache = gt_cache    # Cache dei GT
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]
        
        # Percorsi
        obj_folder = f"{obj_id:02d}"
        img_name = f"{img_id:04d}.png"
        img_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
        
        # 1. Carica Immagine RGB
        img = Image.open(img_path).convert("RGB")
        
        # 2. FILTRAGGIO GT: Cerchiamo l'annotazione specifica per l'oggetto target
        ann_list = self.gt_cache[obj_id][img_id]
        target_ann = None
        for ann in ann_list:
            if ann['obj_id'] == obj_id:
                target_ann = ann
                break
        
        # Fallback di sicurezza (non dovrebbe mai servire)
        if target_ann is None:
            target_ann = ann_list[0]
            
        x, y, w, h = target_ann['obj_bb'] 
        
        # 3. LOGICA SQUARE CROP (Mantiene le proporzioni)
        center_x = x + w / 2
        center_y = y + h / 2
        side = max(w, h)
        
        left = center_x - side / 2
        top = center_y - side / 2
        right = center_x + side / 2
        bottom = center_y + side / 2
        
        img_crop = img.crop((left, top, right, bottom))
        img_resized = img_crop.resize(self.img_size, Image.BILINEAR)
        img_tensor = self.transform(img_resized)
        
        # 4. Pose Ground Truth
        # R è una matrice 3x3, T è un vettore di traslazione 3x1
        R = torch.tensor(target_ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
        T = torch.tensor(target_ann['cam_t_m2c'], dtype=torch.float32)
        
        return {
            "rgb": img_tensor,
            "R": R,
            "T": T,
            "obj_id": obj_id,
            "sample_id": img_id
        }