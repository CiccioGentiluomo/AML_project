import torch
from torch.utils.data import Dataset
import os
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from utils.resNetUtils import matrix_to_quaternion

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
            
            # 1. Costruzione Percorsi
            obj_folder = f"{obj_id:02d}"
            img_name = f"{img_id:04d}.png"
            img_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
            
            # 2. Caricamento Immagine
            img = Image.open(img_path).convert("RGB")
            
            # 3. Recupero Annotazioni Ground Truth (GT) [cite: 160]
            ann_list = self.gt_cache[obj_id][img_id]
            target_ann = next((ann for ann in ann_list if ann['obj_id'] == obj_id), ann_list[0])
                
            x, y, w, h = target_ann['obj_bb'] 
            
            # 4. Logica Square Crop (Mantiene le proporzioni per la ResNet)
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
            
            # 5. Elaborazione della Posa (Rotazione e Traslazione) [cite: 158]
            # R è una matrice 3x3
            R_mat = torch.tensor(target_ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
            # T è un vettore di traslazione 3x1
            T = torch.tensor(target_ann['cam_t_m2c'], dtype=torch.float32)
            
            # CONVERSIONE: Otteniamo il quaternione target per l'addestramento 
            quaternion_gt = matrix_to_quaternion(R_mat)
            
            return {
                "rgb": img_tensor,           # Input per ResNet-50 [cite: 173]
                "quaternion": quaternion_gt, # Target per la rotation loss [cite: 175]
                "R": R_mat,                  # Matrice GT per valutazione ADD [cite: 184]
                "T": T,                      # Traslazione GT per valutazione ADD
                "bbox_originale": torch.tensor([x, y, w, h]), # Necessaria per il metodo Pinhole [cite: 48]
                "obj_id": obj_id,
                "sample_id": img_id
            }