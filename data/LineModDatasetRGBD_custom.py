import torch
from torch.utils.data import Dataset
import os
import cv2 
import trimesh
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.rgbd_utils_custom import convert_depth_to_meters, square_crop_coords, prepare_rgb_tensor, prepare_depth_tensor, build_meta_tensor

class LineModDatasetRGBD_custom(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, info_cache, img_size=(224, 224), n_points=500, is_train=False):
        self.dataset_root = dataset_root
        self.samples = samples
        self.gt_cache = gt_cache
        self.info_cache = info_cache
        self.img_size = img_size
        self.n_points = n_points
        self.is_train = is_train

        # --- PIPELINE DI AUGMENTATION (SOLO FOTOMETRICA) ---
        self.transform = A.Compose([
            # 1. Trasformazioni solo per RGB
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ], additional_targets={'depth': 'mask'}) # Trattiamo depth come mask per proteggerla dal colore

        # Crea anche una pipeline "semplice" per la validazione/test
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.model_points_cache = {}
        unique_obj_ids = sorted({obj_id for obj_id, _ in samples})
        for obj_id in unique_obj_ids:
            self.model_points_cache[obj_id] = self._pre_load_model_points(obj_id)

    def _pre_load_model_points(self, obj_id):
        mesh_path = os.path.join(self.dataset_root, 'models', f"obj_{obj_id:02d}.ply")
        mesh = trimesh.load(mesh_path)
        points = mesh.vertices

        if len(points) > self.n_points:
            idx = np.random.choice(len(points), self.n_points, replace=False)
            points = points[idx]

        return torch.from_numpy(points).float() / 1000.0
    
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]
        ann_list = self.gt_cache[obj_id][img_id]
        ann = next((a for a in ann_list if a['obj_id'] == obj_id), ann_list[0])
        info = self.info_cache[obj_id][img_id]
        
        rgb_img = cv2.imread(os.path.join(self.dataset_root, 'data', f"{obj_id:02d}", 'rgb', f"{img_id:04d}.png"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # Convertiamo subito in RGB

        depth_raw = cv2.imread(os.path.join(self.dataset_root, 'data', f"{obj_id:02d}", 'depth', f"{img_id:04d}.png"), cv2.IMREAD_UNCHANGED)
        depth_meters = convert_depth_to_meters(depth_raw, info['depth_scale'])

        crop_coords = square_crop_coords(ann['obj_bb'], rgb_img.shape)
        left, top, right, bottom = crop_coords

        # Crop manuale prima di passare ad Albumentations
        rgb_crop = cv2.resize(rgb_img[top:bottom, left:right], self.img_size)
        depth_crop = cv2.resize(depth_meters[top:bottom, left:right], self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            # APPLICA AUGMENTATION
            augmented = self.transform(image=rgb_crop, depth=depth_crop)
            rgb_tensor = augmented['image']
            depth_tensor = augmented['depth'].float().unsqueeze(0) # Aggiunge canale 1
        else:
            # SOLO CONVERSIONE IN TENSOR (Normalizzazione inclusa se necessario)
            # Nota: Per coerenza puoi usare ToTensorV2() anche qui in una pipeline separata
            augmented = self.val_transform(image=rgb_crop, depth=depth_crop)
            rgb_tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
            depth_tensor = torch.from_numpy(depth_crop).float().unsqueeze(0)

        meta_tensor = build_meta_tensor(ann['obj_bb'], np.array(info['cam_K'], dtype=np.float32).reshape(3, 3), rgb_img.shape).squeeze(0)

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "meta_info": meta_tensor,
            "rotation_9d": torch.tensor(ann['cam_R_m2c'], dtype=torch.float32).flatten(),
            "translation_3d": torch.tensor(ann['cam_t_m2c'], dtype=torch.float32) / 1000.0,
            "R_matrix": torch.tensor(ann['cam_R_m2c'], dtype=torch.float32).view(3, 3),
            "model_points": self.model_points_cache[obj_id],
            "obj_id": torch.tensor(obj_id, dtype=torch.long)
        }
