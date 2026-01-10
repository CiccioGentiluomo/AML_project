import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2 
import trimesh
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.rgbd_utils import (
    convert_depth_to_meters,
    square_crop_coords,
    prepare_rgb_tensor,
    prepare_depth_tensor,
    build_meta_tensor,
)

class LineModDatasetRGBD(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, info_cache, img_size=(224, 224), n_points=500, is_train=False):
        self.dataset_root = dataset_root
        self.samples = samples      # Lista di (obj_id, img_id)
        self.gt_cache = gt_cache    # Cache caricata da gt.yml
        self.info_cache = info_cache # Cache caricata da info.yml
        self.img_size = img_size
        self.n_points = n_points
        self.is_train = is_train

        # --- PIPELINE DI AUGMENTATION (SOLO FOTOMETRICA) ---
        self.transform = A.Compose([
            # 1. Trasformazioni solo per RGB
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ], additional_targets={'depth': 'mask'})
        
        # Crea anche una pipeline "semplice" per la validazione/test
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'depth': 'mask'})

        # --- CACHE DEI MODELLI 3D ---
        # Pre-carichiamo i punti per ogni oggetto per non leggere il disco ogni volta
        self.model_points_cache = {}
        unique_obj_ids = set([s[0] for s in samples])
        for obj_id in unique_obj_ids:
            self.model_points_cache[obj_id] = self._pre_load_model_points(obj_id)


    def _pre_load_model_points(self, obj_id):
            """Carica il modello .ply e campiona i punti in metri."""
            obj_folder = f"{obj_id:02d}"
            ply_path = os.path.join(self.dataset_root, 'models',  f"obj_{obj_folder}.ply")
            
            mesh = trimesh.load(ply_path)
            points = mesh.vertices
            
            if len(points) > self.n_points:
                idx = np.random.choice(len(points), self.n_points, replace=False)
                points = points[idx]
                
            # IMPORTANTE: Conversione da mm a metri per coerenza con la Depth
            return torch.from_numpy(points).float() / 1000.0
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]
        obj_folder = f"{obj_id:02d}"
        img_name = f"{img_id:04d}.png"
        
        ann_list = self.gt_cache[obj_id][img_id]
        target_ann = next((item for item in ann_list if item['obj_id'] == obj_id), ann_list[0])
        target_info = self.info_cache[obj_id][img_id]
        
        # Recupero della scala e dei parametri intrinseci K
        depth_scale = target_info['depth_scale']
        K = np.array(target_info['cam_K'], dtype=np.float32).reshape(3, 3)

        # 2. Coordinate Bounding Box e preparazione Meta Info
        x, y, w, h = target_ann['obj_bb']
        
        # 3. Percorsi file
        rgb_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
        depth_path = os.path.join(self.dataset_root, 'data', obj_folder, 'depth', img_name)
        
        rgb_img = cv2.imread(rgb_path)
        depth_img_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_img is None or depth_img_raw is None:
            raise FileNotFoundError(f"Immagini mancanti per obj {obj_id} img {img_id}")

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # Albumentations lavora in RGB
        
        depth_meters = convert_depth_to_meters(depth_img_raw, depth_scale)

        # 6. Logica Square Crop
        crop_coords = square_crop_coords([x, y, w, h], rgb_img.shape)
        l, t, r, b = crop_coords
        
        rgb_crop = rgb_img[t:b, l:r]
        depth_crop = depth_meters[t:b, l:r]

        # 4. Resize manuale a dimensione target per Albumentations
        rgb_crop = cv2.resize(rgb_crop, self.img_size, interpolation=cv2.INTER_LINEAR)
        depth_crop = cv2.resize(depth_crop, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 5. APPLICAZIONE AUGMENTATION
        t = self.transform if self.is_train else self.val_transform
        
        # 2. APPLICA QUELLA SELEZIONATA
        augmented = t(image=rgb_crop, depth=depth_crop)
        
        rgb_tensor = augmented['image']
        depth_tensor = augmented['depth']

        if torch.is_tensor(depth_tensor):
            depth_np = depth_tensor.cpu().numpy()
        else:
            depth_np = depth_tensor

        depth_3ch = np.repeat(depth_np[np.newaxis, :, :], 3, axis=0)
        depth_tensor = torch.from_numpy(depth_3ch).float()

        meta_tensor = build_meta_tensor([x, y, w, h], K, rgb_img.shape)
        meta_info = meta_tensor.squeeze(0)
        
        # 7. Target della Posa e Modello 3D
        R_mat = torch.tensor(target_ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
        T_vec = torch.tensor(target_ann['cam_t_m2c'], dtype=torch.float32) / 1000.0
        model_points = self.model_points_cache[obj_id]
        
        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "meta_info": meta_info,
            "rotation_9d": R_mat.flatten(),
            "translation_3d": T_vec,
            "obj_id": obj_id,
            "R_matrix": R_mat,
            "model_points": model_points
        }