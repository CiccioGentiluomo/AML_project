import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2 
from PIL import Image
import torchvision.transforms as transforms
import trimesh

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class LineModDatasetRGBD(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, info_cache, img_size=(224, 224), n_points=500):
        self.dataset_root = dataset_root
        self.samples = samples      # Lista di (obj_id, img_id)
        self.gt_cache = gt_cache    # Cache caricata da gt.yml
        self.info_cache = info_cache # Cache caricata da info.yml
        self.img_size = img_size
        self.n_points = n_points
        
        # Trasformazione standard per il ramo RGB (ResNet-50)
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
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
        K = target_info['cam_K'] # [fx, 0, px, 0, fy, py, 0, 0, 1]
        fx, fy = K[0], K[4]
        px, py = K[2], K[5]

        # 2. Coordinate Bounding Box e preparazione Meta Info
        x, y, w, h = target_ann['obj_bb']
        center_x, center_y = x + w / 2, y + h / 2
        
        # Creazione del vettore di metadati spaziali (normalizzati)
        # Formato: [cx, cy, w, h, fx, fy, px, py]
        meta_info = torch.tensor([
            center_x / 640.0, center_y / 480.0, 
            w / 640.0, h / 480.0,
            fx / 1000.0, fy / 1000.0,
            px / 640.0, py / 480.0
        ], dtype=torch.float32)

        # 3. Percorsi file
        rgb_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
        depth_path = os.path.join(self.dataset_root, 'data', obj_folder, 'depth', img_name)
        
        # 4. Processamento Profondità (Depth)
        depth_img_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(depth_img_raw.shape) == 3:
            depth_img_raw = depth_img_raw[:, :, 0]
        
        # Conversione in metri
        depth_meters = (depth_img_raw * depth_scale) / 1000.0

        # 5. Caricamento RGB
        rgb_img = Image.open(rgb_path).convert("RGB")
        
        # 6. Logica Square Crop
        side = max(w, h)
        left, top = center_x - side / 2, center_y - side / 2
        right, bottom = center_x + side / 2, center_y + side / 2
        
        # Crop e Resize RGB
        rgb_crop = rgb_img.crop((left, top, right, bottom))
        rgb_resized = rgb_crop.resize(self.img_size, Image.BILINEAR)
        rgb_tensor = self.rgb_transform(rgb_resized)
        
        # Crop e Resize DEPTH
        H, W = depth_meters.shape[:2]
        l, t, r, b = int(max(0, left)), int(max(0, top)), int(min(W, right)), int(min(H, bottom))
        depth_crop = depth_meters[t:b, l:r]
        
        depth_resized = cv2.resize(depth_crop, self.img_size, interpolation=cv2.INTER_NEAREST)
        depth_3ch = np.repeat(depth_resized[np.newaxis, :, :], 3, axis=0)
        depth_tensor = torch.from_numpy(depth_3ch).float() 
        
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