import torch
from torch.utils.data import Dataset
import os
import cv2 
import trimesh
import numpy as np
from utils.rgbd_utils_custom import convert_depth_to_meters, square_crop_coords, prepare_rgb_tensor, prepare_depth_tensor, build_meta_tensor

class LineModDatasetRGBD_custom(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, info_cache, img_size=(224, 224), n_points=500):
        self.dataset_root = dataset_root
        self.samples = samples
        self.gt_cache = gt_cache
        self.info_cache = info_cache
        self.img_size = img_size
        self.n_points = n_points

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
        depth_raw = cv2.imread(os.path.join(self.dataset_root, 'data', f"{obj_id:02d}", 'depth', f"{img_id:04d}.png"), cv2.IMREAD_UNCHANGED)
        
        depth_meters = convert_depth_to_meters(depth_raw, info['depth_scale'])
        crop_coords = square_crop_coords(ann['obj_bb'], rgb_img.shape)
        if crop_coords is None:
            raise ValueError(f"Crop non valido per obj {obj_id} img {img_id}")

        rgb_tensor = prepare_rgb_tensor(rgb_img, crop_coords).squeeze(0)
        depth_tensor = prepare_depth_tensor(depth_meters, crop_coords).squeeze(0)
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