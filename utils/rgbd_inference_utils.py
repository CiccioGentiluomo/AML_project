import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import yaml

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)

RGB_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_info_cache(dataset_root, object_ids):
    info_cache = {}
    for obj_id in object_ids:
        obj_folder = f"{obj_id:02d}"
        info_path = os.path.join(dataset_root, 'data', obj_folder, 'info.yml')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info_cache[obj_id] = yaml.safe_load(f)
    return info_cache

def fetch_sample_info(info_cache, obj_id, sample_id):
    obj_info = info_cache.get(obj_id, {})
    for key in (sample_id, str(sample_id), f"{sample_id:04d}"):
        if key in obj_info:
            return obj_info[key]
    return None

def convert_depth_to_meters(depth_raw, depth_scale):
    depth = depth_raw.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return (depth * float(depth_scale)) / 1000.0

def square_crop_coords(bbox, img_shape):
    h_img, w_img = img_shape[:2]
    x, y, w, h = bbox
    side = max(w, h)
    if side <= 1:
        return None

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    left = int(max(0, np.floor(center_x - side / 2.0)))
    top = int(max(0, np.floor(center_y - side / 2.0)))
    right = int(min(w_img, np.ceil(center_x + side / 2.0)))
    bottom = int(min(h_img, np.ceil(center_y + side / 2.0)))

    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom

def prepare_rgb_tensor(img_bgr, crop_coords):
    left, top, right, bottom = crop_coords
    crop = img_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    pil_img = Image.fromarray(crop)
    return RGB_TRANSFORM(pil_img).unsqueeze(0)

def prepare_depth_tensor(depth_meters, crop_coords):
    left, top, right, bottom = crop_coords
    depth_crop = depth_meters[top:bottom, left:right]
    if depth_crop.size == 0:
        return None

    depth_resized = cv2.resize(depth_crop, INPUT_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    depth_3ch = np.repeat(depth_resized[np.newaxis, :, :], 3, axis=0)
    return torch.from_numpy(depth_3ch).float().unsqueeze(0)

def build_meta_tensor(bbox, cam_K, img_shape):
    h_img, w_img = img_shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    meta = torch.tensor([
        center_x / w_img,
        center_y / h_img,
        w / w_img,
        h / h_img,
        cam_K[0, 0] / 1000.0,
        cam_K[1, 1] / 1000.0,
        cam_K[0, 2] / w_img,
        cam_K[1, 2] / h_img,
    ], dtype=torch.float32)
    return meta.unsqueeze(0)

def select_detection_for_object(result, obj_id):
    target_cls = obj_id - 1
    best_box, best_conf = None, -1.0
    for box in result.boxes:
        if int(box.cls) != target_cls:
            continue
        conf = float(box.conf)
        if conf > best_conf:
            best_box, best_conf = box, conf
    return best_box

def get_object_metadata(models_info, obj_id):
    return models_info.get(obj_id) or models_info.get(str(obj_id))
