import torch
import numpy as np
import os
import yaml
import pandas as pd
import trimesh
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_conv
from torchvision import transforms

# Import dai tuoi file
from models.PosePredictor import PosePredictor
from data.linemod_dataset import LineModDataset
from data.split import prepare_data_and_splits

def get_all_models_info(root_path):
    info_path = os.path.join(root_path, 'models', 'models_info.yml')
    with open(info_path, 'r') as f:
        return yaml.safe_load(f)

def load_object_points(root_path, obj_id, n_points=500):
    ply_path = os.path.join(root_path, 'models', f'obj_{obj_id:02d}.ply')
    mesh = trimesh.load(ply_path)
    return mesh.sample(n_points)

def preprocess_yolo_crop(img, bbox):
    """Esegue il cropping e il preprocessing per la ResNet-50."""
    x, y, w, h = map(int, bbox)
    h_img, w_img = img.shape[:2]
    # Clipping dei bordi per evitare errori
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w_img, x+w), min(h_img, y+h)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    
    # Resize a 224x224 e normalizzazione (stessa logica di training)
    crop = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop).unsqueeze(0)

def compute_add_rot_only(pts_3d, R_gt, R_pred):
    """Calcola ADD ignorando la traslazione (rot-only)."""
    pts_gt = np.dot(pts_3d, R_gt.T)
    pts_pred = np.dot(pts_3d, R_pred.T)
    return np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1))

def evaluate_yolo_resnet_rot():
    # --- 1. CONFIGURAZIONE ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    RESNET_PATH = "pose_resnet50_baseline.pth"
    YOLO_PATH = 'runs/detect/linemod_yolo_run/weights/best.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 
                   8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 
                   12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}

    # Caricamento modelli
    yolo_model = YOLO(YOLO_PATH)
    pose_model = PosePredictor().to(DEVICE)
    pose_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    pose_model.eval()
    
    models_info = get_all_models_info(ROOT_DATASET)
    obj_points_cache = {oid: load_object_points(ROOT_DATASET, oid) for oid in class_names.keys()}
    _, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    val_dataset = LineModDataset(ROOT_DATASET, val_samples, gt_cache)

    results = []

    # --- 2. LOOP DI VALUTAZIONE ---
    print(f"📊 Valutazione ROT-ONLY con BBox di YOLO...")
    
    with torch.no_grad():
        for batch in tqdm(val_dataset):
            obj_id = int(batch["obj_id"])
            if obj_id not in class_names: continue

            # Caricamento immagine intera per YOLO
            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{batch['sample_id']:04d}.png")
            img_bgr = cv2.imread(img_path)
            
            # A. YOLO Detection
            yolo_results = yolo_model(img_bgr, verbose=False)[0]
            if len(yolo_results.boxes) == 0:
                continue # Salta se l'oggetto non è rilevato

            # BBox YOLO [x, y, w, h]
            box = yolo_results.boxes[0]
            xyxy = box.xyxy.cpu().numpy()[0]
            y_bbox = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]

            # B. Cropping e ResNet Prediction
            input_resnet = preprocess_yolo_crop(img_bgr, y_bbox)
            if input_resnet is None: continue
            
            pred_quat = pose_model(input_resnet.to(DEVICE)).cpu().numpy()[0]
            pred_quat /= np.linalg.norm(pred_quat)
            R_pred = R_conv.from_quat(pred_quat).as_matrix()

            # C. Calcolo Metrica (Rot-only)
            pts = obj_points_cache[obj_id]
            R_gt = batch["R"].numpy()
            
            add_rot_val = compute_add_rot_only(pts, R_gt, R_pred)
            is_correct = add_rot_val < (0.1 * models_info[obj_id]['diameter'])
            
            results.append({
                "class_name": f"{obj_id:02d} - {class_names[obj_id]}",
                "add": add_rot_val,
                "correct": is_correct
            })

    # --- 3. TABELLA FINALE ---
    df = pd.DataFrame(results)
    summary = df.groupby("class_name").agg(
        Media_ADD_rot=("add", "mean"),
        Accuracy=("correct", lambda x: x.mean() * 100)
    ).reset_index()

    print("\n" + "="*60)
    print(summary.to_string(index=False, formatters={'Media_ADD_rot': '{:,.2f}'.format, 'Accuracy': '{:,.1f}'.format}))
    print("="*60)
    print(f"Media globale ADD (rot-only): {df['add'].mean():.2f}")
    print(f"Accuracy globale (rot-only) (%): {df['correct'].mean()*100:.1f}")

if __name__ == "__main__":
    evaluate_yolo_resnet_rot()