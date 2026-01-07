import torch
import numpy as np
import cv2
import os
import yaml
import random
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_conv
from torchvision import transforms

# Import dai tuoi file
from models.PosePredictor import PosePredictor
from data.linemod_dataset import LineModDataset
from data.split import prepare_data_and_splits

def get_all_models_info(root_path):
    """Carica il database globale models_info.yml situato in models/."""
    info_path = os.path.join(root_path, 'models', 'models_info.yml')
    with open(info_path, 'r') as f:
        return yaml.safe_load(f)

def compute_pinhole_translation(bbox, intrinsics, real_diameter):
    """Calcola T(X, Y, Z) usando la Bounding Box di YOLO e il diametro reale."""
    x, y, w, h = bbox
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # La profondità Z dipende dal diametro reale e dalla dimensione della box YOLO
    pixel_size = max(w, h)
    if pixel_size == 0: return np.array([0, 0, 0])
    
    Z = (fx * real_diameter) / pixel_size

    # Proiezione inversa per ottenere X e Y
    u_center = x + w / 2
    v_center = y + h / 2
    X = (u_center - cx) * Z / fx
    Y = (v_center - cy) * Z / fy

    return np.array([X, Y, Z])

def project_3d_box(img, R, T, K, obj_info, color=(0, 255, 0), thickness=2):
    """Proietta il box 3D usando i dati reali di min e size."""
    min_x, max_x = obj_info['min_x'], obj_info['min_x'] + obj_info['size_x']
    min_y, max_y = obj_info['min_y'], obj_info['min_y'] + obj_info['size_y']
    min_z, max_z = obj_info['min_z'], obj_info['min_z'] + obj_info['size_z']

    pts = np.array([
        [min_x, min_y, min_z], [min_x, min_y, max_z],
        [min_x, max_y, min_z], [min_x, max_y, max_z],
        [max_x, min_y, min_z], [max_x, min_y, max_z],
        [max_x, max_y, min_z], [max_x, max_y, max_z]
    ], dtype=np.float32)

    pts_2d, _ = cv2.projectPoints(pts, R, T, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    edges = [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (0,4), (1,5), (2,6), (3,7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img

def preprocess_for_resnet(img, bbox):
    """Taglia e prepara il crop dell'immagine per la ResNet-50."""
    x, y, w, h = map(int, bbox)
    # Evitiamo crop fuori dai bordi
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x+w), min(h_img, y+h)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    
    # Resize e normalizzazione (stessa logica del LineModDataset)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop).unsqueeze(0)

def main():
    # --- 1. CONFIGURAZIONE PERCORSI ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    RESNET_PATH = "pose_resnet50_baseline.pth"
    YOLO_PATH = 'runs/detect/linemod_yolo_run/weights/best.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    K = np.array([[572.4114, 0, 325.2611], [0, 573.5704, 242.0489], [0, 0, 1]], dtype=np.float32)
    intrinsics = {'fx': K[0,0], 'fy': K[1,1], 'cx': K[0,2], 'cy': K[1,2]}

    # --- 2. CARICAMENTO MODELLI ---
    print("🚀 Caricamento modelli in corso...")
    yolo_model = YOLO(YOLO_PATH)
    
    pose_model = PosePredictor().to(DEVICE)
    pose_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    pose_model.eval()

    models_info = get_all_models_info(ROOT_DATASET)
    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    test_dataset = LineModDataset(ROOT_DATASET, test_samples, gt_cache)

    print("📸 Pipeline pronta. 'q' per uscire.")

    with torch.no_grad():
        indices = list(range(len(test_dataset)))
        random.shuffle(indices)

        for idx in indices:
            sample = test_dataset[idx]
            obj_id = int(sample["obj_id"])
            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample['sample_id']:04d}.png")
            
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            # --- FASE A: YOLO DETECTION ---
            results = yolo_model(img_bgr, verbose=False)[0]
            if len(results.boxes) == 0:
                print(f"⚠️ YOLO non ha trovato l'oggetto {obj_id} in {img_path}")
                continue

            # Prendiamo la box più sicura (confidence alta)
            box = results.boxes[0]
            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]] # [x, y, w, h]

            # --- FASE B: RESNET ROTATION ---
            input_resnet = preprocess_for_resnet(img_bgr, yolo_bbox)
            if input_resnet is None: continue
            
            pred_quat = pose_model(input_resnet.to(DEVICE)).cpu().numpy()[0]
            pred_quat /= np.linalg.norm(pred_quat)
            R_pred = R_conv.from_quat(pred_quat).as_matrix()

            # --- FASE C: PINHOLE TRANSLATION ---
            obj_info = models_info[obj_id]
            T_pred = compute_pinhole_translation(yolo_bbox, intrinsics, obj_info['diameter'])

            # --- FASE D: RENDERING ---
            # GT (Verde)
            vis_img = project_3d_box(img_bgr.copy(), sample["R"].numpy(), sample["T"].numpy(), K, obj_info, color=(0, 255, 0))
            # PRED (Rosso) - Basata su YOLO + ResNet + Pinhole
            vis_img = project_3d_box(vis_img, R_pred, T_pred, K, obj_info, color=(0, 0, 255))

            # Disegniamo anche la BBox di YOLO per verifica
            cv2.rectangle(vis_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 1)
            cv2.putText(vis_img, f"OBJ {obj_id} - YOLO BBox", (int(xyxy[0]), int(xyxy[1]-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow("Pipeline Finale: Verde=GT, Rosso=Full Pred (YOLO+ResNet+Pinhole)", vis_img)
            if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()