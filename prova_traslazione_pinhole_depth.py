import numpy as np
import cv2
import yaml
import os

def calculate_translation_from_depth(depth_img, bbox, K, depth_scale=1.0):
    """
    Calcola X, Y, Z partendo dalla mappa di profondità.
    Formula basata sul modello Pinhole inverso.
    """
    x, y, w, h = bbox
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # 1. Ritaglio della mappa di profondità nell'area della BBox [cite: 193]
    depth_roi = depth_img[int(y):int(y+h), int(x):int(x+w)]

    # 2. Filtriamo i valori zero (pixel senza profondità o sfondo)
    valid_depths = depth_roi[depth_roi > 0]

    if len(valid_depths) == 0:
        return None

    # 3. Calcolo di Z tramite mediana per robustezza al rumore
    z_pred = np.median(valid_depths) * depth_scale

    # 4. Calcolo del centro della BBox in coordinate pixel
    u_c = x + w / 2
    v_c = y + h / 2

    # 5. Proiezione inversa per ottenere X e Y 
    # Usiamo la formula: X = ((u - cx) * Z) / fx
    x_pred = (u_c - cx) * z_pred / fx
    y_pred = (v_c - cy) * z_pred / fy

    return np.array([x_pred, y_pred, z_pred])

# --- CONFIGURAZIONE TEST ---
ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
OBJ_ID = "01"  
IMG_ID = "0120"
DEPTH_SCALE = 1.0  # LineMod preprocessed usa millimetri 

# Matrice Intrinseca K (Standard LineMod) [cite: 160]
K = np.array([[572.4114, 0, 325.2611], 
              [0, 573.5704, 242.0489], 
              [0, 0, 1]])

# 1. Caricamento Annotazioni GT
gt_path = os.path.join(ROOT_DATASET, "data", OBJ_ID, "gt.yml")
with open(gt_path, 'r') as f:
    gt_data = yaml.safe_load(f)

# 2. Caricamento Immagine Depth (16-bit PNG) 
depth_path = os.path.join(ROOT_DATASET, "data", OBJ_ID, "depth", f"{IMG_ID}.png")
depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth_map is None:
    print(f"Errore: Immagine depth non trovata in {depth_path}")
else:
    # Recupero BBox GT e Traslazione GT per il confronto [cite: 160]
    # gt_data[0] corrisponde alla prima immagine (0000)
    target_ann = gt_data[int(IMG_ID)][0] 
    bbox_gt = target_ann['obj_bb'] # [x, y, w, h]
    t_gt = np.array(target_ann['cam_t_m2c']) # [X, Y, Z] in mm

    # 3. Esecuzione Calcolo
    t_pred = calculate_translation_from_depth(depth_map, bbox_gt, K, DEPTH_SCALE)

    if t_pred is not None:
        error = np.abs(t_gt - t_pred)
        
        print(f"\n--- CONFRONTO TRASLAZIONE (ID:{OBJ_ID} IMG:{IMG_ID}) ---")
        print(f"GT T:         {t_gt}")
        print(f"PREDETTA T:   {t_pred}")
        print("-" * 40)
        print(f"ERRORE (mm):  {error}")
        print(f"ERRORE TOT:   {np.linalg.norm(error):.2f} mm")
    else:
        print("Impossibile calcolare la traslazione: nessun pixel depth valido nella BBox.")