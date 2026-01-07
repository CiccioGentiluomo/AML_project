import torch
import numpy as np
import cv2
import os
import yaml
import random
import trimesh
from ultralytics import YOLO

# Import dai tuoi file
from models.RGBD_FusionPredictor import RGBD_FusionPredictor
from data.linemod_dataset import LineModDataset
from data.split import prepare_data_and_splits
from utils.rgbd_utils import (
    load_info_cache,
    fetch_sample_info,
    convert_depth_to_meters,
    square_crop_coords,
    prepare_rgb_tensor,
    prepare_depth_tensor,
    build_meta_tensor,
    select_detection_for_object,
    get_object_metadata,
)

def get_all_models_info(root_path):
    """Carica il database globale models_info.yml situato in models/."""
    info_path = os.path.join(root_path, 'models', 'models_info.yml')
    with open(info_path, 'r') as f:
        return yaml.safe_load(f)

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

    if R.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(R)
    else:
        rvec = R
    T_vec = T.reshape(3, 1)

    pts_2d, _ = cv2.projectPoints(pts, rvec, T_vec, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    edges = [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (0,4), (1,5), (2,6), (3,7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img

def main():
    # --- 1. CONFIGURAZIONE PERCORSI ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    RGBD_MODEL_PATH = "pose_rgbd_fusion_best.pth"
    YOLO_PATH = 'runs/detect/linemod_yolo_run/weights/best.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. CARICAMENTO MODELLI ---
    print("🚀 Caricamento modelli in corso...")
    yolo_model = YOLO(YOLO_PATH)

    pose_model = RGBD_FusionPredictor().to(DEVICE)
    pose_model.load_state_dict(torch.load(RGBD_MODEL_PATH, map_location=DEVICE))
    pose_model.eval()

    models_info = get_all_models_info(ROOT_DATASET)
    _, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    val_dataset = LineModDataset(ROOT_DATASET, val_samples, gt_cache)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)

    ply_cache = {}
    for oid in object_ids:
        ply_path = os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(800)
        else:
            print(f"⚠️ File PLY mancante per obj {oid:02d} ({ply_path}).")

    window_name = "Verde=GT, Rosso=Pred (YOLO+RGBD Fusion)"
    print("📸 Pipeline pronta. 'q' per uscire.")

    with torch.no_grad():
        indices = list(range(len(val_dataset)))
        random.shuffle(indices)

        for idx in indices:
            sample = val_dataset[idx]
            obj_id = int(sample["obj_id"])
            sample_id = int(sample["sample_id"])
            obj_info = get_object_metadata(models_info, obj_id)
            if obj_info is None:
                print(f"⚠️ Metadata mancanti per obj {obj_id:02d}.")
                continue

            pts_model = ply_cache.get(obj_id)
            if pts_model is None:
                print(f"⚠️ Nuvola di punti non disponibile per obj {obj_id:02d}.")
                continue

            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                print(f"⚠️ Info camera mancanti per obj {obj_id:02d} sample {sample_id:04d}.")
                continue
            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)
            depth_scale = info_entry.get('depth_scale', 1.0)
            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample['sample_id']:04d}.png")
            depth_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'depth', f"{sample['sample_id']:04d}.png")

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                print(f"⚠️ Immagini mancanti per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            depth_meters = convert_depth_to_meters(depth_raw, depth_scale)

            # --- FASE A: YOLO DETECTION ---
            results = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(results, obj_id)
            if box is None:
                print(f"⚠️ YOLO non ha trovato obj {obj_id:02d} in {img_path}")
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]-xyxy[0]), float(xyxy[3]-xyxy[1])] # [x, y, w, h]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                print(f"⚠️ Crop non valido per obj {obj_id:02d}.")
                continue

            rgb_tensor = prepare_rgb_tensor(img_bgr, crop_coords)
            depth_tensor = prepare_depth_tensor(depth_meters, crop_coords)
            meta_tensor = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape)

            if rgb_tensor is None or depth_tensor is None or meta_tensor is None:
                print(f"⚠️ Preprocess fallito per obj {obj_id:02d}.")
                continue

            # --- FASE B: RGBD FUSION MODEL ---
            pred_T, pred_R_raw = pose_model(
                rgb_tensor.to(DEVICE),
                depth_tensor.to(DEVICE),
                meta_tensor.to(DEVICE)
            )
            R_pred = pred_R_raw.view(3, 3).cpu().numpy()
            T_pred = (pred_T[0].cpu().numpy() * 1000.0).astype(np.float32)  # converti in millimetri

            R_gt = sample["R"].cpu().numpy() if torch.is_tensor(sample["R"]) else np.array(sample["R"], dtype=np.float32)
            T_gt = sample["T"].cpu().numpy() if torch.is_tensor(sample["T"]) else np.array(sample["T"], dtype=np.float32)

            pts_gt = np.dot(pts_model, R_gt.T) + T_gt
            pts_pred = np.dot(pts_model, R_pred.T) + T_pred
            add_error = float(np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1)))
            trans_error = float(np.linalg.norm(T_gt - T_pred))
            rot_trace = np.trace(np.dot(R_pred, R_gt.T))
            rot_angle = np.degrees(np.arccos(np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0)))
            error_summary = f"ADD {add_error:.2f}mm | dT {trans_error:.2f}mm | dR {rot_angle:.2f} gradi"
            print(f"OBJ {obj_id:02d} sample {sample_id:04d} -> {error_summary}")

            # --- FASE D: RENDERING ---
            # GT (Verde)
            vis_img = project_3d_box(
                img_bgr.copy(),
                sample["R"].numpy(),
                sample["T"].numpy(),
                cam_K,
                obj_info,
                color=(0, 255, 0)
            )
            # PRED (Rosso) - Output del modello RGBD fusion
            vis_img = project_3d_box(vis_img, R_pred, T_pred, cam_K, obj_info, color=(0, 0, 255))

            # Disegniamo anche la BBox di YOLO per verifica
            cv2.rectangle(vis_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 1)
            cv2.putText(vis_img, f"OBJ {obj_id} - YOLO BBox", (int(xyxy[0]), int(xyxy[1]-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            overlay_lines = error_summary.split(" | ")
            for idx_line, line in enumerate(overlay_lines):
                y_pos = 20 + idx_line * 22
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, vis_img)
            try:
                cv2.setWindowTitle(window_name, f"{window_name} | {error_summary}")
            except (cv2.error, AttributeError):
                pass
            if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()