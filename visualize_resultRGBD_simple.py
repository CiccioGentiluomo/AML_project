import os
import random
import cv2
import numpy as np
import torch
import yaml
import trimesh
from ultralytics import YOLO

from models.FusionResNetCustom import RGBD_FusionPredictor_Simple
from data.split import prepare_data_and_splits
from utils.rgbd_inference_utils_simple import (
    load_info_cache,
    fetch_sample_info,
    convert_depth_to_meters,
    square_crop_coords,
    prepare_rgb_tensor,
    prepare_depth_tensor,
    build_meta_tensor,
)
from utils.rgbd_inference_utils import get_object_metadata, select_detection_for_object


def get_all_models_info(root_path):
    info_path = os.path.join(root_path, "models", "models_info.yml")
    with open(info_path, "r") as f:
        return yaml.safe_load(f)


def get_annotation(gt_cache, obj_id, sample_id):
    obj_gt = gt_cache.get(obj_id)
    if obj_gt is None:
        return None

    ann_list = None
    for key in (sample_id, str(sample_id), f"{sample_id:04d}"):
        if key in obj_gt:
            ann_list = obj_gt[key]
            break

    if ann_list is None and isinstance(obj_gt, list) and sample_id < len(obj_gt):
        ann_list = obj_gt[sample_id]

    if ann_list is None:
        return None
    if isinstance(ann_list, list):
        return next((ann for ann in ann_list if ann.get("obj_id") == obj_id), ann_list[0])
    return ann_list


def project_3d_box(img, R, T, K, obj_info, color=(0, 255, 0), thickness=2):
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

    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img


def main():
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    MODEL_PATH = "pose_rgbd_simple_1ch_best.pth"
    YOLO_PATH = "runs/detect/linemod_yolo_run/weights/best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Caricamento modello RGBD simple...")
    pose_model = RGBD_FusionPredictor_Simple().to(DEVICE)
    pose_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    pose_model.eval()

    print("🛰️ Caricamento YOLO...")
    yolo_model = YOLO(YOLO_PATH)

    _, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    models_info = get_all_models_info(ROOT_DATASET)

    ply_cache = {}
    for oid in object_ids:
        ply_path = os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(500)
        else:
            print(f"⚠️ File PLY mancante per obj {oid:02d} ({ply_path}).")

    window_name = "Verde=GT, Rosso=Pred (RGBD Simple)"
    print("📸 Pipeline pronta. 'q' per uscire.")

    with torch.no_grad():
        indices = list(range(len(val_samples)))
        random.shuffle(indices)

        for idx in indices:
            obj_id, sample_id = val_samples[idx]
            obj_info = get_object_metadata(models_info, obj_id)
            if obj_info is None:
                print(f"⚠️ Metadata mancanti per obj {obj_id:02d}.")
                continue

            pts_model = ply_cache.get(obj_id)
            if pts_model is None:
                print(f"⚠️ Nuvola di punti non disponibile per obj {obj_id:02d}.")
                continue

            ann = get_annotation(gt_cache, obj_id, sample_id)
            if ann is None:
                print(f"⚠️ Annotazione mancante per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                print(f"⚠️ Info camera mancanti per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            bbox_gt = ann.get('obj_bb')

            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)
            depth_scale = info_entry.get('depth_scale', 1.0)

            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample_id:04d}.png")
            depth_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'depth', f"{sample_id:04d}.png")

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                print(f"⚠️ Immagini mancanti per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            depth_meters = convert_depth_to_meters(depth_raw, depth_scale)

            yolo_result = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(yolo_result, obj_id)
            if box is None:
                print(f"⚠️ YOLO non ha trovato obj {obj_id:02d} in {img_path}")
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2] - xyxy[0]),
                float(xyxy[3] - xyxy[1])
            ]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                print(f"⚠️ Crop non valido (YOLO) per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            rgb_tensor = prepare_rgb_tensor(img_bgr, crop_coords)
            depth_tensor = prepare_depth_tensor(depth_meters, crop_coords)
            meta_tensor = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape)

            if rgb_tensor is None or depth_tensor is None or meta_tensor is None:
                print(f"⚠️ Preprocess fallito per obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            pred_T, pred_R_raw = pose_model(
                rgb_tensor.to(DEVICE),
                depth_tensor.to(DEVICE),
                meta_tensor.to(DEVICE)
            )
            R_pred = pred_R_raw.view(3, 3).cpu().numpy()
            T_pred = (pred_T[0].cpu().numpy() * 1000.0).astype(np.float32)

            R_gt = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            T_gt = np.array(ann['cam_t_m2c'], dtype=np.float32)

            pts_gt = np.dot(pts_model, R_gt.T) + T_gt
            pts_pred = np.dot(pts_model, R_pred.T) + T_pred
            add_error = float(np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1)))
            trans_error = float(np.linalg.norm(T_gt - T_pred))
            rot_trace = np.trace(np.dot(R_pred, R_gt.T))
            rot_angle = np.degrees(np.arccos(np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0)))
            error_summary = f"ADD {add_error:.2f}mm | dT {trans_error:.2f}mm | dR {rot_angle:.2f} gradi"
            print(f"OBJ {obj_id:02d} sample {sample_id:04d} -> {error_summary}")

            vis_img = project_3d_box(
                img_bgr.copy(),
                R_gt,
                T_gt,
                cam_K,
                obj_info,
                color=(0, 255, 0)
            )
            vis_img = project_3d_box(vis_img, R_pred, T_pred, cam_K, obj_info, color=(0, 0, 255))

            if bbox_gt is not None:
                x_gt, y_gt, w_gt, h_gt = bbox_gt
                cv2.rectangle(vis_img, (int(x_gt), int(y_gt)), (int(x_gt + w_gt), int(y_gt + h_gt)), (255, 255, 0), 1)
                cv2.putText(vis_img, f"OBJ {obj_id} - GT BBox", (int(x_gt), max(15, int(y_gt) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            x_det, y_det, w_det, h_det = yolo_bbox
            cv2.rectangle(vis_img, (int(x_det), int(y_det)), (int(x_det + w_det), int(y_det + h_det)), (255, 0, 0), 1)
            cv2.putText(vis_img, "YOLO BBox", (int(x_det), max(15, int(y_det) - 5)),
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
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
