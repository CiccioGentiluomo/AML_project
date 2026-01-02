import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

# Import necessari dai tuoi file
from models.RGBD_FusionPredictor import RGBD_FusionPredictor
from data.LineModDatasetRGBD import LineModDatasetRGBD
from data.split import prepare_data_and_splits

# --- FUNZIONI DI UTILITÀ GEOMETRICA ---

def get_3d_bbox_and_edges(points):
    """
    Calcola gli 8 vertici del BBox 3D e definisce gli spigoli per disegnarlo.
    """
    min_pts = torch.min(points, dim=0)[0]
    max_pts = torch.max(points, dim=0)[0]

    def corner(x_idx, y_idx, z_idx):
        xs = [min_pts[0], max_pts[0]]
        ys = [min_pts[1], max_pts[1]]
        zs = [min_pts[2], max_pts[2]]
        return torch.stack([xs[x_idx], ys[y_idx], zs[z_idx]])

    # 8 corners del cubo 3D (ordinamento coerente con gli spigoli definiti sotto)
    corners = torch.stack([
        corner(0, 0, 0), # 0: min, min, min
        corner(0, 0, 1), # 1: min, min, max
        corner(0, 1, 0), # 2: min, max, min
        corner(0, 1, 1), # 3: min, max, max
        corner(1, 0, 0), # 4: max, min, min
        corner(1, 0, 1), # 5: max, min, max
        corner(1, 1, 0), # 6: max, max, min
        corner(1, 1, 1)  # 7: max, max, max
    ], dim=0)

    # Lista di coppie di indici che formano gli spigoli del cubo
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    return corners, edges

def project_points_to_image(points_3d, R, T, K):
    """
    Proietta punti 3D (N, 3) sul piano immagine 2D usando la K originale.
    Restituisce punti 2D (N, 2) in coordinate pixel.
    """
    # Trasformazione Rigida: P_cam = R * P_model + T
    # Nota: usiamo .t() perché i punti sono righe (N,3)
    points_cam = torch.mm(points_3d, R.t()) + T.view(1, 3)
    
    # Proiezione Prospettica: P_img = K * P_cam
    # Nota: K è (3,3), points_cam è (N,3). Facciamo (N,3) x (3,3)
    points_2d_hom = torch.mm(points_cam, K.t())
    
    # Normalizzazione per Z (da omogenee a cartesiane)
    # Aggiungiamo un epsilon per evitare divisioni per zero
    z_coords = points_2d_hom[:, 2:3] + 1e-8
    points_2d = points_2d_hom[:, :2] / z_coords
    
    return points_2d.cpu().numpy()

def draw_cube(ax, points_2d, edges, color, label_text):
    """Disegna gli spigoli del cubo su un asse Matplotlib."""
    # Disegna i vertici
    ax.scatter(points_2d[:, 0], points_2d[:, 1], color=color, s=20, label=label_text)
    # Disegna gli spigoli
    for start_idx, end_idx in edges:
        p1 = points_2d[start_idx]
        p2 = points_2d[end_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2)

# --- FUNZIONE PRINCIPALE DI VISUALIZZAZIONE ---

def visualize_complete_evaluation():
    # CONFIGURAZIONE
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    MODEL_PATH = "pose_rgbd_fusion_best.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento Dati e Modello
    _, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    # Serve la cache per il dataset
    from trainRGBD import load_info_cache
    info_cache = load_info_cache(ROOT_DATASET, sorted(gt_cache.keys()))
    val_set = LineModDatasetRGBD(ROOT_DATASET, val_samples, gt_cache, info_cache)

    model = RGBD_FusionPredictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

# 2. Selezione Campione Casuale
    idx = random.randint(0, len(val_set) - 1)
    batch = val_set[idx]
    
    # Prepariamo i tensori per la rete (aggiungendo la dimensione del batch .unsqueeze(0))
    rgb_tensor = batch["rgb"].unsqueeze(0).to(DEVICE)
    depth_tensor = batch["depth"].unsqueeze(0).to(DEVICE)
    meta_tensor = batch["meta_info"].unsqueeze(0).to(DEVICE)
    
    # Dati reali (GT)
    gt_R = batch["R_matrix"].to(DEVICE)
    gt_T = batch["translation_3d"].to(DEVICE)
    all_model_points = batch["model_points"].to(DEVICE)
    
    # CORREZIONE: obj_id è già un int, non serve .item()
    obj_id = batch["obj_id"] 
    print(f"Analyzing sample index: {idx}, Object ID: {obj_id}")

    # 3. Inferenza della Rete
    with torch.no_grad():
        pred_T, pred_R_raw = model(rgb_tensor, depth_tensor, meta_tensor)
        pred_R = pred_R_raw.view(3, 3) # Formato 3x3 per calcoli geometrici
        pred_T = pred_T[0]

    # --- CARICAMENTO IMMAGINE ORIGINALE ---
    # I campioni sono tuple (obj_id, img_id), quindi costruiamo il percorso RGB direttamente
    sample_obj_id, sample_img_id = val_set.samples[idx]
    obj_folder = f"{sample_obj_id:02d}"
    img_stem = f"{sample_img_id:04d}"
    rgb_path = os.path.join(val_set.dataset_root, 'data', obj_folder, 'rgb', f"{img_stem}.png")

    if not os.path.exists(rgb_path):
        print(f"⚠️ Immagine originale non trovata in {rgb_path}. Uso il crop normalizzato.")
        orig_img_rgb = batch["rgb"].permute(1, 2, 0).cpu().numpy()
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        orig_img_rgb = (orig_img_rgb * std) + mean
        orig_img_rgb = np.clip(orig_img_rgb, 0, 1)
    else:
        orig_img_bgr = cv2.imread(rgb_path)
        orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

    # --- PREPARAZIONE DATI ORIGINALI (NON RITAGLIATI) ---
    
    # A) Ricostruzione Matrice K Originale (640x480) dai metadati
    # meta_info è [cx_n, cy_n, w_n, h_n, fx_n, fy_n, px_n, py_n]
    meta_cpu = batch["meta_info"].cpu()
    K_orig = torch.tensor([
        [meta_cpu[4]*1000, 0, meta_cpu[6]*640],
        [0, meta_cpu[5]*1000, meta_cpu[7]*480],
        [0, 0, 1]
    ], dtype=torch.float32).to(DEVICE)

    # B) Assicuriamoci di avere un'immagine RGB utilizzabile (già caricata sopra)

    # --- VISUALIZZAZIONE 1: CUBI 3D (Bounding Box) ---
    
    bbox_3d_points, cube_edges = get_3d_bbox_and_edges(all_model_points)
    
    # Proiezione
    bbox_2d_gt = project_points_to_image(bbox_3d_points, gt_R, gt_T, K_orig)
    bbox_2d_pred = project_points_to_image(bbox_3d_points, pred_R, pred_T, K_orig)

    # Plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(orig_img_rgb)
    draw_cube(ax1, bbox_2d_gt, cube_edges, 'lime', 'GT BBox')
    draw_cube(ax1, bbox_2d_pred, cube_edges, 'red', 'Pred BBox')
    
    # Calcolo errori per il titolo
    err_t_mm = torch.norm(pred_T - gt_T).item() * 1000
    ax1.set_title(f"FIGURE 1: 3D Bounding Boxes (Obj {obj_id:02d})\nT-Error: {err_t_mm:.1f} mm")
    ax1.legend()
    ax1.axis('off')

    # --- VISUALIZZAZIONE 2: NUVOLA DI PUNTI DENSA (500 punti) ---
    
    # Campionamento casuale di 500 punti
    num_points = all_model_points.shape[0]
    if num_points > 500:
        indices = torch.randperm(num_points)[:500]
        sampled_points = all_model_points[indices]
    else:
        sampled_points = all_model_points

    # Proiezione
    dense_2d_gt = project_points_to_image(sampled_points, gt_R, gt_T, K_orig)
    dense_2d_pred = project_points_to_image(sampled_points, pred_R, pred_T, K_orig)

    # Plot
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(orig_img_rgb)
    # Usiamo scatter con alpha per vedere la densità
    ax2.scatter(dense_2d_gt[:, 0], dense_2d_gt[:, 1], c='lime', s=5, alpha=0.6, label='GT Points (500)')
    ax2.scatter(dense_2d_pred[:, 0], dense_2d_pred[:, 1], c='red', s=5, alpha=0.6, label='Pred Points (500)')
    
    ax2.set_title(f"FIGURE 2: Dense Point Cloud Projection (500 points)")
    ax2.legend()
    ax2.axis('off')

    print("Visualizzazioni generate. Chiudi le finestre per terminare.")
    plt.show()

if __name__ == "__main__":
    visualize_complete_evaluation()