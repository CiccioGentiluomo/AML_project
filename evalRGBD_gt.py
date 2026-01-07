import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import dai tuoi file
from models.RGBD_FusionPredictor import RGBD_FusionPredictor
from data.LineModDatasetRGBD import LineModDatasetRGBD
from data.split import prepare_data_and_splits

def compute_add_metric(pred_R, pred_T, gt_R, gt_T, model_points):
    """
    Calcola la distanza media ADD tra punti trasformati.
    """
    # Trasforma i punti con la posa predetta: (Points @ R_T) + T
    pred_points = torch.mm(model_points, pred_R.t()) + pred_T
    # Trasforma i punti con la posa reale (GT)
    gt_points = torch.mm(model_points, gt_R.t()) + gt_T
    
    # Distanza euclidea media tra i corrispondenti punti
    dis = torch.norm(pred_points - gt_points, dim=1)
    return torch.mean(dis).item()

def evaluate():
    # --- 1. CONFIGURAZIONE ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    MODEL_PATH = "pose_rgbd_fusion_best.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Diametri LineMOD per soglia 10% ADD
    DIAMETERS = {
        1: 102.09, 2: 282.60, 4: 171.64, 5: 201.50, 6: 154.54,
        8: 261.47, 9: 108.99, 10: 164.66, 11: 175.88, 12: 162.24,
        13: 258.41, 14: 282.25, 15: 212.35
    }

    # --- 2. CARICAMENTO DATI ---
        _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    # Nota: Carichiamo la cache info solo se necessaria per il dataset
    from trainRGBD import load_info_cache
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    
        test_set = LineModDatasetRGBD(ROOT_DATASET, test_samples, gt_cache, info_cache)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # Batch 1 per analisi singola

    # --- 3. CARICAMENTO MODELLO ---
    model = RGBD_FusionPredictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 4. LOOP DI VALUTAZIONE ---
    results = {obj_id: {"errors": [], "correct": 0, "total": 0} for obj_id in object_ids}

    print(f"🚀 Inizio Valutazione su {len(test_set)} campioni...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            meta = batch["meta_info"].to(DEVICE)
            gt_R = batch["R_matrix"][0].to(DEVICE)
            gt_T = batch["translation_3d"][0].to(DEVICE)
            model_points = batch["model_points"][0].to(DEVICE)
            obj_id = batch["obj_id"][0].item()

            # Predizione
            pred_T, pred_R_raw = model(rgb, depth, meta)
            pred_R = pred_R_raw.view(3, 3) # Reshape per geometria

            # Calcolo ADD
            add_error_m = compute_add_metric(pred_R, pred_T[0], gt_R, gt_T, model_points)
            add_error_mm = add_error_m * 1000
            
            # Verifica soglia 10% diametro
            threshold = DIAMETERS[obj_id] * 0.1
            is_correct = add_error_mm <= threshold

            # Update statistiche
            results[obj_id]["errors"].append(add_error_mm)
            results[obj_id]["total"] += 1
            if is_correct:
                results[obj_id]["correct"] += 1

    # --- 5. REPORT FINALE ---
    print("\n" + "="*50)
    print(f"{'OBJ ID':<10} | {'AVG ADD (mm)':<15} | {'ACCURACY (0.1d)':<15}")
    print("-" * 50)
    
    total_correct = 0
    total_samples = 0
    
    for obj_id in sorted(results.keys()):
        data = results[obj_id]
        avg_err = np.mean(data["errors"])
        acc = (data["correct"] / data["total"]) * 100
        
        total_correct += data["correct"]
        total_samples += data["total"]
        
        print(f"{obj_id:02d}         | {avg_err:<15.2f} | {acc:<15.2f}%")

    overall_acc = (total_correct / total_samples) * 100
    print("-" * 50)
    print(f"OVERALL ACCURACY: {overall_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate()