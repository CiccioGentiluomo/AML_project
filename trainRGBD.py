import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
from datetime import datetime

# Import dai tuoi file
from models.RGBD_FusionPredictor import RGBD_FusionPredictor
from data.LineModDatasetRGBD import LineModDatasetRGBD
from utils.add_loss import ADDLoss
from data.split import prepare_data_and_splits

def log_and_print(message, log_file):
    """Stampa a video e scrive nel file di log."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def load_info_cache(dataset_root, object_ids):
    info_cache = {}
    for obj_id in object_ids:
        obj_folder = f"{obj_id:02d}"
        info_path = os.path.join(dataset_root, 'data', obj_folder, 'info.yml')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info_cache[obj_id] = yaml.safe_load(f)
    return info_cache

def train():
    # --- 1. CONFIGURAZIONE ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Percorsi file
    SAVE_PATH_BEST = "pose_rgbd_fusion_best.pth"
    CHECKPOINT_PATH = "pose_rgbd_checkpoint.pth"
    LOG_FILE = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    SYMMETRIC_OBJECTS = {10, 11} 

    # --- 2. DATI ---
    train_samples, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    
    train_set = LineModDatasetRGBD(ROOT_DATASET, train_samples, gt_cache, info_cache)
    val_set = LineModDatasetRGBD(ROOT_DATASET, val_samples, gt_cache, info_cache)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. MODELLO, LOSS E OTTIMIZZATORE ---
    model = RGBD_FusionPredictor().to(DEVICE)
    criterion = ADDLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 4. MECCANISMO DI RESUME ---
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        log_and_print(f"🔄 Caricamento checkpoint da {CHECKPOINT_PATH}...", LOG_FILE)
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        log_and_print(f"▶️ Ripresa dall'epoca {start_epoch}", LOG_FILE)

    # --- 5. LOOP DI TRAINING ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_total = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            gt_R = batch["R_matrix"].to(DEVICE)
            gt_T = batch["translation_3d"].to(DEVICE)
            model_points = batch["model_points"].to(DEVICE)

            optimizer.zero_grad()
            pred_T, pred_R = model(rgb, depth)

            # Loss del batch (gestione oggetti simmetrici/asimmetrici)
            batch_loss = 0
            for i in range(rgb.size(0)):
                curr_id = batch["obj_id"][i].item()
                is_sym = curr_id in SYMMETRIC_OBJECTS
                item_loss = criterion(
                    pred_R[i:i+1], pred_T[i:i+1], 
                    gt_R[i:i+1], gt_T[i:i+1], 
                    model_points[i:i+1], 
                    is_symmetric=is_sym
                )
                batch_loss += item_loss

            batch_loss /= rgb.size(0)
            batch_loss.backward()
            optimizer.step()

            train_loss_total += batch_loss.item()
            pbar.set_postfix({'loss': batch_loss.item()})

        avg_train_loss = train_loss_total / len(train_loader)

        # --- VALIDAZIONE ---
        model.eval()
        val_loss_total = 0.0
        obj_errors = {obj_id: 0.0 for obj_id in object_ids}
        obj_counts = {obj_id: 0 for obj_id in object_ids}

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(DEVICE)
                depth = batch["depth"].to(DEVICE)
                gt_R = batch["R_matrix"].to(DEVICE)
                gt_T = batch["translation_3d"].to(DEVICE)
                model_points = batch["model_points"].to(DEVICE)
                ids = batch["obj_id"]

                pred_T, pred_R = model(rgb, depth)
                
                for i in range(rgb.size(0)):
                    curr_id = ids[i].item()
                    is_sym = curr_id in SYMMETRIC_OBJECTS
                    loss = criterion(
                        pred_R[i:i+1], pred_T[i:i+1], gt_R[i:i+1], gt_T[i:i+1], 
                        model_points[i:i+1], is_symmetric=is_sym
                    )
                    val_loss_total += loss.item()
                    obj_errors[curr_id] += loss.item()
                    obj_counts[curr_id] += 1

        avg_val_loss = val_loss_total / len(val_set)
        scheduler.step(avg_val_loss)

        # Reportistica finale epoca
        log_and_print(f"\n--- Epoch {epoch+1} Summary ---", LOG_FILE)
        log_and_print(f"Avg Train ADD Loss: {avg_train_loss:.6f} m", LOG_FILE)
        log_and_print(f"Avg Val ADD Loss:   {avg_val_loss:.6f} m", LOG_FILE)
        
        # Dettaglio per oggetto
        for obj_id in object_ids:
            if obj_counts[obj_id] > 0:
                err_mm = (obj_errors[obj_id] / obj_counts[obj_id]) * 1000
                log_and_print(f" Object {obj_id:02d}: {err_mm:.2f} mm", LOG_FILE)

        # Salvataggio Checkpoint (per resume)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        # Salvataggio Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            log_and_print(f"⭐ NEW BEST! Model saved to {SAVE_PATH_BEST}", LOG_FILE)
        
        log_and_print("-" * 40, LOG_FILE)

if __name__ == "__main__":
    train()