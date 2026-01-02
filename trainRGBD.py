import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
from datetime import datetime
import wandb

# Import dai tuoi file
from models.RGBD_FusionPredictor import RGBD_FusionPredictor
from data.LineModDatasetRGBD import LineModDatasetRGBD
from utils.add_loss import ADDLoss
from data.split import prepare_data_and_splits

def log_and_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
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
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SAVE_PATH_BEST = "pose_rgbd_fusion_best.pth"
    CHECKPOINT_PATH = "pose_rgbd_checkpoint.pth"
    LOG_FILE = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    SYMMETRIC_OBJECTS = {10, 11} 

    # Inizializza wandb
    wandb.init(
        project="linemod-pose-estimation",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "RGBD_FusionPredictor",
            "dataset": "LineMod_RGBD",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": 1e-4
        }
    )

    # --- 2. DATI ---
    train_samples, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    
    train_set = LineModDatasetRGBD(ROOT_DATASET, train_samples, gt_cache, info_cache)
    val_set = LineModDatasetRGBD(ROOT_DATASET, val_samples, gt_cache, info_cache)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. MODELLO E OTTIMIZZATORE CON REGOLARIZZAZIONE ---
    model = RGBD_FusionPredictor().to(DEVICE)
    criterion = ADDLoss().to(DEVICE)
    
    # AGGIUNTO: weight_decay per combattere l'overfitting
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4 
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        log_and_print(f"🔄 Caricamento checkpoint da {CHECKPOINT_PATH}...", LOG_FILE)
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        log_and_print(f"▶️ Ripresa dall'epoca {start_epoch} con reset best_loss.", LOG_FILE)

    # --- 5. LOOP DI TRAINING ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_trans_mse = 0.0
        train_rot_mse = 0.0
        
        # Recupero del Learning Rate corrente per il log
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [LR: {current_lr:.2e}]")
        for batch in pbar:
            rgb, depth, meta = batch["rgb"].to(DEVICE), batch["depth"].to(DEVICE), batch["meta_info"].to(DEVICE)
            gt_R, gt_T = batch["R_matrix"].to(DEVICE), batch["translation_3d"].to(DEVICE)
            model_points = batch["model_points"].to(DEVICE)

            optimizer.zero_grad()
            pred_T, pred_R_raw = model(rgb, depth, meta)
            pred_R = pred_R_raw.view(-1, 3, 3)

            # Training con ADD pura (is_symmetric=False)
            batch_loss = criterion(pred_R, pred_T, gt_R, gt_T, model_points, is_symmetric=False)

            with torch.no_grad():
                train_trans_mse += F.mse_loss(pred_T, gt_T).item()
                train_rot_mse += F.mse_loss(pred_R, gt_R).item()

            batch_loss.backward()
            optimizer.step()

            train_loss_total += batch_loss.item()
            pbar.set_postfix({'ADD': batch_loss.item()})

        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_t_mse = train_trans_mse / len(train_loader)
        avg_train_r_mse = train_rot_mse / len(train_loader)

        # Log metriche di training
        wandb.log({
            "train/epoch": epoch + 1,
            "train/loss_add": avg_train_loss,
            "train/t_mse": avg_train_t_mse,
            "train/r_mse": avg_train_r_mse,
            "train/lr": current_lr
        })

        # --- VALIDAZIONE ---
        model.eval()
        val_loss_total = 0.0
        val_trans_mse = 0.0
        val_rot_mse = 0.0
        obj_errors = {obj_id: 0.0 for obj_id in object_ids}
        obj_counts = {obj_id: 0 for obj_id in object_ids}

        with torch.no_grad():
            for batch in val_loader:
                rgb, depth, meta = batch["rgb"].to(DEVICE), batch["depth"].to(DEVICE), batch["meta_info"].to(DEVICE)
                gt_R, gt_T = batch["R_matrix"].to(DEVICE), batch["translation_3d"].to(DEVICE)
                model_points, ids = batch["model_points"].to(DEVICE), batch["obj_id"]

                pred_T, pred_R_raw = model(rgb, depth, meta)
                pred_R = pred_R_raw.view(-1, 3, 3)
                
                val_trans_mse += F.mse_loss(pred_T, gt_T).item()
                val_rot_mse += F.mse_loss(pred_R, gt_R).item()

                for i in range(rgb.size(0)):
                    curr_id = ids[i].item()
                    # Validazione rigorosa con ADD standard per monitoraggio reale
                    loss = criterion(
                        pred_R[i:i+1], pred_T[i:i+1], gt_R[i:i+1], gt_T[i:i+1], 
                        model_points[i:i+1], is_symmetric=False 
                    )
                    val_loss_total += loss.item()
                    obj_errors[curr_id] += loss.item()
                    obj_counts[curr_id] += 1

        avg_val_loss = val_loss_total / len(val_set)
        avg_val_t_mse = val_trans_mse / len(val_loader)
        avg_val_r_mse = val_rot_mse / len(val_loader)
        
        scheduler.step(avg_val_loss)

        # --- REPORTISTICA FINALE CON LR ---
        log_and_print(f"\n--- Epoch {epoch+1} Summary ---", LOG_FILE)
        log_and_print(f"Learning Rate: {current_lr:.2e}", LOG_FILE) # Riga richiesta
        log_and_print(f"Avg Train ADD: {avg_train_loss:.6f} m | T-MSE: {avg_train_t_mse:.6f} | R-MSE: {avg_train_r_mse:.6f}", LOG_FILE)
        log_and_print(f"Avg Val ADD:   {avg_val_loss:.6f} m | T-MSE: {avg_val_t_mse:.6f} | R-MSE: {avg_val_r_mse:.6f}", LOG_FILE)
        

        # Log metriche di validazione
        val_metrics = {
            "val/loss_add": avg_val_loss,
            "val/t_mse": avg_val_t_mse,
            "val/r_mse": avg_val_r_mse,
        }


        for obj_id in object_ids:
            if obj_counts[obj_id] > 0:
                err_mm = (obj_errors[obj_id] / obj_counts[obj_id]) * 1000
                val_metrics[f"val_obj/error_mm_{obj_id:02d}"] = err_mm
                log_and_print(f" Object {obj_id:02d}: {err_mm:.2f} mm", LOG_FILE)

        wandb.log(val_metrics)

        # Salvataggio checkpoint e best model
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wandb.save(SAVE_PATH_BEST) # Carica il file .pth su wandb
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            log_and_print(f"⭐ NEW BEST! Model saved to {SAVE_PATH_BEST}", LOG_FILE)
        
        log_and_print("-" * 40, LOG_FILE)

if __name__ == "__main__":
    train()
    wandb.finish()