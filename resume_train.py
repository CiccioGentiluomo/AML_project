import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import dai tuoi file
from models.PosePredictor import PosePredictor
from data.linemod_dataset import LineModDataset
from data.split import prepare_data_and_splits
from utils.resNetUtils import rotation_loss

def quaternion_angle_error(pred, target):
    """Return angular error (deg) between predicted and GT quaternions."""
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    dot = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0, 1.0).abs()
    angles = 2.0 * torch.acos(dot) * (180.0 / math.pi)
    return angles

def resume_training():
    # --- 1. CONFIGURAZIONE ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    BATCH_SIZE = 32
    # Partiamo con l'ultimo LR registrato dallo scheduler (2.5e-6)
    LEARNING_RATE = 2.5e-6 
    EXTRA_EPOCHS = 10 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "pose_resnet50_baseline.pth"

    print(f"♻️ Ripresa training su: {DEVICE} con LR: {LEARNING_RATE}")

    # --- 2. DATI ---
    # Manteniamo lo split 80/20 concordato [cite: 96]
    train_samples, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    
    train_set = LineModDataset(ROOT_DATASET, train_samples, gt_cache)
    val_set = LineModDataset(ROOT_DATASET, val_samples, gt_cache)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. MODELLO E CARICAMENTO CHECKPOINT ---
    # Il modello usa ResNet-50 come backbone [cite: 174]
    model = PosePredictor().to(DEVICE)
    
    if os.path.exists(SAVE_PATH):
        print(f"📂 Caricamento pesi da: {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH))
    else:
        print("⚠️ Errore: Checkpoint non trovato!")
        return

    # Sblocchiamo tutto per continuare il fine-tuning [cite: 174]
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- 4. LOOP DI TRAINING EXTRA ---
    best_val_loss = float('inf') # Verrà aggiornato al primo giro di val

    for epoch in range(EXTRA_EPOCHS):
        model.train()
        train_loss, train_deg = 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Extra Epoch {epoch+1}/{EXTRA_EPOCHS} [Train]"):
            inputs = batch["rgb"].to(DEVICE)
            targets = batch["quaternion"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss basata solo sulla rotazione per ResNet 
            loss = rotation_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_deg += quaternion_angle_error(outputs, targets).mean().item()

        # --- VALIDAZIONE ---
        model.eval()
        val_loss, val_deg = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["rgb"].to(DEVICE)
                targets = batch["quaternion"].to(DEVICE)
                outputs = model(inputs)
                
                loss = rotation_loss(outputs, targets)
                val_loss += loss.item()
                val_deg += quaternion_angle_error(outputs, targets).mean().item()

        # --- CALCOLO MEDIE FINALI EPOCO ---
        avg_train_loss = train_loss / len(train_loader)
        avg_train_deg = train_deg / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_deg = val_deg / len(val_loader)
        
        # Aggiorna lo scheduler basandosi sulla val_loss
        scheduler.step(avg_val_loss)

        # STAMPA COMPLETA PER MONITORARE OVERFITTING
        print(
            f"Extra Epoch [{epoch+1}/{EXTRA_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Deg: {avg_train_deg:.2f}° | Val Deg: {avg_val_deg:.2f}° | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Salvataggio se migliora la val_loss (il criterio del baseline [cite: 176])
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Miglioramento trovato! Modello salvato.")

    print("🏁 Sessione extra completata!")

if __name__ == "__main__":
    resume_training()