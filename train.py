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

def train():
    # --- 1. CONFIGURAZIONE ---
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50 
    FREEZE_EPOCHS = 10 # Epoche in cui la ResNet-50 è bloccata
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "pose_resnet50_baseline.pth"

    # --- 2. DATI ---
    # Split 80/20 come concordato [cite: 96]
    train_samples, val_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET, test_size=0.2)
    
    train_set = LineModDataset(ROOT_DATASET, train_samples, gt_cache)
    val_set = LineModDataset(ROOT_DATASET, val_samples, gt_cache)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. MODELLO E FREEZE INIZIALE ---
    # Usiamo pesi pre-addestrati ImageNet [cite: 174]
    model = PosePredictor().to(DEVICE)
    
    print(f"❄️ Congelamento della backbone ResNet-50 per {FREEZE_EPOCHS} epoche...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Ottimizzatore: passiamo solo i parametri "vivi" (la testa di regressione)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Scheduler: riduce il LR se la val_loss non migliora per 5 epoche
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 4. LOOP ---
    best_val_loss = float('inf')
    backbone_unfrozen = False

    for epoch in range(EPOCHS):
        # Logica di Unfreeze
        if epoch >= FREEZE_EPOCHS and not backbone_unfrozen:
            print("🔥 Sblocco della backbone per il fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Aggiorniamo l'ottimizzatore per includere tutta la rete con un LR più basso
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            backbone_unfrozen = True

        model.train()
        train_loss, train_deg = 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs = batch["rgb"].to(DEVICE)
            targets = batch["quaternion"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss basata solo sulla rotazione [cite: 33, 176]
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

        avg_val_loss = val_loss / len(val_loader)
        avg_val_deg = val_deg / len(val_loader)
        
        # Aggiorna lo scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Val Loss {avg_val_loss:.4f} | Val Deg {avg_val_deg:.2f}°")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Miglior modello salvato!")

if __name__ == "__main__":
    train()