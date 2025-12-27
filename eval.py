import os
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Assicurati che questo percorso sia corretto rispetto a dove lanci lo script
DATASET_ROOT = 'datasets/linemod/Linemod_preprocessed'

# L'oggetto che vuoi ispezionare (2 = Benchvise, la morsa)
TARGET_OBJ_ID = 2 

def visualize_sample():
    # 1. Costruisci i percorsi
    obj_dir_name = f"{TARGET_OBJ_ID:02d}"
    base_path = os.path.join(DATASET_ROOT, 'data', obj_dir_name)
    
    gt_path = os.path.join(base_path, 'gt.yml')
    rgb_dir = os.path.join(base_path, 'rgb')
    
    # Verifica preliminare
    if not os.path.exists(gt_path):
        print(f"❌ Errore: Non trovo il file {gt_path}")
        print(f"   Verifica che il percorso '{DATASET_ROOT}' sia corretto.")
        return

    # 2. Carica le annotazioni
    print(f"📂 Caricamento GT per oggetto {TARGET_OBJ_ID} da {gt_path}...")
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)

    # 3. CERCA UN'IMMAGINE VALIDA
    # LineMOD è "tricky": il file gt.yml contiene TUTTI gli oggetti visibili nella foto.
    # Dobbiamo trovare una foto che contenga l'annotazione specifica per il nostro TARGET_OBJ_ID.
    
    valid_samples = []
    
    # gt_data è un dizionario: { img_id (int): [lista_annotazioni] }
    for img_id, anns in gt_data.items():
        for ann in anns:
            if ann['obj_id'] == TARGET_OBJ_ID:
                valid_samples.append(img_id)
                break # Trovato, passiamo alla prossima immagine
    
    if not valid_samples:
        print(f"⚠️ Nessuna annotazione trovata per l'oggetto ID {TARGET_OBJ_ID} in questo file YAML!")
        return

    print(f"✅ Trovate {len(valid_samples)} immagini contenenti l'oggetto {TARGET_OBJ_ID}.")

    # 4. Seleziona un'immagine a caso
    random_img_id = random.choice(valid_samples)
    
    # 5. Recupera l'annotazione ESATTA
    anns = gt_data[random_img_id]
    target_ann = None
    
    # Filtriamo di nuovo per essere sicuri al 100% di prendere la box giusta
    for ann in anns:
        if ann['obj_id'] == TARGET_OBJ_ID:
            target_ann = ann
            break
            
    raw_bbox = target_ann['obj_bb'] # [x, y, w, h]
    
    print(f"🔹 Visualizzo Immagine ID: {random_img_id}")
    print(f"🔹 Oggetto ID Annotazione: {target_ann['obj_id']} (Deve essere {TARGET_OBJ_ID})")
    print(f"🔹 BBox [x, y, w, h]: {raw_bbox}")

    # 6. Carica e Visualizza
    img_filename = f"{random_img_id:04d}.png"
    img_path = os.path.join(rgb_dir, img_filename)
    
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"❌ Impossibile aprire l'immagine {img_path}: {e}")
        return

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    
    # Disegna il rettangolo
    # patches.Rectangle( (x,y), width, height )
    rect = patches.Rectangle(
        (raw_bbox[0], raw_bbox[1]), 
        raw_bbox[2], 
        raw_bbox[3], 
        linewidth=3, 
        edgecolor='#00FF00', # Verde Lime brillante
        facecolor='none'
    )
    
    ax.add_patch(rect)
    ax.set_title(f"Verifica Ground Truth - Oggetto {TARGET_OBJ_ID} - Img {random_img_id}")
    plt.axis('off')
    
    print("🖼️  Finestra del grafico aperta...")
    plt.show()

if __name__ == "__main__":
    visualize_sample()