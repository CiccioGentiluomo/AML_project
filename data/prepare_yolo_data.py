import os
import shutil
import sys
from pathlib import Path
import yaml
import glob
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.split import prepare_data_and_splits

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

# Percorso dove si trova il dataset LineMod originale (Linemod_preprocessed)
SOURCE_ROOT = 'datasets/linemod/Linemod_preprocessed'

# Percorso dove salvare il dataset pronto per YOLO
OUTPUT_DIR = 'datasets/linemod/linemod_yolo_format'

# Mappatura Nomi Classi (ID LineMod -> Nome)
# LineMod usa ID 1-15. YOLO userà ID 0-14.
CLASS_NAMES = {
    0: "Ape",
    1: "Benchvise",
    2: "Bowl",
    3: "Cam",
    4: "Can",
    5: "Cat",
    6: "Cup",
    7: "Driller",
    8: "Duck",
    9: "Eggbox",
    10: "Glue",
    11: "Holepuncher",
    12: "Iron",
    13: "Lamp",
    14: "Phone"
}

# ==========================================
# 2. LETTORE DATI GREZZI (Semplificato)
# ==========================================
class LineModRawScanner:
    """
    Legge il dataset originale senza usare PyTorch o trasformazioni.
    Serve solo per trovare i file e le bounding box.
    """
    def __init__(self, root):
        self.root = root
        self.samples = [] # Lista di (obj_id, img_id, path_immagine)
        self.gt_data = {} # Cache annotazioni

        data_path = os.path.join(root, 'data')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cartella dati non trovata: {data_path}")

        # Trova cartelle numeriche (01, 02, ...)
        object_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
        print(f"📂 Scansione dataset in corso... Trovati {len(object_dirs)} oggetti.")

        for obj_dir in object_dirs:
            try:
                obj_id = int(obj_dir)
            except ValueError:
                continue

            # A. Carica GT
            gt_path = os.path.join(data_path, obj_dir, 'gt.yml')
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    self.gt_data[obj_id] = yaml.safe_load(f)
            else:
                print(f"⚠️ gt.yml mancante per {obj_dir}, salto.")
                continue

            # B. Trova tutte le immagini RGB (Usa glob per prendere TUTTO)
            rgb_path = os.path.join(data_path, obj_dir, 'rgb')
            image_files = glob.glob(os.path.join(rgb_path, "*.png"))
            
            for img_file in image_files:
                basename = os.path.basename(img_file)
                img_id = int(os.path.splitext(basename)[0])
                
                # Salviamo la tupla
                self.samples.append((obj_id, img_id, img_file))

    def get_bbox(self, obj_id, img_id):
        """Estrae la bbox grezza dal file yml caricato"""
        if obj_id in self.gt_data and img_id in self.gt_data[obj_id]:
            anns = self.gt_data[obj_id][img_id]
            # Cerca l'annotazione che corrisponde all'oggetto corrente
            for ann in anns:
                if ann['obj_id'] == obj_id:
                    return ann['obj_bb'] # [x, y, w, h]
        return None

# ==========================================
# 3. FUNZIONE DI CONVERSIONE
# ==========================================
def main():
    print(f"🚀 Inizio creazione dataset YOLO...")
    
    # 1. Scansiona i file
    reader = LineModRawScanner(SOURCE_ROOT)
    print(f"✅ Totale immagini trovate: {len(reader.samples)}")

    # Dizionario rapido per recuperare il percorso partendo dalla tupla (obj_id, img_id)
    sample_to_path = {(obj_id, img_id): path for obj_id, img_id, path in reader.samples}

    # 1.b Otteniamo gli split globali (60/20/20) da riutilizzare ovunque
    train_samples, val_samples, test_samples, _ = prepare_data_and_splits(SOURCE_ROOT)
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }

    # 2. Crea cartelle di output
    dirs = {
        'train_img': os.path.join(OUTPUT_DIR, 'images', 'train'),
        'val_img':   os.path.join(OUTPUT_DIR, 'images', 'val'),
        'test_img':  os.path.join(OUTPUT_DIR, 'images', 'test'),
        'train_lbl': os.path.join(OUTPUT_DIR, 'labels', 'train'),
        'val_lbl':   os.path.join(OUTPUT_DIR, 'labels', 'val'),
        'test_lbl':  os.path.join(OUTPUT_DIR, 'labels', 'test'),
    }
    
    # Se esiste già, avvisa ma prosegue
    if os.path.exists(OUTPUT_DIR):
        print("⚠️  ATTENZIONE: La cartella di output esiste già. I file verranno sovrascritti/aggiunti.")
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("📊 Split configurato:")
    for split_name, sample_list in splits.items():
        print(f"   - {split_name.capitalize()}: {len(sample_list)} campioni")

    # 3. Loop Principale
    count_ok = {split: 0 for split in splits}
    count_err = {split: 0 for split in splits}

    for split_name, sample_list in splits.items():
        progress = tqdm(sample_list, desc=f"Preparazione {split_name}")
        for obj_id, img_id in progress:
            src_img_path = sample_to_path.get((obj_id, img_id))
            if src_img_path is None:
                count_err[split_name] += 1
                continue

            raw_bb = reader.get_bbox(obj_id, img_id)
            if raw_bb is None:
                count_err[split_name] += 1
                continue

            try:
                with Image.open(src_img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                count_err[split_name] += 1
                continue

            x_min, y_min, w, h = raw_bb
            cx_norm = (x_min + w / 2) / img_w
            cy_norm = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            class_id = obj_id - 1
            unique_name = f"obj{obj_id:02d}_{img_id:04d}"

            dst_img_path = os.path.join(dirs[f"{split_name}_img"], unique_name + ".png")
            dst_txt_path = os.path.join(dirs[f"{split_name}_lbl"], unique_name + ".txt")

            shutil.copy(src_img_path, dst_img_path)
            with open(dst_txt_path, 'w') as f:
                f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            count_ok[split_name] += 1

    # 5. Genera data.yaml
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print("\n========================================")
    print(f"✅ OPERAZIONE COMPLETATA")
    print(f"📁 Dataset YOLO salvato in: {OUTPUT_DIR}")
    print(f"📄 Configurazione: {yaml_path}")
    for split_name in ['train', 'val', 'test']:
        print(
            f"   - {split_name.capitalize()}: {count_ok[split_name]} file ok"
            + (f" | {count_err[split_name]} errori" if count_err[split_name] else "")
        )
    print("========================================")

if __name__ == "__main__":
    main()