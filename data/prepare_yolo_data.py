import os
import shutil
import yaml
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

# Percorso dove si trova il dataset LineMod originale (Linemod_preprocessed)
SOURCE_ROOT = 'datasets/linemod/Linemod_preprocessed'

# Percorso dove salvare il dataset pronto per YOLO
OUTPUT_DIR = 'datasets/linemod/linemod_yolo_format'

# Percentuale di validazione (20%)
VAL_SIZE = 0.2

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

    # 2. Crea cartelle di output
    dirs = {
        'train_img': os.path.join(OUTPUT_DIR, 'images', 'train'),
        'val_img':   os.path.join(OUTPUT_DIR, 'images', 'val'),
        'train_lbl': os.path.join(OUTPUT_DIR, 'labels', 'train'),
        'val_lbl':   os.path.join(OUTPUT_DIR, 'labels', 'val')
    }
    
    # Se esiste già, avvisa ma prosegue
    if os.path.exists(OUTPUT_DIR):
        print("⚠️  ATTENZIONE: La cartella di output esiste già. I file verranno sovrascritti/aggiunti.")
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 3. Split Train/Val (80/20 casuale su tutto il dataset)
    indices = list(range(len(reader.samples)))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_SIZE, random_state=42, shuffle=True)
    
    # Mappa indice -> 'train' o 'val'
    split_map = {i: 'train' for i in train_idx}
    split_map.update({i: 'val' for i in val_idx})
    
    print(f"📊 Split configurato: {len(train_idx)} Train / {len(val_idx)} Validation")

    # 4. Loop Principale
    count_ok = 0
    count_err = 0

    for idx in tqdm(range(len(reader.samples)), desc="Generazione File"):
        obj_id, img_id, src_img_path = reader.samples[idx]
        split = split_map[idx]
        
        # Recupera BBox
        raw_bb = reader.get_bbox(obj_id, img_id)
        if raw_bb is None:
            count_err += 1
            continue

        # Apri immagine per dimensioni (necessario per normalizzare)
        try:
            with Image.open(src_img_path) as img:
                img_w, img_h = img.size
        except Exception:
            count_err += 1
            continue

        # Calcolo Coordinate YOLO Normalizzate
        x_min, y_min, w, h = raw_bb
        cx_norm = (x_min + w / 2) / img_w
        cy_norm = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # YOLO Class ID (parte da 0, LineMod parte da 1)
        class_id = obj_id - 1
        
        # Genera nome univoco
        unique_name = f"obj{obj_id:02d}_{img_id:04d}"
        
        # Percorsi destinazione
        dst_img_path = os.path.join(dirs[f'{split}_img'], unique_name + ".png")
        dst_txt_path = os.path.join(dirs[f'{split}_lbl'], unique_name + ".txt")
        
        # A. Copia Immagine
        shutil.copy(src_img_path, dst_img_path)
        
        # B. Scrivi Label .txt
        with open(dst_txt_path, 'w') as f:
            f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
        count_ok += 1

    # 5. Genera data.yaml
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
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
    print(f"✅ File processati con successo: {count_ok}")
    if count_err > 0:
        print(f"⚠️  File con errori o annotazioni mancanti: {count_err}")
    print("========================================")

if __name__ == "__main__":
    main()