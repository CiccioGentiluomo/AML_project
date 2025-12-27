import os
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
try:
    from .linemod_dataset import LineModDataset
except ImportError:
    from linemod_dataset import LineModDataset  # fallback when running as script

def prepare_data_and_splits(dataset_root, test_size=0.2, random_seed=42):
    all_samples = [] 
    gt_cache = {}    
    
    data_path = os.path.join(dataset_root, 'data')
    obj_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"🔍 Scansione di {len(obj_dirs)} oggetti...")

    for obj_dir in obj_dirs:
        try:
            obj_id = int(obj_dir)
        except: continue
            
        gt_file = os.path.join(data_path, obj_dir, 'gt.yml')
        if not os.path.exists(gt_file): continue
            
        with open(gt_file, 'r') as f:
            gt_content = yaml.safe_load(f)
            gt_cache[obj_id] = gt_content
            
        for img_id in gt_content.keys():
            all_samples.append((obj_id, int(img_id)))

    train_samples, val_samples = train_test_split(
        all_samples, test_size=test_size, random_state=random_seed, shuffle=True
    )
    
    return train_samples, val_samples, gt_cache

def show_comparison(dataset, index=0):
    # Recupero ID dal dataset
    obj_id, img_id = dataset.samples[index]
    
    # Caricamento immagine originale per il confronto
    obj_folder = f"{obj_id:02d}"
    img_path = os.path.join(dataset.dataset_root, 'data', obj_folder, 'rgb', f"{img_id:04d}.png")
    img_original = Image.open(img_path).convert("RGB")
    
    # Recupero BBox corretta filtrando per ID (come nel dataset)
    ann_list = dataset.gt_cache[obj_id][img_id]
    target_ann = next((a for a in ann_list if a['obj_id'] == obj_id), ann_list[0])
    bbox = target_ann['obj_bb'] 
    
    # Recupero dato processato
    sample = dataset[index]
    img_tensor = sample['rgb']
    
    # De-normalizzazione
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_processed = img_tensor.permute(1, 2, 0).numpy()
    img_processed = (img_processed * std) + mean
    img_processed = np.clip(img_processed, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img_original)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(f"ORIGINALE (Oggetto {obj_id})\nBBox corretta filtrata")
    ax1.axis('off')
    
    ax2.imshow(img_processed)
    ax2.set_title(f"INPUT RESNET (224x224)\nSquare Crop dell'oggetto {obj_id}")
    ax2.axis('off')
    
    plt.show()

if __name__ == "__main__":
    ROOT = "datasets/linemod/Linemod_preprocessed"
    train_samples, val_samples, gt_cache = prepare_data_and_splits(ROOT)
    
    train_set = LineModDataset(ROOT, train_samples, gt_cache)

    #verifico dimensioni campione dataset in pixels
    sample_img = train_set[0]['rgb']
    print(f"Dimensione immagine campione nel dataset: {sample_img.shape[1]}x{sample_img.shape[2]} pixels")
    
    
    # Proviamo a visualizzare specificamente la morsa (ID 02)
    print("📸 Cerco un esempio della morsa (ID 2)...")
    try:
        idx_02 = next(i for i, s in enumerate(train_set.samples) if s[0] == 2)
        show_comparison(train_set, index=idx_02)
    except StopIteration:
        show_comparison(train_set, index=0)