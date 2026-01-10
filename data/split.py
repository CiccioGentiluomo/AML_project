import os
import sys
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.linemod_dataset import LineModDataset


def prepare_data_and_splits(
    dataset_root,
    random_seed=42,
    val_ratio=0.2,
    test_ratio=0.2,
    **legacy_kwargs,
):
    """Return train/val/test splits plus cached ground-truth annotations."""
    if "test_size" in legacy_kwargs:
        val_ratio = legacy_kwargs["test_size"]

    holdout_ratio = val_ratio + test_ratio
    if not 0 < holdout_ratio < 1:
        raise ValueError("val_ratio + test_ratio must be between 0 and 1")

    all_samples = []
    gt_cache = {}

    data_path = os.path.join(dataset_root, "data")
    obj_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    print(f"🔍 Scansione di {len(obj_dirs)} oggetti...")

    for obj_dir in obj_dirs:
        try:
            obj_id = int(obj_dir)
        except ValueError:
            continue

        gt_file = os.path.join(data_path, obj_dir, "gt.yml")
        if not os.path.exists(gt_file):
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_content = yaml.safe_load(f)
            gt_cache[obj_id] = gt_content

        for img_id in gt_content.keys():
            all_samples.append((obj_id, int(img_id)))

    # Split train vs (val+test)
    train_samples, holdout_samples = train_test_split(
        all_samples,
        test_size=holdout_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    relative_test_ratio = test_ratio / holdout_ratio
    val_samples, test_samples = train_test_split(
        holdout_samples,
        test_size=relative_test_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    train_pct = (len(train_samples) / len(all_samples)) * 100
    val_pct = (len(val_samples) / len(all_samples)) * 100
    test_pct = (len(test_samples) / len(all_samples)) * 100

    print("✅ Split completato:")
    print(f"   - Train: {len(train_samples)} campioni ({train_pct:.1f}%)")
    print(f"   - Val:   {len(val_samples)} campioni ({val_pct:.1f}%)")
    print(f"   - Test:  {len(test_samples)} campioni ({test_pct:.1f}%)")

    return train_samples, val_samples, test_samples, gt_cache

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
    train_samples, val_samples, test_samples, gt_cache = prepare_data_and_splits(ROOT)

    train_set = LineModDataset(ROOT, train_samples, gt_cache)
    val_set = LineModDataset(ROOT, val_samples, gt_cache)
    test_set = LineModDataset(ROOT, test_samples, gt_cache)

    print("Dataset pronti per l'addestramento!")