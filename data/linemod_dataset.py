import torch
from torch.utils.data import Dataset
import os
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Definiamo le trasformazioni standard per il tuo progetto
# Per la Fase 3 (ResNet), ResNet-50 richiede input 224x224 e la normalizzazione ImageNet.
# Assumiamo di applicare queste trasformazioni solo all'immagine RGB, non alla Depth.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class LineModDataset(Dataset):
    
    def __init__(self, dataset_root, split='train', img_size=(224, 224)):
        
        self.dataset_root = dataset_root
        self.split = split
        self.gt_data = {}  # Cache per le annotazioni di TUTTI gli oggetti
        self.samples = []  # Lista finale di tuple (obj_id, img_id)
        
        # 1. Definizione delle Trasformazioni
        self.transform_rgb = transforms.Compose([
            transforms.Resize(img_size),          # Resize a 224x224 per ResNet
            transforms.ToTensor(),                # Conversione in Tensore [0, 1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalizzazione standard
        ])
        
        # La Depth va solo in Tensore, senza normalizzazione ImageNet
        self.transform_depth = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
        # 2. Scansione Dataset e Logica di Split (Come nel Lab 02)
        data_path = os.path.join(self.dataset_root, 'data')
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Cartella dati non trovata: {data_path}")

        object_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
        print(f"🚀 Caricamento {split.upper()} set per {len(object_dirs)} oggetti...")

        for obj_dir in object_dirs:
            obj_id_int = int(obj_dir)
            obj_path = os.path.join(data_path, obj_dir)
            
            # A. Carica il gt.yml di QUESTO oggetto nella cache
            gt_path = os.path.join(obj_path, 'gt.yml')
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    self.gt_data[obj_id_int] = yaml.safe_load(f)
            else:
                continue

            # B. Leggi il file train.txt o test.txt per gli indici (Logica di Split)
            split_file = os.path.join(obj_path, f'{split}.txt') # Legge 'train.txt' o 'test.txt'
            
            if not os.path.exists(split_file):
                print(f"ATTENZIONE: File di split {split}.txt mancante per oggetto {obj_dir}. Skippo.")
                continue

            with open(split_file, 'r') as f:
                # Ogni riga è un ID immagine (es. '0001\n'). Strip rimuove lo spazio/newline
                image_ids = [line.strip() for line in f.readlines()]
                
            # C. Popola self.samples solo con gli ID trovati nel file
            for img_id_str in image_ids:
                img_id_int = int(img_id_str)
                # Controlliamo che l'annotazione esista (prevenzione errori)
                if img_id_int in self.gt_data[obj_id_int]:
                    # Aggiungiamo alla lista: (obj_id_int, img_id_int)
                    self.samples.append((obj_id_int, img_id_int))
                else:
                     print(f"Annotazione GT mancante per {obj_dir}/{img_id_str}. Skippo.")


        print(f"✅ LineModDataset {split.upper()} set pronto con {len(self.samples)} campioni totali.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        
        path_obj_folder = f"{folder_id:02d}"
        fname_img = f"{sample_id:04d}.png"
        
        # 1. Percorsi e Caricamento Immagini
        img_path = os.path.join(self.dataset_root, 'data', path_obj_folder, 'rgb', fname_img)
        depth_path = os.path.join(self.dataset_root, 'data', path_obj_folder, 'depth', fname_img)
        
        img = Image.open(img_path).convert("RGB")
        # Per la Depth, usiamo il modo standard per immagini in scala di grigi
        depth_img = Image.open(depth_path)

        # Convertiamo in Array Numpy per preservare i valori in millimetri
        depth_np = np.array(depth_img, dtype=np.float32)
        
        # Convertiamo in Tensore PyTorch
        # Aggiungiamo unsqueeze(0) per avere la shape [1, H, W]
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        
        # 2. Trasformazione
        img_tensor = self.transform_rgb(img)
        depth_tensor = self.transform_depth(depth_tensor)
        
        # 3. Recupera Annotazioni R, T, BBox
        # Il GT è una lista di annotazioni. In LineMOD classico, è quasi sempre 1 solo oggetto.
        ann_list = self.gt_data[folder_id][sample_id]
        target_ann = ann_list[0] # Prendiamo la prima (e unica) annotazione
        
        # Estrazione Pose 6D
        R_list = target_ann['cam_R_m2c'] # Rotazione (matrice 3x3 appiattita a 9)
        T_list = target_ann['cam_t_m2c'] # Traslazione (vettore 3)
        obj_bb = target_ann['obj_bb']    # BBox in pixel [xmin, ymin, w, h]

        # 4. Conversione in Tensori
        R = torch.tensor(R_list, dtype=torch.float32).view(3, 3)
        T = torch.tensor(T_list, dtype=torch.float32)
        obj_bb = torch.tensor(obj_bb, dtype=torch.float32)

        # 5. RETURN FINALE (Il set di dati che alimenta la rete per la Fase 3)
        return {
            "rgb": img_tensor,             # Tensore RGB (3, 224, 224)
            "depth": depth_tensor,         # Tensore Depth (1, 224, 224)
            "obj_id": folder_id,
            "R": R,                        # Matrice Rotazione (3x3)
            "T": T,                        # Vettore Traslazione (3)
            "obj_bb": obj_bb,              # BBox in pixel (4)
            "sample_id": sample_id         # Utile per il debug
        }
    
