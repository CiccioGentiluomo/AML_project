import argparse
import os
import wandb
from ultralytics import YOLO

def parse_opt():
    """
    Definisce e parsa gli argomenti da linea di comando.
    """
    parser = argparse.ArgumentParser(description='Addestramento YOLOv8/11 per LineMOD')
    
    # Percorsi e Modello
    parser.add_argument('--data', type=str, default='datasets/linemod/linemod_yolo_format/data.yaml', help='Percorso al file data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Modello di partenza (es. yolo11n.pt, yolov8n.pt)')
    parser.add_argument('--save_dir', type=str, default='runs/detect', help='Cartella dove salvare i pesi e i log')
    
    # Iperparametri di Training
    parser.add_argument('--epochs', type=int, default=50, help='Numero totale di epoche')
    parser.add_argument('--batch', type=int, default=32, help='Dimensione del batch (riduci se GPU OOM)')
    parser.add_argument('--imgsz', type=int, default=640, help='Dimensione immagini (pixel)')
    parser.add_argument('--workers', type=int, default=4, help='Numero di workers per il dataloader')
    parser.add_argument('--device', type=str, default='0', help='Dispositivo cuda (es. 0, 1, 2) o cpu')
    
    # Logging e Nomi
    parser.add_argument('--name', type=str, default='linemod_yolo_run', help='Nome dell\'esperimento/run')
    parser.add_argument('--wandb_project', type=str, default='linemod_yolo_training', help='Nome del progetto su WandB')
    
    return parser.parse_args()

def main(opt):
    # 1. Verifica Esistenza Dati
    if not os.path.exists(opt.data):
        print(f"❌ ERRORE: File yaml non trovato in: {opt.data}")
        print("   Assicurati di aver eseguito lo script di setup (setup_data.sh).")
        return

    # 2. Carica il modello
    print(f"🔄 Caricamento modello: {opt.model}...")
    model = YOLO(opt.model)  

    # 3. Addestramento
    print(f"🚀 Avvio training su device={opt.device} per {opt.epochs} epoche...")
    
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        name=opt.name,
        device=opt.device,
        project=opt.save_dir,   # Dove Ultralytics salva i file locali
        workers=opt.workers,
        
        # Opzioni extra utili
        exist_ok=True,          # Sovrascrive la cartella se esiste (utile per debug)
        pretrained=True,        # Usa pesi pre-addestrati
        optimizer='auto',       # Lascia scegliere a YOLO l'ottimizzatore migliore
        verbose=True
    )
    
    print("✅ Addestramento completato!")
    print(f"📁 Risultati salvati in: {os.path.join(opt.save_dir, opt.name)}")

if __name__ == "__main__":
    opt = parse_opt()
    
    # Inizializza WandB passando la configurazione (così vedi batch, lr, epochs sulla dashboard)
    wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))
    
    main(opt)