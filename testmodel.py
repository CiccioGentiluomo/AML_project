from ultralytics import YOLO

# 1. Carica il tuo modello appena addestrato
model = YOLO('runs/detect/linemod_yolo_run/weights/best.pt')

# 2. Esegui la predizione su un'immagine (sostituisci con il percorso di una tua immagine)
# Se non hai un'immagine pronta, YOLO ne userà una di esempio o puoi scaricarne una
results = model('datasets/linemod/linemod_yolo_format/images/train/obj13_0047.png', show=True) 

for result in results:
    boxes = result.boxes
    for box in boxes:
        # Estraiamo le coordinate in formato xyxy (pixel)
        # .tolist() serve per avere i numeri puri senza approssimazioni di stampa dei tensori
        coords = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        
        print(f"--- INFO BOUNDING BOX PREDETTA ---")
        print(f"ID Classe: {cls}")
        print(f"Confidenza: {conf}")
        print(f"Coordinate (xmin, ymin, xmax, ymax): {coords}")
        
        # Calcoliamo anche larghezza e altezza per il confronto
        w = coords[2] - coords[0]
        h = coords[3] - coords[1]
        print(f"Dimensioni calcolate: Width={w}, Height={h}")

