from ultralytics import YOLO

# 1. Carica il tuo modello appena addestrato
model = YOLO('runs/detect/linemod_yolo_run/weights/best.pt')

# 2. Esegui la predizione su un'immagine (sostituisci con il percorso di una tua immagine)
# Se non hai un'immagine pronta, YOLO ne userà una di esempio o puoi scaricarne una
results = model('datasets/linemod/linemod_yolo_format/images/val/obj01_0254.png', show=True) 

# Oppure, se vuoi salvare l'immagine col risultato invece di mostrarla a video:
results[0].save(filename='risultato.jpg')
print("Fatto! Controlla se si è aperta la finestra o se hai salvato il file.")