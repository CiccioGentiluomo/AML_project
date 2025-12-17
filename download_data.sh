#!/bin/bash

# Interrompi se c'è un errore
set -e

echo "============================================="
echo "   SETUP PIPELINE: LINEMOD & YOLO DATA"
echo "============================================="

# 0. Verifica preliminare
if [ ! -f "data/prepare_yolo_data.py" ]; then
    echo "❌ ERRORE: Non trovo il file 'prepare_yolo_data.py' dentro la cartella 'data/'!"
    echo "   Assicurati di aver creato la cartella 'data' e messo lì il file python."
    exit 1
fi

# 1. Crea la cartella per i datasets (nella root)
echo "📂 [1/5] Creazione cartella datasets/linemod..."
mkdir -p datasets/linemod

# 2. Scarica i dati
echo "⬇️  [2/5] Scaricamento dataset..."
# Controlla se il zip esiste già per evitare riscaricamenti inutili
if [ ! -f "datasets/linemod/Linemod_preprocessed.zip" ]; then
    gdown --fuzzy "https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing" -O datasets/linemod/Linemod_preprocessed.zip
else
    echo "   Archivio già presente, salto il download."
fi

# 3. Estrazione
echo "📦 [3/5] Estrazione archivio..."
unzip -q -o datasets/linemod/Linemod_preprocessed.zip -d datasets/linemod/

# 4. Pulizia
echo "🧹 [4/5] Rimozione file zip temporaneo..."
rm datasets/linemod/Linemod_preprocessed.zip

# 5. Esecuzione Script Python (Dalla cartella data/)
echo "⚙️  [5/5] Esecuzione data/prepare_yolo_data.py..."

# Lanciamo python specificando il percorso relativo 'data/...'
# NOTA: Python viene eseguito dalla root, quindi i percorsi dentro il py
# rimangono 'datasets/linemod/...' e funzioneranno correttamente.
python data/prepare_yolo_data.py

echo "============================================="
echo "✅ SETUP COMPLETATO!"
echo "   Output salvato in: datasets/linemod/linemod_yolo_format"
echo "============================================="