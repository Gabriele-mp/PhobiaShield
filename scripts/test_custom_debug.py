import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.detector import PhobiaDetector

# Configurazione
MODEL_PATH = "weights/FPN_epoch18_loss6.3037.pth"
TEST_IMG = "test_images_input/shark.jpg" # O una qualsiasi immagine che hai

def main():
    if not os.path.exists(MODEL_PATH):
        print("Modello non trovato!")
        return

    print("Inizializzazione Detector Custom...")
    # Qui carichiamo il CUSTOM
    detector = PhobiaDetector(model_path=MODEL_PATH, model_type='custom')

    img = cv2.imread(TEST_IMG)
    if img is None:
        # Se non trova l'immagine specifica, prova a cercarne una a caso nella cartella
        import glob
        files = glob.glob("test_images_input/*.jpg")
        if files:
            img = cv2.imread(files[0])
            print(f"Usata immagine: {files[0]}")
        else:
            print("Nessuna immagine trovata.")
            return

    print("Esecuzione Predict...")
    detector.predict(img)
    print("Test completato.")

if __name__ == "__main__":
    main()