import cv2
import numpy as np

# Configurazione
FILENAME = "input_videos/test_video.mp4"
DURATION = 5  # secondi
FPS = 30
WIDTH, HEIGHT = 1280, 720

def create_dummy():
    # Crea cartella se non esiste
    import os
    if not os.path.exists("input_videos"):
        os.makedirs("input_videos")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
    
    print(f"Generazione video finto di {DURATION} secondi...")
    
    for i in range(DURATION * FPS):
        # Sfondo nero
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Disegna un cerchio che si muove (per simulare movimento)
        x = int((i * 10) % WIDTH)
        y = int(HEIGHT // 2)
        cv2.circle(frame, (x, y), 50, (0, 255, 255), -1)
        
        # Scritta
        cv2.putText(frame, f"FRAME {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"[SUCCESS] Video creato: {FILENAME}")

if __name__ == "__main__":
    create_dummy()