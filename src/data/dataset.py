import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random # Fondamentale per la casualità

class PhobiaDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=False):
        """
        Args:
            list_path (str): Path alla cartella images
            img_size (int): 416
            augment (bool): Se True, applica trasformazioni random (da usare solo in Training, non in Test!)
        """
        self.img_path = list_path
        self.label_path = list_path.replace("images", "labels")
        self.img_size = img_size
        self.augment = augment # Interruttore attivazione
        
        self.img_files = [
            f for f in os.listdir(self.img_path) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 1. Percorsi
        img_file = self.img_files[index]
        img_path = os.path.join(self.img_path, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_path, label_file)

        # 2. Carico Immagine
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Carico Label (se esiste)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                # Carichiamo tutto in una matrice numpy per facilità di calcolo
                # Formato: [class, x, y, w, h]
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if len(l):
                    boxes = np.array(l, dtype=np.float32)

        # 4. APPLICO DATA AUGMENTATION (Solo se abilitata)
        if self.augment:
            image, boxes = self.augment_image(image, boxes)

        # 5. Resize e Normalizzazione (Standard)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        # 6. Preparazione Target per PyTorch
        if len(boxes) > 0:
            # Aggiungiamo un indice 0 fittizio per il batch index
            # Target finale: [idx_batch, class, x, y, w, h]
            output_boxes = torch.zeros((len(boxes), 6))
            output_boxes[:, 1:] = torch.from_numpy(boxes)
        else:
            output_boxes = torch.zeros((0, 6))

        return image, output_boxes

    def augment_image(self, image, boxes):
        """
        Gestisce le trasformazioni casuali: Flip, Mirror, Grayscale.
        """
        # A. GRAYSCALE (Probabilità 20%)
        # Nota: Anche se diventa grigia, la riconvertiamo in RGB (3 canali)
        # perché la rete si aspetta sempre 3 canali in ingresso.
        if random.random() < 0.2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # B. FLIP ORIZZONTALE / SPECCHIO (Probabilità 50%)
        if random.random() < 0.5:
            image = cv2.flip(image, 1) # 1 = flip orizzontale
            if len(boxes) > 0:
                # Modifica coordinata X del centro: 1.0 - x
                boxes[:, 1] = 1.0 - boxes[:, 1]

        # C. FLIP VERTICALE / CAPOVOLTO (Probabilità 50%)
        if random.random() < 0.5:
            image = cv2.flip(image, 0) # 0 = flip verticale
            if len(boxes) > 0:
                # Modifica coordinata Y del centro: 1.0 - y
                boxes[:, 2] = 1.0 - boxes[:, 2]
        
        return image, boxes

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        for i, box in enumerate(targets):
            if box.shape[0] > 0:
                box[:, 0] = i 
        targets = torch.cat(targets, 0)
        return images, targets