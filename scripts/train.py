import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# --- IMPORTAZIONI CUSTOM ---
# Assumiamo che il Membro 1 rispetti i nomi delle classi concordati
from src.models.phobia_net import PhobiaNet 
from src.models.loss import PhobiaLoss
from src.data.dataset import PhobiaDataset

# --- HYPERPARAMETERS & CONFIG ---
# Questi parametri definiscono "come" impara il cervello
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "best_checkpoint.pth.tar"
IMG_DIR = "PhobiaDataset/images"
LABEL_DIR = "PhobiaDataset/labels"

# Parametri specifici dell'Architettura
SPLIT_SIZE = 13       # Griglia 13 x 13
NUM_BOXES = 2        # 2 box per cella
NUM_CLASSES = 5      # Ragni (0) , Aghi, Serpenti, Squali, Sangue


# Configurazione che phobia_net.py si aspetta
MODEL_CONFIG = {
    "output": {
        "num_classes": NUM_CLASSES
    },
    "architecture": {
        "grid_size": SPLIT_SIZE,  # Nota: deve essere coerente con i pooling (vedi sotto*)
        "num_boxes_per_cell": NUM_BOXES,
        "in_channels": 3,
        "leaky_relu_slope": 0.1,
        # Qui definiamo i layer come descritto nei commenti di PhobiaNet
        "layers": [
            {"filters": 16, "kernel_size": 3, "stride": 1, "pool": True},  # -> 208
            {"filters": 32, "kernel_size": 3, "stride": 1, "pool": True},  # -> 104
            {"filters": 64, "kernel_size": 3, "stride": 1, "pool": True},  # -> 52
            {"filters": 128, "kernel_size": 3, "stride": 1, "pool": True}, # -> 26
            {"filters": 256, "kernel_size": 3, "stride": 1, "pool": True}, # -> 13
            {"filters": 512, "kernel_size": 3, "stride": 1, "padding": 1, "pool": False}, # -> 13 (Output finale)
        ]
    },
    "init": {"type": "kaiming"}
}


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fn(train_loader, model, optimizer, loss_fn):
    """
    Funzione che gestisce una singola epoca di addestramento.
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 1. Forward Pass
        # Il modello sputa le predizioni
        out = model(x)
        
        # 2. Calcolo Loss
        # La Loss Function confronta le predizioni (out) con la verità (y)
        loss = loss_fn(out, y)
        
        # Aggiungiamo il valore alla lista per calcolare la media alla fine
        mean_loss.append(loss.item())
        
        # 3. Backward Pass (Backpropagation)
        optimizer.zero_grad() # Azzeriamo i gradienti vecchi
        loss.backward()       # Calcoliamo i nuovi gradienti
        optimizer.step()      # Aggiorniamo i pesi della rete

        # Aggiorniamo la barra di progresso
        loop.set_description(f"Loss: {loss.item():.4f}")

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def main():
    seed_everything(42) 
    print(f"[SETUP] Device: {DEVICE}")
    
    # 1. Inizializzazione Modello (Responsabilità: Membro 1 + Membro 2)
    # Nota: Passiamo i parametri dinamici per adattare la rete ai nostri 5 target
    model = PhobiaNet(config=MODEL_CONFIG).to(DEVICE)
    
    # 2. Inizializzazione Optimizer
    # Adam è lo standard d'oro per la convergenza veloce
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 3. Inizializzazione Loss (Responsabilità: Membro 1)
    loss_fn = PhobiaLoss(grid_size=SPLIT_SIZE,num_boxes=NUM_BOXES,num_classes=NUM_CLASSES)

    # 4. Caricamento Checkpoint (Opzionale)
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # 5. Dataset e DataLoader (Responsabilità: Membro 2) [Cite: 31]
    # Augment=True è fondamentale qui per usare la tua data augmentation "on the fly"
    train_dataset = PhobiaDataset(
        list_path=IMG_DIR,
        transform=None, # La trasformazione è gestita internamente dalla classe col parametro augment
        augment=True,
        img_size=416
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True, # Mischiamo i dati per evitare bias
        drop_last=False,
        collate_fn=PhobiaDataset.collate_fn # Fondamentale per gestire batch di box variabili
    )

    print(f"[INFO] Inizio training su {len(train_dataset)} immagini per {EPOCHS} epoche.")

    # 6. Ciclo di Training [Cite: 32]
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_fn(train_loader, model, optimizer, loss_fn)
        
        # Salviamo il checkpoint ad ogni epoca (o potremmo farlo ogni 10)
        # Questo file sarà quello che passerai al Membro 3 per la Demo
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

if __name__ == "__main__":
    main()