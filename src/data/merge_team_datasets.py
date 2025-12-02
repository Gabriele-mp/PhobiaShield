import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# ==============================================================================
# CONFIGURAZIONE MASTER DEL TEAM (The Architect's Law)
# ==============================================================================
GLOBAL_MAPPING = {
    "clown": 0,
    "shark": 1,
    # Spazio per i futuri compagni:
    "spider": 2,
    # "doll": 3,
    # "needle": 4, 
    # "blood": 5
}

def merge_datasets(source_dirs, output_dir):
    """
    Fonde diversi dataset YOLO in uno unico, rinominando i file e rimappando gli ID.
    Agisce anche da filtro: le classi non mappate vengono scartate.
    """
    output_dir = Path(output_dir)
    print(f"🏗️  Merging datasets into {output_dir}...")
    
    # 1. Pulisci la cartella di destinazione se esiste già
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # 2. Crea la struttura delle cartelle (train/val)
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. Processa ogni dataset sorgente
    for name, path, local_mapping in source_dirs:
        print(f"\n📦 Processing dataset: {name}")
        path = Path(path)
        
        # Creiamo la mappa di conversione ID: (es: ID Locale 0 -> ID Globale 1)
        id_map = {}
        for local_id, class_name in local_mapping.items():
            if class_name in GLOBAL_MAPPING:
                global_id = GLOBAL_MAPPING[class_name]
                id_map[local_id] = global_id
                print(f"   Mapping '{class_name}': ID {local_id} -> Global ID {global_id}")
            else:
                print(f"   ⚠️ Warning: Classe '{class_name}' non trovata nel Global Mapping. Verrà ignorata.")

        # Processa sia train che val
        for split in ['train', 'val']:
            src_imgs = path / split / 'images'
            src_lbls = path / split / 'labels'
            
            if not src_imgs.exists():
                continue

            files = list(src_imgs.glob("*"))
            # Usa tqdm per la barra di caricamento
            for img_path in tqdm(files, desc=f"   Merging {split}"):
                
                # A. Copia e Rinomina Immagine
                # Aggiungiamo il prefisso (es: "gabriele_shark_") per evitare file duplicati
                new_filename = f"{name}_{img_path.name}"
                dest_img = output_dir / split / 'images' / new_filename
                shutil.copy2(img_path, dest_img)
                
                # B. Processa la Label (Filtro e Conversione)
                lbl_name = img_path.stem + ".txt"
                src_lbl = src_lbls / lbl_name
                
                # Se l'immagine ha una label corrispondente...
                if src_lbl.exists():
                    dest_lbl = output_dir / split / 'labels' / (Path(new_filename).stem + ".txt")
                    
                    # Leggi vecchia label, scrivi nuova label
                    valid_lines = []
                    with open(src_lbl, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts: continue
                            
                            old_id = int(parts[0])
                            
                            # IL FILTRO MAGICO:
                            # Scriviamo la riga SOLO se l'ID vecchio è nella nostra mappa.
                            # Se old_id è 1 (Balena) e non è nella mappa, viene scartato.
                            if old_id in id_map:
                                new_id = id_map[old_id]
                                # Ricostruisci la riga con il nuovo ID
                                new_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                                valid_lines.append(new_line)
                    
                    # Salviamo il file solo se ci sono righe valide (non salviamo file vuoti)
                    if valid_lines:
                        with open(dest_lbl, 'w') as f_out:
                            f_out.writelines(valid_lines)

    # 4. Genera il file data.yaml finale per YOLO
    create_global_yaml(output_dir)
    print("\n✅ GLOBAL DATASET READY!")
    print(f"   Location: {output_dir}")

def create_global_yaml(output_dir):
    # Inverti la mappa per avere una lista ordinata di nomi
    # [0: 'clown', 1: 'shark', ...]
    names = [''] * len(GLOBAL_MAPPING)
    for name, idx in GLOBAL_MAPPING.items():
        names[idx] = name
        
    yaml_data = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': names,
        'nc': len(names)
    }
    
    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

# ==============================================================================
# ESECUZIONE
# ==============================================================================
if __name__ == "__main__":
    
    # CONFIGURAZIONE DELLE SORGENTI
    # Formato: ("Prefisso_File", "Percorso_Cartella", {ID_LOCALE: "NOME_CLASSE_GLOBALE"})
    
    SOURCES = [
        # 1. Dataset CLOWN
        # Nella cartella originale, ID 0 significa "clown".
        ("gabriele_clown", "data/final/clown", {0: "clown"}),
        
        # 2. Dataset SHARK
        # Nella cartella originale, ID 0 significa "shark".
        # NOTA: Gli ID 1-37 (balene, persone, ecc.) verranno SCARTATI automaticamente
        # perché non li stiamo inserendo in questo dizionario {0: "shark"}.
        ("gabriele_shark", "data/final/shark", {0: "shark"}),
        
        ("gianlu_spider", "data/final/spider", {0: "spider"}),
    ]
    
    # Avvia il processo
    merge_datasets(SOURCES, "data/global_dataset")