from src.data.convert_to_yolo import AnnotationConverter
import shutil
from pathlib import Path

def prepare_dataset(source_name, raw_path, output_path):
    print(f"⚙️  Processing external dataset: {source_name}")
    
    # 1. Istanziamo il converter
    # CORREZIONE: La mappa deve essere { "nome_classe": ID }
    # Usiamo 0 temporaneamente per questo singolo dataset
    converter = AnnotationConverter({source_name: 0})
    
    # 2. Creiamo lo split 80/20
    print(f"   Splitting files from {raw_path}...")
    converter.create_train_val_split(
        images_dir=f"{raw_path}/images",
        labels_dir=f"{raw_path}/labels",
        output_dir=output_path,
        val_split=0.2
    )
    print(f"✅ {source_name} ready at {output_path}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    
    # CASO RAGNI (Gianlu)
    prepare_dataset(
        source_name="spider",
        raw_path="data/external/ragni",       
        output_path="data/final/spider"       
    )