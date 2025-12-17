"""
PhobiaShield - YOLOv8 Training Script

Trains YOLOv8s model on DATASET_ULTIMATE_COMPLETE for baseline comparison.

Usage:
    python scripts/train_yolov8.py --data data/phobiashield.yaml --epochs 50
"""

from ultralytics import YOLO
import os
import argparse
from pathlib import Path
import yaml

def create_data_yaml(dataset_path, output_path='data/phobiashield.yaml'):
    """Create YOLOv8 data.yaml file"""
    
    data_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 5,
        'names': ['clown', 'shark', 'spider', 'blood', 'needle']
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml: {output_path}")
    return output_path

def train_yolov8(
    data_yaml='data/phobiashield.yaml',
    model_size='yolov8s.pt',
    epochs=50,
    batch=64,
    imgsz=416,
    save_dir='outputs/yolov8s',
    patience=20,
    device=0
):
    """
    Train YOLOv8 model.
    
    Args:
        data_yaml: Path to data.yaml
        model_size: Model size (yolov8n/s/m/l/x)
        epochs: Number of epochs
        batch: Batch size
        imgsz: Image size
        save_dir: Save directory
        patience: Early stopping patience
        device: GPU device (0, 1, ...) or 'cpu'
    """
    
    print("="*70)
    print("🚀 PhobiaShield - YOLOv8 Training")
    print("="*70)
    
    # Verify data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml not found: {data_yaml}")
        return None
    
    # Load model
    print(f"\n📦 Loading model: {model_size}")
    model = YOLO(model_size)
    
    # Print model info
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training config
    print(f"\n⚙️  Training configuration:")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {device}")
    print(f"   Patience: {patience}")
    
    print(f"\n🚀 Starting training...")
    print("="*70 + "\n")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=save_dir,
        name='train',
        exist_ok=True,
        
        # Optimization
        amp=True,           # Mixed precision
        augment=True,       # Data augmentation
        
        # Hyperparameters (optimized for PhobiaShield)
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final learning rate fraction
        momentum=0.937,     # Momentum
        weight_decay=0.0005,# Weight decay
        
        # Loss weights (tuned)
        box=7.5,            # Box loss weight
        cls=0.5,            # Classification loss weight
        dfl=1.5             # Distribution focal loss weight
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Print results
    best_model_path = f"{save_dir}/train/weights/best.pt"
    print(f"\n💾 Best model saved: {best_model_path}")
    
    # Validation results
    print(f"\n📊 Final Results:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return results

def evaluate_model(model_path, data_yaml):
    """Evaluate trained model"""
    
    print(f"\n📊 Evaluating model: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")
    print("="*70)
    
    return results

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 on PhobiaShield')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='data/phobiashield_ultimate',
                        help='Dataset root directory')
    parser.add_argument('--data', type=str, default='data/phobiashield.yaml',
                        help='data.yaml path (will be created if not exists)')
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='Model size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Image size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device (0, 1, ...) or cpu')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='outputs/yolov8s',
                        help='Save directory')
    
    # Actions
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Create data.yaml if needed
    if not os.path.exists(args.data):
        print(f"📝 Creating data.yaml...")
        create_data_yaml(args.dataset, args.data)
    
    # Train
    if not args.no_train:
        results = train_yolov8(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            save_dir=args.save_dir,
            patience=args.patience,
            device=args.device
        )
        
        if results is None:
            return 1
    
    # Evaluate
    if args.evaluate:
        best_model = f"{args.save_dir}/train/weights/best.pt"
        if os.path.exists(best_model):
            evaluate_model(best_model, args.data)
        else:
            print(f"❌ Model not found: {best_model}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
