"""
PhobiaShield - Simple Training Script
The Architect Module

Clean, working training script with all fixes included.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
from omegaconf import OmegaConf

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.models.phobia_net import PhobiaNet
from src.models.loss import PhobiaLoss
from src.data.phobia_dataset import PhobiaDataset


def train_phobiashield(
    data_root='data/phobiashield_final',
    model_config='cfg/model/tiny_yolo_5class.yaml',
    epochs=10,
    batch_size=8,
    lr=0.0001,
    device='cuda'
):
    """
    Main training function.
    
    Args:
        data_root: Path to dataset
        model_config: Path to model config YAML
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("PhobiaShield Training - The Architect")
    print("="*60)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")
    
    # Load config
    print("🔧 Loading configuration...")
    try:
        model_cfg = OmegaConf.load(model_config)
        num_classes = model_cfg.output.num_classes
        print(f"✓ Model config loaded: {num_classes} classes")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("Using default: 5 classes")
        num_classes = 5
    
    # Initialize model
    print("\n🔧 Initializing model...")
    try:
        model = PhobiaNet(model_cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized: {total_params:,} parameters ({total_params*4/1e6:.2f} MB)")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Initialize loss
    print("\n🔧 Initializing loss function...")
    loss_fn = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        grid_size=13,
        num_boxes=2,
        num_classes=num_classes
    )
    print("✓ Loss function initialized")
    
    # Load datasets
    print("\n📦 Loading datasets...")
    data_root = Path(data_root)
    
    try:
        train_dataset = PhobiaDataset(
            img_dir=data_root / 'train' / 'images',
            label_dir=data_root / 'train' / 'labels',
            img_size=416,
            grid_size=13,
            num_boxes=2,
            num_classes=num_classes,
            augment=False  # Disabled for stability
        )
        
        val_dataset = PhobiaDataset(
            img_dir=data_root / 'val' / 'images',
            label_dir=data_root / 'val' / 'labels',
            img_size=416,
            grid_size=13,
            num_boxes=2,
            num_classes=num_classes,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✓ Train: {len(train_dataset)} images")
        print(f"✓ Val:   {len(val_dataset)} images")
        print(f"✓ Batch size: {batch_size}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Initialize optimizer
    print("\n⚙️  Initializing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    print(f"✓ Optimizer: Adam (lr={lr})")
    print(f"✓ Scheduler: ReduceLROnPlateau")
    
    # Create output directory
    output_dir = Path('outputs/checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for {epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train phase
        model.train()
        train_loss = 0
        train_metrics = {'coord': 0, 'obj': 0, 'noobj': 0, 'class': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss, metrics = loss_fn(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Accumulate
            train_loss += metrics['total_loss']
            train_metrics['coord'] += metrics['coord_loss']
            train_metrics['obj'] += metrics['conf_loss_obj']
            train_metrics['noobj'] += metrics['conf_loss_noobj']
            train_metrics['class'] += metrics['class_loss']
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.2f}",
                'coord': f"{metrics['coord_loss']:.2f}",
                'noobj': f"{metrics['conf_loss_noobj']:.1f}"
            })
        
        # Average train metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'coord': 0, 'obj': 0, 'noobj': 0, 'class': 0}
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss, metrics = loss_fn(outputs, targets)
                
                val_loss += metrics['total_loss']
                val_metrics['coord'] += metrics['coord_loss']
                val_metrics['obj'] += metrics['conf_loss_obj']
                val_metrics['noobj'] += metrics['conf_loss_noobj']
                val_metrics['class'] += metrics['class_loss']
                
                pbar_val.set_postfix({'val_loss': f"{metrics['total_loss']:.2f}"})
        
        # Average val metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.2f} (coord={train_metrics['coord']:.2f}, noobj={train_metrics['noobj']:.1f})")
        print(f"  Val Loss:   {val_loss:.2f} (coord={val_metrics['coord']:.2f}, noobj={val_metrics['noobj']:.1f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (val_loss: {val_loss:.2f})\n")
        else:
            print()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Checkpoint saved (epoch {epoch+1})\n")
    
    print(f"{'='*60}")
    print(f"✅ Training complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Model saved: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PhobiaShield')
    parser.add_argument('--data', type=str, default='data/phobiashield_final', help='Dataset root')
    parser.add_argument('--config', type=str, default='cfg/model/tiny_yolo_5class.yaml', help='Model config')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_phobiashield(
        data_root=args.data,
        model_config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )