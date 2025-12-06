"""
PhobiaShield - Simple Training Script
The Architect Module

OPTIMIZED VERSION with:
- Early stopping (patience=5)
- Weight decay (AdamW optimizer)
- Augmentation enabled
- Class weighting
- Scheduler with min_lr
- Reduced default epochs (30)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
from omegaconf import OmegaConf
import argparse

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from src.models.phobia_net import PhobiaNet
from src.models.loss import PhobiaLoss
from src.data.phobia_dataset import PhobiaDataset


def train_phobiashield(
    data_root='data/phobiashield_final',
    model_config='cfg/model/tiny_yolo_5class.yaml',
    epochs=30,  # REDUCED from 100
    batch_size=8,
    lr=0.0001,
    device='cuda',
    resume=None,  # NEW: Resume support
    early_stop_patience=5  # NEW: Early stopping
):
    """
    Main training function with optimizations.
    
    Args:
        data_root: Path to dataset
        model_config: Path to model config YAML
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        resume: Path to checkpoint to resume from (optional)
        early_stop_patience: Stop if no improvement for N epochs
    """
    
    print("="*60)
    print("PhobiaShield Training - OPTIMIZED VERSION")
    print("The Architect")
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
    
    # Initialize loss with CLASS WEIGHTING
    print("\n🔧 Initializing loss function with class weighting...")
    class_weights = [1.0, 2.0, 1.7, 0.5, 10.0]  # Clown, Shark, Spider, Blood, Needle
    print(f"   Class weights: {class_weights}")
    print(f"   (Blood penalized 0.5x, Needle amplified 10x)")
    
    loss_fn = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        grid_size=13,
        num_boxes=2,
        num_classes=num_classes,
        class_weights=class_weights  # NEW!
    )
    print("✓ Loss function initialized with class balancing")
    
    # Load datasets with AUGMENTATION ENABLED
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
            augment=True  # ENABLED! (was False)
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
            num_workers=4,  # Increased from 2
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=PhobiaDataset.collate_fn  # Keep if defined
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Increased from 2
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=PhobiaDataset.collate_fn  # Keep if defined
        )
        
        print(f"✓ Train: {len(train_dataset)} images (augmentation: ON)")
        print(f"✓ Val:   {len(val_dataset)} images (augmentation: OFF)")
        print(f"✓ Batch size: {batch_size}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Initialize optimizer with WEIGHT DECAY (AdamW)
    print("\n⚙️  Initializing optimizer...")
    optimizer = optim.AdamW(  # Changed from Adam to AdamW!
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4  # L2 regularization
    )
    print(f"✓ Optimizer: AdamW (lr={lr}, weight_decay=1e-4)")
    
    # Scheduler with MINIMUM LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,  # NEW: Won't go below this!
        verbose=False
    )
    print(f"✓ Scheduler: ReduceLROnPlateau (min_lr=1e-6)")
    
    # Create output directory
    output_dir = Path('outputs/checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # RESUME: Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume:
        print(f"\n🔄 Resuming from checkpoint: {resume}")
        try:
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"✓ Best val_loss so far: {best_val_loss:.2f}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            start_epoch = 0
            best_val_loss = float('inf')
    
    # EARLY STOPPING setup
    print(f"\n🛑 Early stopping enabled (patience={early_stop_patience})")
    epochs_no_improve = 0
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting training from epoch {start_epoch} to {epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, epochs):
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
            
            # Accumulate (SAME FORMAT AS BEFORE!)
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
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {train_loss:.2f} (coord={train_metrics['coord']:.2f}, noobj={train_metrics['noobj']:.1f})")
        print(f"   Val Loss:   {val_loss:.2f} (coord={val_metrics['coord']:.2f}, noobj={val_metrics['noobj']:.1f})")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved (epoch {epoch+1})")
        
        # EARLY STOPPING CHECK
        if val_loss < best_val_loss:
            # New best model!
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            
            if epoch > 0:  # Skip "improved by" message on first epoch
                print(f"   🏆 New best model! Val Loss: {val_loss:.2f} (improved by {improvement:.2f})\n")
            else:
                print(f"   🏆 Best model saved (val_loss: {val_loss:.2f})\n")
        else:
            # No improvement
            epochs_no_improve += 1
            print(f"   ⚠️  No improvement for {epochs_no_improve} epoch(s) (best: {best_val_loss:.2f})")
            
            # Check early stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"\n🛑 EARLY STOPPING triggered!")
                print(f"   No improvement for {early_stop_patience} epochs")
                print(f"   Best Val Loss: {best_val_loss:.2f} (Epoch {epoch+1-early_stop_patience})")
                print(f"   Stopping training at epoch {epoch+1}")
                break
            
            print()
    
    # Training completed
    print(f"{'='*60}")
    print(f"✅ Training complete!")
    if epochs_no_improve >= early_stop_patience:
        print(f"🛑 Stopped early (saved {epochs - epoch - 1} epochs)")
    print(f"{'='*60}")
    print(f"📈 Best validation loss: {best_val_loss:.2f}")
    print(f"💾 Model saved: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PhobiaShield - Optimized')
    parser.add_argument('--data', type=str, default='data/phobiashield_final', help='Dataset root')
    parser.add_argument('--config', type=str, default='cfg/model/tiny_yolo_5class.yaml', help='Model config')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (reduced from 100)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_phobiashield(
        data_root=args.data,
        model_config=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        resume=args.resume,
        early_stop_patience=args.patience
    )