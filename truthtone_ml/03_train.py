"""
03_train.py
Main training script for the deepfake audio detection model.

Supports:
  - Single GPU training
  - Multi-GPU on one machine (DataParallel or DDP)
  - Multi-node distributed (2 servers with shared filesystem)
  - Mixed precision (FP16) for 2x speed on RTX 6000 Pro
  - Cosine annealing LR schedule with warmup
  - Early stopping
  - TensorBoard logging
  - Checkpoint saving/resuming

Usage:
    # Single GPU
    python 03_train.py --epochs 30 --batch-size 64

    # 2 GPUs on same machine
    torchrun --nproc_per_node=2 03_train.py --epochs 30 --batch-size 64 --distributed

    # 2 GPUs on 2 servers (shared filesystem)
    # Server 1:
    torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
             --nproc_per_node=1 03_train.py --distributed --epochs 30
    # Server 2:
    torchrun --nnodes=2 --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 \
             --nproc_per_node=1 03_train.py --distributed --epochs 30
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime

from config import (
    CHECKPOINT_DIR, LOG_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA,
    WARMUP_EPOCHS, EARLY_STOP_PATIENCE, SPEC_DIR, AUGMENT_TRAIN,
    AUG_MIXUP_ALPHA
)
from model import build_model
from dataset import SpectrogramDataset


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log(msg):
    """Print only from main process."""
    if is_main_process():
        print(msg)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blend pairs of samples."""
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup: weighted combination of losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, 
                    epoch, use_mixup=False, mixup_alpha=0.2):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with autocast("cuda"):
            if use_mixup and np.random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step LR scheduler per batch (for warmup + cosine annealing)
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if use_mixup and 'lam' in dir():
            # For mixup, count as correct if matches either label
            correct += (lam * predicted.eq(labels_a).sum().item() +
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0 and is_main_process():
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.1f}%", end="\r")
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def get_lr_scheduler(optimizer, num_epochs, warmup_epochs, loader_len):
    """Create learning rate scheduler with warmup."""
    total_steps = num_epochs * loader_len
    warmup_steps = warmup_epochs * loader_len
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    if LR_SCHEDULER == "cosine":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_acc, val_loss, path):
    """Save model checkpoint."""
    # Handle DDP wrapper
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('val_acc', 0), checkpoint.get('val_loss', float('inf'))


def main():
    parser = argparse.ArgumentParser(description="Train TruthTone++ detection model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet18", "resnet34", "efficientnet_b0"])
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup augmentation")
    parser.add_argument("--unfreeze-epoch", type=int, default=5, 
                        help="Epoch to unfreeze all layers (0=never)")
    args = parser.parse_args()
    
    # ── Setup ──
    local_rank = 0
    if args.distributed:
        local_rank = setup_distributed()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(device)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # ── Directories ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Data ──
    log("\nLoading datasets...")
    train_dir = SPEC_DIR / "train"
    val_dir = SPEC_DIR / "dev"
    
    if not train_dir.exists():
        log(f"ERROR: Training data not found at {train_dir}")
        log("Run 02_preprocess.py first.")
        sys.exit(1)
    
    train_dataset = SpectrogramDataset(train_dir, augment=AUGMENT_TRAIN)
    val_dataset = SpectrogramDataset(val_dir, augment=False)
    
    # Distributed sampler
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        # Use balanced sampler for single-GPU
        train_sampler = train_dataset.get_sampler()
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
    )
    
    log(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    log(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # ── Model ──
    log(f"\nBuilding model: {args.model}")
    model = build_model(model_name=args.model)
    model = model.to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    
    # ── Loss (with class weights for imbalance) ──
    class_weights = train_dataset.get_class_weights().to(device)
    log(f"Class weights: real={class_weights[0]:.3f}, fake={class_weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ── Optimizer ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # ── LR Scheduler ──
    scheduler = get_lr_scheduler(optimizer, args.epochs, WARMUP_EPOCHS, len(train_loader))
    
    # ── Mixed Precision ──
    scaler = GradScaler("cuda")
    
    # ── TensorBoard ──
    writer = None
    if is_main_process():
        run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(LOG_DIR / run_name)
    
    # ── Resume ──
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        log(f"Resuming from {args.resume}")
        start_epoch, best_val_acc, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        log(f"Resumed at epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # ── Training Loop ──
    best_val_loss = float('inf')
    patience_counter = 0
    use_mixup = not args.no_mixup and AUG_MIXUP_ALPHA > 0
    
    log(f"\n{'='*60}")
    log(f"Starting training: {args.epochs} epochs")
    log(f"Effective batch size: {args.batch_size * (dist.get_world_size() if args.distributed else 1)}")
    log(f"Mixup: {'ON' if use_mixup else 'OFF'}")
    log(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Set epoch for distributed sampler
        if args.distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # Unfreeze all layers after initial training
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            log(f"\n>>> Unfreezing all layers at epoch {epoch}")
            model_to_unfreeze = model.module if hasattr(model, 'module') else model
            model_to_unfreeze.unfreeze_all()
            # Reset optimizer with lower LR for fine-tuning
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=WEIGHT_DECAY
            )
            scheduler = get_lr_scheduler(optimizer, args.epochs - epoch, 1, len(train_loader))
            scaler = GradScaler("cuda")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device,
            epoch, use_mixup=use_mixup, mixup_alpha=AUG_MIXUP_ALPHA
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        log(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        
        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Save checkpoints (only from main process)
        if is_main_process():
            # Save latest
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_acc, val_loss,
                CHECKPOINT_DIR / "latest.pt"
            )
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_acc, val_loss,
                    CHECKPOINT_DIR / "best_model.pt"
                )
                log(f"  ★ New best model! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_acc, val_loss,
                    CHECKPOINT_DIR / f"epoch_{epoch+1}.pt"
                )
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break
    
    # ── Final Summary ──
    log(f"\n{'='*60}")
    log(f"TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"Best validation accuracy: {best_val_acc:.2f}%")
    log(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    log(f"\nNext step: python 04_evaluate.py --model {CHECKPOINT_DIR / 'best_model.pt'}")
    
    if writer:
        writer.close()
    
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
