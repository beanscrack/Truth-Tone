"""
04_evaluate.py
Comprehensive evaluation of the trained model.

Produces:
  - Accuracy, Precision, Recall, F1 per class
  - Confusion matrix (saved as image)
  - ROC curve with AUC (saved as image)
  - EER (Equal Error Rate) - standard metric for spoofing detection
  - Per-attack-type breakdown (if using ASVspoof)
  - Saves all metrics as JSON for documentation

Usage:
    python 04_evaluate.py --model checkpoints/best_model.pt
    python 04_evaluate.py --model checkpoints/best_model.pt --split eval
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from config import SPEC_DIR, CHECKPOINT_DIR, CLASS_NAMES, BATCH_SIZE
from model import build_model
from dataset import SpectrogramDataset


def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER).
    
    EER is the standard metric for speaker verification and anti-spoofing.
    It's the point where False Acceptance Rate = False Rejection Rate.
    Lower is better.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run model on all samples and collect predictions."""
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        
        with autocast(device.type):
            outputs = model(images)
        
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_probs, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - TruthTone++ Deepfake Detector', fontsize=14)
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_scores, eer, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#E94560', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    
    # Mark EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    ax.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
            label=f'EER = {eer:.4f}')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - TruthTone++ Deepfake Detector', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curve to {save_path}")


def plot_score_distribution(y_true, y_scores, save_path):
    """Plot distribution of model scores for real vs fake."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    
    ax.hist(real_scores, bins=50, alpha=0.6, color='#2ECC71', label='Real audio', density=True)
    ax.hist(fake_scores, bins=50, alpha=0.6, color='#E94560', label='Fake audio', density=True)
    
    ax.set_xlabel('Model Score (probability of being fake)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution - Real vs Fake', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved score distribution to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TruthTone++ model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="eval", choices=["train", "dev", "eval"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Load Model ──
    print(f"\nLoading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val accuracy at save: {checkpoint.get('val_acc', '?')}")
    
    # ── Load Data ──
    split_dir = SPEC_DIR / args.split
    if not split_dir.exists():
        print(f"ERROR: {split_dir} not found. Available splits:")
        for d in SPEC_DIR.iterdir():
            if d.is_dir():
                print(f"  {d.name}")
        return
    
    dataset = SpectrogramDataset(split_dir, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # ── Evaluate ──
    print(f"\nEvaluating on {args.split} split ({len(dataset)} samples)...")
    probs, preds, labels = evaluate_model(model, loader, device)
    
    # Fake probability scores (for ROC/EER)
    fake_scores = probs[:, 1]
    
    # ── Metrics ──
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Classification report
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    
    # Overall accuracy
    accuracy = (preds == labels).mean() * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # EER
    eer, eer_threshold = compute_eer(labels, fake_scores)
    print(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
    print(f"EER Threshold: {eer_threshold:.4f}")
    
    # Accuracy using EER-optimal threshold
    eer_preds = (fake_scores >= eer_threshold).astype(int)
    eer_accuracy = (eer_preds == labels).mean() * 100
    print(f"Accuracy @ EER Threshold: {eer_accuracy:.2f}%")
    
    # AUC
    fpr, tpr, _ = roc_curve(labels, fake_scores)
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # ── Save Results ──
    output_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plots
    plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    plot_roc_curve(labels, fake_scores, eer, output_dir / "roc_curve.png")
    plot_score_distribution(labels, fake_scores, output_dir / "score_distribution.png")
    
    # JSON metrics
    metrics = {
        "split": args.split,
        "num_samples": len(dataset),
        "accuracy": float(accuracy),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "auc_roc": float(roc_auc),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "model_checkpoint": str(args.model),
        "checkpoint_epoch": checkpoint.get('epoch', None),
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics to {metrics_path}")
    
    print(f"\n  All results saved to {output_dir}/")
    print(f"\nNext step: python 05_api.py --model {args.model}")


if __name__ == "__main__":
    main()
