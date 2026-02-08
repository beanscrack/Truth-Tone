"""
dataset.py
PyTorch Dataset for loading preprocessed mel spectrograms.

Includes SpecAugment and other augmentations for training robustness.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
try:
    from .config import (
        SPEC_DIR, SPEC_IMAGE_SIZE, CLASS_MAP, LABEL_BONAFIDE, LABEL_SPOOF,
        AUGMENT_TRAIN, AUG_TIME_MASK_MAX, AUG_FREQ_MASK_MAX, AUG_NOISE_STD
    )
except ImportError:
    from config import (
        SPEC_DIR, SPEC_IMAGE_SIZE, CLASS_MAP, LABEL_BONAFIDE, LABEL_SPOOF,
        AUGMENT_TRAIN, AUG_TIME_MASK_MAX, AUG_FREQ_MASK_MAX, AUG_NOISE_STD
    )


class SpectrogramDataset(Dataset):
    """
    Dataset that loads preprocessed .npy spectrogram files.
    
    Directory structure expected:
    split_dir/
    ├── real/     *.npy files
    └── fake/     *.npy files
    """
    
    def __init__(self, split_dir, augment=False):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.samples = []  # list of (path, label)
        
        # Load all file paths
        real_dir = self.split_dir / "real"
        fake_dir = self.split_dir / "fake"
        
        if real_dir.exists():
            for f in sorted(real_dir.glob("*.npy")):
                self.samples.append((f, 0))  # 0 = real
        
        if fake_dir.exists():
            for f in sorted(fake_dir.glob("*.npy")):
                self.samples.append((f, 1))  # 1 = fake
        
        # Class counts for balancing
        self.n_real = sum(1 for _, l in self.samples if l == 0)
        self.n_fake = sum(1 for _, l in self.samples if l == 1)
        
        print(f"  Loaded {len(self.samples)} samples "
              f"(real: {self.n_real}, fake: {self.n_fake}) "
              f"from {self.split_dir.name}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load spectrogram
        spec = np.load(path).astype(np.float32)
        
        # Normalize to [0, 1]
        spec = spec / 255.0
        
        # Apply augmentations
        if self.augment:
            spec = self._augment(spec)
        
        # Add channel dimension: (H, W) → (1, H, W)
        spec = np.expand_dims(spec, axis=0)
        
        return torch.FloatTensor(spec), torch.LongTensor([label])[0]
    
    def _augment(self, spec):
        """
        Apply SpecAugment + noise augmentation.
        
        SpecAugment (Park et al., 2019):
          - Time masking: zero out a random contiguous block of time steps
          - Frequency masking: zero out a random contiguous block of frequency bins
        
        These are the standard augmentations for audio ML and significantly
        improve robustness.
        """
        h, w = spec.shape  # (freq_bins, time_steps)
        
        # Time masking (mask a vertical stripe)
        if AUG_TIME_MASK_MAX > 0 and np.random.random() < 0.5:
            t = np.random.randint(0, AUG_TIME_MASK_MAX)
            t0 = np.random.randint(0, max(1, w - t))
            spec[:, t0:t0 + t] = 0
        
        # Frequency masking (mask a horizontal stripe)
        if AUG_FREQ_MASK_MAX > 0 and np.random.random() < 0.5:
            f = np.random.randint(0, AUG_FREQ_MASK_MAX)
            f0 = np.random.randint(0, max(1, h - f))
            spec[f0:f0 + f, :] = 0
        
        # Gaussian noise
        if AUG_NOISE_STD > 0 and np.random.random() < 0.3:
            noise = np.random.normal(0, AUG_NOISE_STD, spec.shape).astype(np.float32)
            spec = np.clip(spec + noise, 0, 1)
        
        # Random brightness/contrast shift
        if np.random.random() < 0.3:
            brightness = np.random.uniform(-0.1, 0.1)
            contrast = np.random.uniform(0.9, 1.1)
            spec = np.clip((spec - 0.5) * contrast + 0.5 + brightness, 0, 1)
        
        return spec
    
    def get_class_weights(self):
        """
        Compute class weights for imbalanced dataset.
        ASVspoof has ~10x more fake than real samples.
        """
        total = len(self.samples)
        w_real = total / (2 * max(self.n_real, 1))
        w_fake = total / (2 * max(self.n_fake, 1))
        return torch.FloatTensor([w_real, w_fake])
    
    def get_sampler(self):
        """
        Create a WeightedRandomSampler for balanced batches.
        This ensures each batch has roughly 50/50 real/fake.
        """
        weights = []
        w_real = 1.0 / max(self.n_real, 1)
        w_fake = 1.0 / max(self.n_fake, 1)
        
        for _, label in self.samples:
            weights.append(w_real if label == 0 else w_fake)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


def create_dataloaders(batch_size, num_workers=4, use_balanced_sampler=True):
    """
    Create train, validation, and test dataloaders.
    
    Uses ASVspoof splits:
      - train → training
      - dev → validation  
      - eval → test
    """
    train_dir = SPEC_DIR / "train"
    val_dir = SPEC_DIR / "dev"
    test_dir = SPEC_DIR / "eval"
    
    # Fall back to custom directory if ASVspoof not found
    if not train_dir.exists():
        train_dir = SPEC_DIR / "custom"
        val_dir = SPEC_DIR / "custom"  # Will need manual split
        test_dir = SPEC_DIR / "custom"
        print("WARNING: Using custom directory. Consider splitting data manually.")
    
    train_dataset = SpectrogramDataset(train_dir, augment=AUGMENT_TRAIN)
    val_dataset = SpectrogramDataset(val_dir, augment=False)
    
    test_dataset = None
    if test_dir.exists() and test_dir != train_dir:
        test_dataset = SpectrogramDataset(test_dir, augment=False)
    
    # Balanced sampler for training (handles class imbalance)
    train_sampler = train_dataset.get_sampler() if use_balanced_sampler else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    # Class weights for loss function
    class_weights = train_dataset.get_class_weights()
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader, weights = create_dataloaders(
        batch_size=32, num_workers=0
    )
    
    print(f"\nClass weights: {weights}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels[:10]}")
    print(f"Label distribution: real={sum(labels==0).item()}, fake={sum(labels==1).item()}")
