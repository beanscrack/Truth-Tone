"""
model.py
Model architecture for deepfake audio detection.

Core idea: Audio → Mel Spectrogram (2D image) → CNN classifier.
We use transfer learning from ImageNet-pretrained models because:
  1. Low-level features (edges, textures) transfer well to spectrograms
  2. Training from scratch on <50k samples risks overfitting
  3. Fine-tuning converges 10x faster than from scratch

The model learns to distinguish between:
  - Real audio: organic, irregular spectral patterns
  - Fake audio: subtle mathematical regularities from TTS/VC systems
"""
import torch
import torch.nn as nn
import torchvision.models as models
try:
    from .config import NUM_CLASSES, DROPOUT, FREEZE_LAYERS, UNFREEZE_FROM, MODEL_NAME
except ImportError:
    from config import NUM_CLASSES, DROPOUT, FREEZE_LAYERS, UNFREEZE_FROM, MODEL_NAME


class SpectrogramClassifier(nn.Module):
    """
    CNN classifier for mel spectrogram images.
    
    Input: (batch, 1, 224, 224) grayscale spectrogram
    Output: (batch, 2) logits for [real, fake]
    """
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, 
                 dropout=DROPOUT, pretrained=True):
        super().__init__()
        self.model_name = model_name
        
        # Load pretrained backbone
        if model_name == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            # Modify first conv to accept 1-channel (grayscale) input
            # instead of 3-channel RGB. Average the pretrained weights.
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Average RGB weights → grayscale weight
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )
            
            # Replace classifier head
            in_features = self.backbone.fc.in_features  # 512
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(256, num_classes)
            )
            
        elif model_name == "resnet34":
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(256, num_classes)
            )
            
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze early layers for transfer learning
        if FREEZE_LAYERS and pretrained:
            self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze early layers, only train later blocks + classifier."""
        if "resnet" in self.model_name:
            freeze = True
            for name, param in self.backbone.named_parameters():
                if UNFREEZE_FROM in name:
                    freeze = False
                param.requires_grad = not freeze
            
            # Always train the classifier head
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
                
        elif "efficientnet" in self.model_name:
            # Freeze first 5 of 8 feature blocks
            for i, block in enumerate(self.backbone.features):
                if i < 5:
                    for param in block.parameters():
                        param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning (call after initial training)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 224, 224) mel spectrogram images
        Returns:
            logits: (batch, 2) raw scores for [real, fake]
        """
        return self.backbone(x)
    
    def predict_proba(self, x):
        """Get probability of each class."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def get_embedding(self, x):
        """Get feature embedding before classifier (for visualization)."""
        if "resnet" in self.model_name:
            # Forward through all layers except FC
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(model_name=MODEL_NAME, pretrained=True):
    """Factory function to build the model."""
    model = SpectrogramClassifier(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        pretrained=pretrained
    )
    
    total, trainable = model.count_parameters()
    print(f"Model: {model_name}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters:    {total - trainable:,}")
    
    return model


if __name__ == "__main__":
    # Quick test
    model = build_model()
    dummy = torch.randn(4, 1, 224, 224)
    output = model(dummy)
    probs = model.predict_proba(dummy)
    embed = model.get_embedding(dummy)
    
    print(f"\nInput shape:     {dummy.shape}")
    print(f"Output shape:    {output.shape}")
    print(f"Probs shape:     {probs.shape}")
    print(f"Embedding shape: {embed.shape}")
    print(f"Sample probs:    {probs[0].detach().numpy()}")
