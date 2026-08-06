# ================================================================
# model.py
# ResNet-18 + CBAM Attention Model Definition
# Project: Multi-Task Facial Behaviour and Occlusion Detection
#
# Used by: realtime_system.py, evaluate.py
#
# Architecture inspired by:
#   Rana et al. (2025) — "Facial Emotion Recognition in Occluded
#   Images Using Attention-Based CNN" (JITI, 3(2), 40-55)
#
# FIX: Dropout changed 0.4 → 0.45 to match training checkpoint
# ================================================================

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """
    Channel Attention — determines WHAT features are meaningful.
    Attribute named self.fc to match training checkpoint keys.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_w = self.fc(self.avg_pool(x))
        max_w = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg_w + max_w)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention — determines WHERE important features are.
    Attribute named self.conv to match training checkpoint keys.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x,  dim=1, keepdim=True).values
        concat  = torch.cat([avg_map, max_map], dim=1)
        scale   = self.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    F'  = Mc(F)  ⊗ F
    F'' = Ms(F') ⊗ F'
    Attributes named self.channel and self.spatial to match checkpoint.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResNet18_CBAM(nn.Module):
    """
    ResNet-18 with CBAM after each residual block.

    Architecture:
        Input (3×224×224)
        → Stem (conv1 + bn + relu + maxpool)
        → Layer1 (64ch)  → CBAM
        → Layer2 (128ch) → CBAM
        → Layer3 (256ch) → CBAM
        → Layer4 (512ch) → CBAM
        → GlobalAvgPool
        → Dropout(0.45)   ← matches training checkpoint
        → FC(512 → 7)
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        base    = models.resnet18(weights=weights)

        self.stem   = nn.Sequential(base.conv1, base.bn1,
                                    base.relu,  base.maxpool)
        self.layer1 = base.layer1;  self.cbam1 = CBAM(64)
        self.layer2 = base.layer2;  self.cbam2 = CBAM(128)
        self.layer3 = base.layer3;  self.cbam3 = CBAM(256)
        self.layer4 = base.layer4;  self.cbam4 = CBAM(512)
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=0.45)   # ✅ fixed: matches training script
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout(x)
        return self.fc(x)


def load_model(checkpoint_path: str,
               device: torch.device,
               num_classes: int = 7) -> nn.Module:
    """
    Load trained ResNet18_CBAM from checkpoint.
    Handles both plain state_dict and wrapped checkpoint dict.
    """
    model      = ResNet18_CBAM(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ Loaded checkpoint — epoch {checkpoint.get('epoch','?')} "
              f"| val acc: {checkpoint.get('val_acc', 0):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights.")

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cpu")
    m      = ResNet18_CBAM(num_classes=7, pretrained=False)
    out    = m(torch.randn(2, 3, 224, 224))
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {sum(p.numel() for p in m.parameters()):,}")
    print("model.py — OK")
