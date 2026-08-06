# ================================================================
# KAGGLE TRAINING SCRIPT — P100 GPU OPTIMIZED
# Project: Multi-Task Facial Behaviour and Occlusion Detection
# Model:   ResNet-18 + CBAM Attention (inspired by Rana et al., 2025)
# Dataset: FER-2013
#
# HOW TO USE ON KAGGLE:
#   1. Create new notebook → paste this entire file
#   2. Settings → Accelerator → GPU P100
#   3. Add dataset: search "msambare/fer2013" → Add
#   4. Run All Cells
#   5. Download fer_resnet18_cbam.pth from Output panel
# ================================================================

import os, time, torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ── Constants ───────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES     = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = 7
EPOCHS      = 40          # P100 handles 40 epochs in ~35-40 min
BATCH_SIZE  = 128         # P100 has 16GB VRAM — use larger batches
LR          = 1e-3
WEIGHT_DECAY= 1e-4
IMG_SIZE    = 224

# FER-2013 dataset paths on Kaggle (msambare/fer2013 dataset)
TRAIN_DIR   = "/kaggle/input/fer2013/train"
TEST_DIR    = "/kaggle/input/fer2013/test"
OUTPUT_DIR  = "/kaggle/working"

print(f"Device  : {DEVICE}")
print(f"Epochs  : {EPOCHS}")
print(f"Batch   : {BATCH_SIZE}")
print(f"PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ================================================================
# 1. CBAM ATTENTION MODULE
#    Based on: Rana et al. (2025) — channel + spatial attention
#    Reference: Woo et al. ECCV 2018 (CBAM)
# ================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module — answers 'WHAT' is meaningful.
    Squeezes spatial dimensions, learns inter-channel relationships.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module — answers 'WHERE' features are important.
    Maps pixel-wise importance across spatial dimensions.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x,  dim=1, keepdim=True).values
        attn    = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies channel then spatial attention.
    F'' = Ms(F') ⊗ F'  where  F' = Mc(F) ⊗ F
    (Equations 1 & 2 from Rana et al., 2025)
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ================================================================
# 2. RESNET-18 + CBAM MODEL
# ================================================================

class ResNet18_CBAM(nn.Module):
    """
    ResNet-18 backbone with CBAM attention injected after
    each residual layer block. Pretrained on ImageNet.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.resnet18(weights=weights)

        # Keep early layers as-is
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )

        # Residual layers + CBAM after each
        self.layer1 = base.layer1
        self.cbam1  = CBAM(64)

        self.layer2 = base.layer2
        self.cbam2  = CBAM(128)

        self.layer3 = base.layer3
        self.cbam3  = CBAM(256)

        self.layer4 = base.layer4
        self.cbam4  = CBAM(512)

        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ================================================================
# 3. DATA LOADING WITH AUGMENTATION
# ================================================================

# Training transforms — augmentation improves generalization
# Techniques chosen to simulate occlusion variations
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # Random erasing simulates occlusion patches (Rana et al., 2025)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15),
                             ratio=(0.5, 2.0), value=0),
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\nLoading datasets...")
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

# Weighted sampling to handle FER-2013 class imbalance
# (FER-2013 has 8988 happy vs 546 disgust — highly imbalanced)
class_counts = Counter(train_ds.targets)
sample_weights = [1.0 / class_counts[t] for t in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"Train samples : {len(train_ds)}")
print(f"Val samples   : {len(val_ds)}")
print(f"Class counts  : {dict(sorted(class_counts.items()))}")
print(f"Class mapping : {train_ds.class_to_idx}")


# ================================================================
# 4. TRAINING SETUP
# ================================================================

model     = ResNet18_CBAM(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Count trainable params
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters: {total_params:,}")


# ================================================================
# 5. TRAINING LOOP
# ================================================================

best_val_acc   = 0.0
train_losses   = []
val_losses     = []
val_accuracies = []
start_time     = time.time()

print(f"\n{'='*60}")
print(f"Starting Training — {EPOCHS} epochs on {DEVICE}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss   = 0.0
    epoch_start    = time.time()

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    scheduler.step()

    # ── Validate ──
    model.eval()
    val_loss  = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs   = model(imgs)
            val_loss += criterion(outputs, labels).item()
            preds     = outputs.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = correct / total * 100
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - epoch_start
    marker = " ✓ BEST" if val_acc > best_val_acc else ""

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train: {avg_train_loss:.4f} | "
          f"Val: {avg_val_loss:.4f} | "
          f"Acc: {val_acc:.2f}% | "
          f"{epoch_time:.1f}s{marker}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'val_acc':     val_acc,
            'classes':     CLASSES,
            'architecture':'ResNet18_CBAM',
        }, os.path.join(OUTPUT_DIR, "fer_resnet18_cbam.pth"))

total_time = (time.time() - start_time) / 60
print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"Total Training Time      : {total_time:.1f} minutes")
print(f"Model saved to           : {OUTPUT_DIR}/fer_resnet18_cbam.pth")
print(f"{'='*60}")


# ================================================================
# 6. EVALUATION & PLOTS
# ================================================================

# -- Training Curves --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet-18 + CBAM — Training on FER-2013", fontsize=14)

axes[0].plot(train_losses, label='Train Loss', color='#e74c3c')
axes[0].plot(val_losses,   label='Val Loss',   color='#3498db')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(val_accuracies, color='#2ecc71', linewidth=2)
axes[1].axhline(y=best_val_acc, color='red', linestyle='--', alpha=0.5,
                label=f'Best: {best_val_acc:.2f}%')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Validation Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.show()
print("Saved: training_curves.png")

# -- Load best model for final evaluation --
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "fer_resnet18_cbam.pth"),
                        map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# -- Confusion Matrix --
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5)
plt.title(f'Confusion Matrix — ResNet-18 + CBAM on FER-2013\n'
          f'Best Val Acc: {best_val_acc:.2f}%', fontsize=13)
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

print("\n✅ All done! Download these files from Kaggle Output:")
print("   → fer_resnet18_cbam.pth   (your trained model)")
print("   → training_curves.png     (for your report)")
print("   → confusion_matrix.png    (for your report)")
