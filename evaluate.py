# ================================================================
# evaluate.py — FINAL VERSION
# Evaluates trained model on:
#   1. FER2013 test set
#   2. RAF-DB test set
#   3. Combined test set
#
# Generates report-quality plots for PPT/report:
#   - confusion_matrix.png
#   - per_class_accuracy.png
#   - roc_curves.png
#   - classification_report.txt
#
# RUN: python evaluate.py
# ================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from PIL import Image
from collections import Counter
from model import load_model

# ── Config ────────────────────────────────────────────────────────
CLASSES    = ['angry','disgust','fear','happy','neutral','sad','surprise']
MODEL_PATH = "fer_resnet18_cbam.pth"
FER_TEST   = "fer2013/test"
RAFDB_TEST = "rafdb/test"   # adjust if different path
RESULTS    = "results"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS, exist_ok=True)

# ── RAF-DB label mapper ───────────────────────────────────────────
class RAFDBDataset(Dataset):
    RAFDB_TO_OURS = {1:6, 2:2, 3:1, 4:3, 5:5, 6:0, 7:4}
    def __init__(self, root, transform=None):
        self.samples   = []
        self.targets   = []
        self.transform = transform
        if not os.path.exists(root):
            print(f"  ⚠️  RAF-DB path not found: {root}")
            return
        for folder in sorted(os.listdir(root)):
            fp = os.path.join(root, folder)
            if not os.path.isdir(fp): continue
            try: rafdb_label = int(folder)
            except ValueError: continue
            our_label = self.RAFDB_TO_OURS.get(rafdb_label)
            if our_label is None: continue
            for fname in os.listdir(fp):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(fp, fname), our_label))
                    self.targets.append(our_label)
        print(f"  RAF-DB test loaded: {len(self.samples)} images")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

# ── Transform ─────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Load model ────────────────────────────────────────────────────
print(f"\nDevice : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
model = load_model(MODEL_PATH, DEVICE)

# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate(loader, name):
    """Run inference and return preds, labels, probs."""
    print(f"\nEvaluating on {name}...")
    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            out   = model(imgs.to(DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy()
                              if hasattr(labels, 'numpy')
                              else labels)
    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


def save_results(preds, labels, probs, name, tag):
    """Generate and save all plots for one evaluation run."""
    acc = (preds == labels).mean() * 100
    print(f"  Overall Accuracy: {acc:.2f}%")

    # Classification report
    report = classification_report(labels, preds,
                                   target_names=CLASSES, digits=4)
    print(report)
    with open(os.path.join(RESULTS,
              f"classification_report_{tag}.txt"), "w") as f:
        f.write(f"Dataset        : {name}\n")
        f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
        f.write(report)
    print(f"  ✅ Saved: classification_report_{tag}.txt")

    # Confusion matrix
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix — {name}\n"
                 f"Overall Accuracy: {acc:.2f}%", fontsize=14)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[0], xticklabels=CLASSES,
                yticklabels=CLASSES, linewidths=0.4)
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='RdYlGn',
                ax=axes[1], xticklabels=CLASSES,
                yticklabels=CLASSES, linewidths=0.4,
                vmin=0, vmax=100)
    axes[1].set_title("Normalised (%)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(RESULTS, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: confusion_matrix_{tag}.png")

    # Per-class accuracy
    per_class_acc = cm_norm.diagonal()
    colors = ['#e74c3c','#27ae60','#9b59b6',
               '#f1c40f','#95a5a6','#3498db','#e67e22']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASSES, per_class_acc, color=colors,
                  width=0.6, edgecolor='white')
    ax.axhline(y=acc, color='black', linestyle='--',
               linewidth=1.5, label=f'Overall: {acc:.2f}%')
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Class Accuracy — {name}")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS, f"per_class_accuracy_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: per_class_accuracy_{tag}.png")

    # ROC curves
    y_bin      = label_binarize(labels, classes=list(range(len(CLASSES))))
    colors_roc = ['#e74c3c','#27ae60','#9b59b6',
                   '#f39c12','#7f8c8d','#2980b9','#e67e22']
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (cls, col) in enumerate(zip(CLASSES, colors_roc)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {name}")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(RESULTS, f"roc_curves_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: roc_curves_{tag}.png")

    return acc, per_class_acc


# ================================================================
# RUN EVALUATIONS
# ================================================================

summary = {}

# ── 1. FER2013 Test ───────────────────────────────────────────────
print(f"\n{'='*55}")
print(" EVALUATION 1: FER2013 Test Set")
print(f"{'='*55}")

if os.path.exists(FER_TEST):
    fer_ds     = datasets.ImageFolder(FER_TEST, transform=tf)
    fer_loader = DataLoader(fer_ds, batch_size=64,
                            shuffle=False, num_workers=0)
    print(f"  FER2013 test samples: {len(fer_ds)}")
    preds, labels, probs = evaluate(fer_loader, "FER2013 Test")
    acc, per_cls = save_results(
        preds, labels, probs, "FER2013 Test", "fer2013")
    summary['FER2013'] = {'acc': acc, 'per_cls': per_cls}
else:
    print(f"  ⚠️  FER2013 test not found at: {FER_TEST}")
    print("  Update FER_TEST path in evaluate.py")

# ── 2. RAF-DB Test ────────────────────────────────────────────────
print(f"\n{'='*55}")
print(" EVALUATION 2: RAF-DB Test Set")
print(f"{'='*55}")

rafdb_ds = RAFDBDataset(RAFDB_TEST, transform=tf)
if len(rafdb_ds) > 0:
    rafdb_loader = DataLoader(rafdb_ds, batch_size=64,
                              shuffle=False, num_workers=0)
    preds, labels, probs = evaluate(rafdb_loader, "RAF-DB Test")
    acc, per_cls = save_results(
        preds, labels, probs, "RAF-DB Test", "rafdb")
    summary['RAFDB'] = {'acc': acc, 'per_cls': per_cls}
else:
    print("  ⚠️  RAF-DB test not found — skipping")
    print(f"  Update RAFDB_TEST path in evaluate.py")

# ── 3. Combined Test ──────────────────────────────────────────────
if 'FER2013' in summary and 'RAFDB' in summary:
    print(f"\n{'='*55}")
    print(" EVALUATION 3: Combined Test Set")
    print(f"{'='*55}")

    combined_ds     = ConcatDataset([fer_ds, rafdb_ds])
    combined_loader = DataLoader(combined_ds, batch_size=64,
                                 shuffle=False, num_workers=0)
    print(f"  Combined test samples: {len(combined_ds)}")
    preds, labels, probs = evaluate(combined_loader, "Combined Test")
    acc, per_cls = save_results(
        preds, labels, probs, "Combined Test", "combined")
    summary['Combined'] = {'acc': acc, 'per_cls': per_cls}

# ── Summary comparison plot ───────────────────────────────────────
if len(summary) > 1:
    print(f"\n{'='*55}")
    print(" GENERATING SUMMARY COMPARISON")
    print(f"{'='*55}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Evaluation Summary — ResNet-18 + CBAM\n"
                 "Trained on FER2013 + RAF-DB Combined",
                 fontsize=13, fontweight='bold')

    # Bar chart of overall accuracy
    names = list(summary.keys())
    accs  = [summary[n]['acc'] for n in names]
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71'][:len(names)]
    bars = axes[0].bar(names, accs, color=colors_bar,
                       width=0.5, edgecolor='white')
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{acc:.2f}%', ha='center',
                     va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_ylim(60, 95)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Overall Accuracy by Test Set")
    axes[0].grid(axis='y', alpha=0.3)

    # Per-class comparison
    x = np.arange(len(CLASSES))
    width = 0.25
    for i, (name, color) in enumerate(zip(names, colors_bar)):
        axes[1].bar(x + i*width,
                    summary[name]['per_cls'],
                    width, label=name, color=color,
                    alpha=0.8, edgecolor='white')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(CLASSES, rotation=15)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Per-Class Accuracy by Test Set")
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "evaluation_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: evaluation_summary.png")

# ── Final summary ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print("✅ EVALUATION COMPLETE")
print(f"{'='*55}")
for name, res in summary.items():
    print(f"  {name:12s} : {res['acc']:.2f}%")
print(f"\nAll outputs saved to: ./{RESULTS}/")
print(f"{'='*55}")