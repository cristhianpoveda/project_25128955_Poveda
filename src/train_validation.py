import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

from src.model import HandGestureModel # Model architecture
from src.dataloader_augmentation import HandGestureDataset # Custom dataloader for the dataset

class FocalLoss(nn.Module):
    """
    FOCAL LOSS FUNCTION FOR CLASSIFICATION IMPROVEMENT
    Higher penalisation on misclassified samples
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        PyTorch Focal Loss implementation
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the custom forward pass
        Args:
            inputs: input tensor
            targets: annotated ground truth
        """

        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probability of the correct class
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss # Focal weighting: (1 - pt)^gamma
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_student_split(root_dir):
    """
    Splits students into 75% Train, 25% Val.
    Supports train/val/test splitting as well
    Args:
        root_dir: dataset root directory

    Returns:
        train_students: List of students for the training set
        val_students: List of students for the validation set
        test_students: List of students for the test set
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root '{root_dir}' not found!")
        
    students = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    students.sort()

    # Split proportions     
    n_total = len(students)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.25)
    
    # Data partitioning
    train_students = students[:n_train]
    val_students = students[n_train:n_train+n_val]
    test_students = students[n_train+n_val:]
    
    return train_students, val_students, test_students

def save_plots(train_hist, val_hist, train_acc, val_acc, path):
    """
    Saves Loss and Accuracy curves to a file.
    Args:
        train_hist: train loss values
        val_hist: val loss values
        train_acc: train accuracy values
        val_acc: val accuracy values
    """
    plt.figure(figsize=(10, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def run_epoch(model, loader, optimizer, scaler, seg_criterion, cls_criterion, channels, is_train=True):
    """
    Unified function for both Training and Validation loops.
    Args:
        model: model architecture
        loader: dataloader
        optimiser: training optimiser (Adam)
        scaler: model scaler
        seg_criterion: segmentation loss
        cls_criterion: classification loss
        channels: number of input channels
        is_train: true if it is a training epoch

    Returns:
        avg_loss: epoch loss value
        avg_acc: epoch accuracy value
    """
    if is_train:
        model.train()
    else:
        model.eval()
    
    loop = tqdm(loader, leave=True, desc="Train" if is_train else "Val") # Data loading
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loop:
        # Tensors to GPU
        rgb = batch['rgb'].to(DEVICE, non_blocking=True)
        depth = batch['depth'].to(DEVICE, non_blocking=True)
        masks = batch['mask'].to(DEVICE, non_blocking=True).float()
        labels = batch['label'].to(DEVICE, non_blocking=True)

        if channels == 4:
            images = torch.cat([rgb, depth], dim=1) # Dpeth + RGB concatenation
        else:
            images = rgb

        # Safety Check for NaNs
        if torch.isnan(images).any():
            continue

        # Update gradients only if training
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast('cuda'): # CUDA adaptive max precision (16-bit)
                mask_preds, class_preds = model(images)
                
                loss_seg = seg_criterion(mask_preds, masks)
                loss_cls = cls_criterion(class_preds, labels)
                loss = loss_seg + loss_cls # Total loss (unweighted)

            # Accuracy Calculation
            _, predicted = torch.max(class_preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Backpropagation (only training)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        
        acc = 100 * correct / total if total > 0 else 0
        loop.set_description(f"{'Train' if is_train else 'Val'} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    avg_acc = 100 * correct / total if total > 0 else 0
    return avg_loss, avg_acc

def main(args):
    print(f"Device: {DEVICE} | Channels: {args.channels} | Batch: {args.batch_size}")
    
    # Dataset root directory
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent 
    dataset_root = project_root.parent / "rgb_depth"
    
    # Data splitting
    train_s, val_s, test_s = get_student_split(str(dataset_root))
    print(f"Data Split: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)} students")
    
    # Save student names used for test (If test is available)
    with open(f"{RESULTS_DIR}/test_students.txt", "w") as f:
        for s in test_s:
            f.write(s + "\n")
    print(f"Test split saved to {RESULTS_DIR}/test_students.txt")
    
    # Load as Pytorch train dataset
    train_set = HandGestureDataset(str(dataset_root), train_s, augment=True)
    
    # Load as Pytorch validation dataset
    val_set = HandGestureDataset(str(dataset_root), val_s, augment=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Setup
    model = HandGestureModel(in_channels=args.channels, n_classes=10).to(DEVICE)

    # If want to pre load model weights

    # weights_path = f"{WEIGHTS_DIR}/reduced_overfitting/best_model_{args.channels}ch.pth"
    # if os.path.exists(weights_path):
    #     print(f"Loading existing weights from {weights_path} for fine-tuning!")
    #     model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    # else:
    #     print("No existing weights found. Training from scratch!")
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Model optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Loss functions
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_cls = FocalLoss(gamma=2.)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # Training loop
    try:
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            
            # Train Step
            t_loss, t_acc = run_epoch(model, train_loader, optimizer, scaler, criterion_seg, criterion_cls, args.channels, is_train=True)
            
            # Validation Step
            v_loss, v_acc = run_epoch(model, val_loader, None, None, criterion_seg, criterion_cls, args.channels, is_train=False)
            
            # Update History
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)
            
            print(f"Summary: Train Loss {t_loss:.4f} / Acc {t_acc:.1f}% || Val Loss {v_loss:.4f} / Acc {v_acc:.1f}%")

            # Save plot after every epoch
            save_plots(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], f"{RESULTS_DIR}/training_curves.png")

            # Save Best Model
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), f"{WEIGHTS_DIR}/finetuned_model_{args.channels}ch.pth")
                print(f"Best Model Saved! (Acc: {best_val_acc:.1f}%)")

    except KeyboardInterrupt:
        print("\n\nTraining Interrupted by User!")
        save_plots(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], f"{RESULTS_DIR}/training_curves_interrupted.png")
        print(f"Exiting. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    main(args)