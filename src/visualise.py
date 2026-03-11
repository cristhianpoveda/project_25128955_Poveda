import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add project root to path so we can import src modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.model import HandGestureModel
from src.dataloader_augmentation import HandGestureDataset
from src.train_validation import get_student_split 

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = "weights/best_model_4ch.pth"
SAVE_PATH = "results/prediction_stack.png"
CLASS_MAP = {
    0: 'call', 1: 'dislike', 2: 'like', 3: 'ok', 4: 'one',
    5: 'palm', 6: 'peace', 7: 'rock', 8: 'stop', 9: 'three'
}

def visualize_stack(num_samples=5):
    print(f"Generating visualization for {num_samples} samples...")
    
    # 1. Setup Data
    # Go up from src/visualize.py to project root, then to rgb_depth
    dataset_root = project_root.parent / "rgb_depth"
    
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return

    # Get Test Split
    _, _, test_students = get_student_split(str(dataset_root))
    
    # Dataset (No Augmentation)
    ds = HandGestureDataset(str(dataset_root), test_students, augment=False)
    # Shuffle is TRUE to get random different samples
    loader = DataLoader(ds, batch_size=1, shuffle=True) 

    # 2. Load Model
    model = HandGestureModel(in_channels=4, n_classes=10).to(DEVICE)
    if not os.path.exists(WEIGHTS):
        print(f"Error: Weights file not found at {WEIGHTS}")
        return
        
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()

    # 3. Setup Plot Grid (Rows = num_samples, Cols = 3)
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    iterator = iter(loader)

    with torch.no_grad():
        for i in range(num_samples):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            rgb = batch['rgb'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            true_mask = batch['mask'].squeeze().cpu().numpy()
            true_label = batch['label'].item()

            # Forward Pass
            images = torch.cat([rgb, depth], dim=1)
            mask_pred, class_pred = model(images)
            
            # Post-Process
            pred_mask = (torch.sigmoid(mask_pred) > 0.5).float().squeeze().cpu().numpy()
            pred_label_idx = torch.argmax(class_pred, dim=1).item()
            pred_name = CLASS_MAP.get(pred_label_idx, "Unknown")
            true_name = CLASS_MAP.get(true_label, "Unknown")

            # --- PLOTTING ---
            
            # Column 1: RGB Input
            rgb_show = rgb.squeeze().permute(1, 2, 0).cpu().numpy()
            axs[i, 0].imshow(rgb_show)
            axs[i, 0].set_title(f"Sample {i+1}: {true_name.upper()}", fontsize=10)
            axs[i, 0].axis('off')

            # Column 2: Ground Truth Mask
            axs[i, 1].imshow(true_mask, cmap='gray')
            axs[i, 1].set_title("Ground Truth", fontsize=10)
            axs[i, 1].axis('off')

            # Column 3: Predicted Mask + Class
            axs[i, 2].imshow(pred_mask, cmap='gray')
            
            # Color code the title: Green if correct, Red if wrong
            color = 'green' if true_label == pred_label_idx else 'red'
            axs[i, 2].set_title(f"Pred: {pred_name.upper()}", color=color, fontweight='bold', fontsize=10)
            axs[i, 2].axis('off')

    # Save
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"Saved stack visualization to {SAVE_PATH}")

if __name__ == "__main__":
    visualize_stack()