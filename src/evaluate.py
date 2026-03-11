import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
import argparse

# Custom Imports
from src.model import HandGestureModel
from src.dataloader_augmentation import HandGestureDataset

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = "weights"

def get_test_split(root_dir):
    """
    REPLICATES the logic from train.py to ensure we use the exact same Test Set.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root '{root_dir}' not found!")
        
    students = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    students.sort()
    
    n_total = len(students)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    test_students = students[n_train+n_val:]
    
    return test_students

def calculate_iou_batch(pred_mask, true_mask, smooth=1e-6):
    """
    Calculates IoU for each image in the batch individually.
    Returns a Tensor of shape [Batch_Size] containing IoU for each sample.
    """
    # Convert logits to binary (0 or 1)
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    
    # Flatten to [B, -1]
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    true_flat = true_mask.view(true_mask.size(0), -1)
    
    intersection = (pred_flat * true_flat).sum(1)
    union = pred_flat.sum(1) + true_flat.sum(1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def evaluate(model, loader, channels):
    model.eval()
    loop = tqdm(loader, desc="Testing")
    
    total_samples = 0
    correct_class = 0
    total_iou = 0
    
    # NEW: Counter for samples with IoU > 0.5
    iou_above_05_count = 0
    
    with torch.no_grad():
        for batch in loop:
            rgb = batch['rgb'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            masks = batch['mask'].to(DEVICE).float()
            labels = batch['label'].to(DEVICE)

            if channels == 4:
                images = torch.cat([rgb, depth], dim=1)
            else:
                images = rgb

            # Forward Pass
            mask_logits, class_logits = model(images)
            
            # --- METRIC 1: CLASSIFICATION ---
            _, predicted = torch.max(class_logits, 1)
            correct_class += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # --- METRIC 2: SEGMENTATION IoU ---
            # Get IoU for *each* image in the batch
            batch_ious = calculate_iou_batch(mask_logits, masks)
            
            # Sum up total IoU (for average later)
            total_iou += batch_ious.sum().item()
            
            # Count how many images had IoU > 0.5
            iou_above_05_count += (batch_ious > 0.5).sum().item()

    # Final Calculations
    final_acc = 100 * correct_class / total_samples
    final_mean_iou = 100 * total_iou / total_samples
    
    # IoU@0.5 calculation
    iou_at_05 = 100 * iou_above_05_count / total_samples
    
    return final_acc, final_mean_iou, iou_at_05

def main(args):
    print(f"Evaluating on {DEVICE} | Channels: {args.channels}")
    
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent 
    dataset_root = project_root.parent / "rgb_depth"
    
    test_students = get_test_split(str(dataset_root))
    print(f"Test Set: {len(test_students)} students")
    
    test_dataset = HandGestureDataset(str(dataset_root), test_students, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HandGestureModel(in_channels=args.channels, n_classes=10).to(DEVICE)
    weights_path = f"{WEIGHTS_DIR}/best_model_{args.channels}ch.pth"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    acc, mean_iou, iou_05 = evaluate(model, test_loader, args.channels)
    
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    print(f"Classification Accuracy: {acc:.2f}%")
    print(f"Mean IoU (mIoU):         {mean_iou:.2f}%")
    print(f"Detection IoU > 0.5:     {iou_05:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    main(args)