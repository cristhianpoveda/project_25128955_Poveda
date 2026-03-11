import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import argparse

# Custom Imports
from src.model import HandGestureModel
from src.utils import load_depth_map

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = "weights"

class OfficialTestDataset(Dataset):
    """
    Custom Dataset class specifically designed to parse the official test set
    structure: root / GXX_gesture / clipXX / [rgb, annotation, depth_raw, etc.]
    """
    def __init__(self, root_dir):
        self.samples = []
        root = Path(root_dir)
        
        CLASS_MAP = {
            'call': 0, 'dislike': 1, 'like': 2, 'ok': 3, 'one': 4,
            'palm': 5, 'peace': 6, 'rock': 7, 'stop': 8, 'three': 9
        }
        
        # 1. Iterate over Gestures (G01_call, G02_dislike, etc.)
        for gesture_dir in sorted(root.iterdir()):
            if not gesture_dir.is_dir() or not gesture_dir.name.startswith('G'):
                continue
                
            # Extract label (e.g., "G01_call" -> "call")
            gesture_name = gesture_dir.name.split('_', 1)[1].lower()
            label = CLASS_MAP.get(gesture_name, -1)
            
            # 2. Iterate over Clips
            for clip_dir in sorted(gesture_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip'):
                    continue
                    
                rgb_dir = clip_dir / 'rgb'
                mask_dir = clip_dir / 'annotation'
                
                # Try 'depth' first, fallback to 'depth_raw' if needed
                depth_dir = clip_dir / 'depth_raw' if (clip_dir / 'depth').exists() else clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'
                
                if not rgb_dir.exists() or not mask_dir.exists() or not depth_dir.exists():
                    continue
                    
                # 3. Match frames by sorting
                rgb_files = sorted([f for f in rgb_dir.iterdir() if f.is_file()])
                mask_files = sorted([f for f in mask_dir.iterdir() if f.is_file()])
                depth_files = sorted([f for f in depth_dir.iterdir() if f.is_file()])
                
                for r, m, d in zip(rgb_files, mask_files, depth_files):
                    self.samples.append({
                        'rgb': str(r),
                        'mask': str(m),
                        'depth': str(d),
                        'meta': str(meta_path),
                        'label': label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Images
        rgb_img = Image.open(sample['rgb']).convert('RGB')
        mask_img = Image.open(sample['mask']).convert('L')
        depth_map = load_depth_map(sample['depth'], sample['meta'])
        
        # Process Depth
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        max_dist = 1.5
        depth_map = np.clip(depth_map, 0, max_dist)
        depth_map = depth_map / max_dist
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)
        
        # Resize to 256x256 (Nearest for mask/depth to prevent artifacts)
        resize_dims = (256, 256)
        rgb_img = TF.resize(rgb_img, resize_dims)
        mask_img = TF.resize(mask_img, resize_dims, interpolation=TF.InterpolationMode.NEAREST)
        depth_tensor = TF.resize(depth_tensor, resize_dims, interpolation=TF.InterpolationMode.NEAREST)
        
        return {
            'rgb': TF.to_tensor(rgb_img),
            'depth': depth_tensor,
            'mask': TF.to_tensor(mask_img),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

def calculate_iou_batch(pred_mask, true_mask, smooth=1e-6):
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    true_flat = true_mask.view(true_mask.size(0), -1)
    
    intersection = (pred_flat * true_flat).sum(1)
    union = pred_flat.sum(1) + true_flat.sum(1) - intersection
    
    return (intersection + smooth) / (union + smooth)

def evaluate(model, loader, channels):
    model.eval()
    loop = tqdm(loader, desc="Testing")
    
    total_samples = 0
    correct_class = 0
    total_iou = 0
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

            mask_logits, class_logits = model(images)
            
            # Metric 1: Classification
            _, predicted = torch.max(class_logits, 1)
            correct_class += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Metric 2: Segmentation IoU
            batch_ious = calculate_iou_batch(mask_logits, masks)
            total_iou += batch_ious.sum().item()
            iou_above_05_count += (batch_ious > 0.5).sum().item()

    final_acc = 100 * correct_class / total_samples
    final_mean_iou = 100 * total_iou / total_samples
    iou_at_05 = 100 * iou_above_05_count / total_samples
    
    return final_acc, final_mean_iou, iou_at_05

def main(args):
    print(f"Evaluating on {DEVICE} | Channels: {args.channels}")
    
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Test directory not found at {args.test_dir}")
        
    print(f"Loading official test set from: {args.test_dir}")
    
    # Use the new custom dataset
    test_dataset = OfficialTestDataset(root_dir=args.test_dir)
    print(f"Found {len(test_dataset)} total frames across all clips.")
    
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
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the official test dataset directory")
    args = parser.parse_args()
    
    main(args)