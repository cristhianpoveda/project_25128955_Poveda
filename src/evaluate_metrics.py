import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import torch.backends.cudnn as cudnn

from src.model import HandGestureModel # Model architecture 
from src.dataloader_augmentation import HandGestureDataset # Custom dataloader for the dataset
from src.utils import load_depth_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = "weights/fine"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_MAP = {
    'call': 0, 'dislike': 1, 'like': 2, 'ok': 3, 'one': 4,
    'palm': 5, 'peace': 6, 'rock': 7, 'stop': 8, 'three': 9
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
CLASSES = [IDX_TO_CLASS[i] for i in range(10)]

def get_student_split(root_dir):
    """
    Replicates train_validation.py logic to get exact Train and Val students
    Args:
        root_dir: Dataset root directory

    Returns:
        train_students: List of students for the training set
        val_students: List of students for the validation set
        test_students: List of students for the test set
    """
    students = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    students.sort()
    n_total = len(students)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_students = students[:n_train]
    val_students = students[n_train:n_train+n_val]
    test_students = students[n_train+n_val:]
    return train_students, val_students, test_students

class OfficialTestDataset(Dataset):
    """
    Custom Dataset for the test set structure

    """
    def __init__(self, root_dir):
        self.samples = []
        root = Path(root_dir)
        for gesture_dir in sorted(root.iterdir()):
            if not gesture_dir.is_dir() or not gesture_dir.name.startswith('G'): continue
            label = CLASS_MAP.get(gesture_dir.name.split('_', 1)[1].lower(), -1)
            for clip_dir in sorted(gesture_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip'): continue
                
                rgb_dir, mask_dir = clip_dir / 'rgb', clip_dir / 'annotation'
                depth_dir = clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'
                
                if not rgb_dir.exists() or not mask_dir.exists() or not depth_dir.exists(): continue
                
                r_files, m_files, d_files = sorted(rgb_dir.iterdir()), sorted(mask_dir.iterdir()), sorted(depth_dir.iterdir())
                for r, m, d in zip(r_files, m_files, d_files):
                    self.samples.append({'rgb': str(r), 'mask': str(m), 'depth': str(d), 'meta': str(meta_path), 'label': label})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        rgb_img = Image.open(sample['rgb']).convert('RGB')
        mask_img = Image.open(sample['mask']).convert('L')
        depth_map = load_depth_map(sample['depth'], sample['meta'])
        
        depth_map = np.clip(np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0), 0, 1.5) / 1.5
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)
        
        resize_dims = (256, 256)
        return {
            'rgb': TF.to_tensor(TF.resize(rgb_img, resize_dims)),
            'depth': TF.resize(depth_tensor, resize_dims, interpolation=TF.InterpolationMode.NEAREST),
            'mask': TF.to_tensor(TF.resize(mask_img, resize_dims, interpolation=TF.InterpolationMode.NEAREST)),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

# Segmentation and detection utils
def keep_largest_connected_component(binary_masks):
    """
    Filters a batch of binary masks to keep only the largest blob per image
    Args:
        binary_masks: Predicted  segmentation mask
    """
    cleaned_masks = []
    masks_cpu = binary_masks.cpu()
    
    for mask in masks_cpu:
        mask_np = mask.squeeze().numpy().astype(np.uint8)
        
        # Calculate Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            cleaned_masks.append(torch.zeros_like(mask))
            continue
            
        # Find the largest component (ignores ID 0: background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create a new mask from the largest label
        cleaned_mask_np = (labels == largest_label).astype(np.float32)
        cleaned_masks.append(torch.from_numpy(cleaned_mask_np).unsqueeze(0))
        
    # Push new mask back to GPU
    return torch.stack(cleaned_masks).to(binary_masks.device)

def extract_bboxes_from_masks(masks):
    """
    COMPUTE MINIMUM BOUNDING BOX OF A BINARY MASK
    Args:
        masks: predicted mask
        
    Returns:
        bounding_boxes
    """
    boxes = []
    masks_cpu = masks.cpu() 
    
    for mask in masks_cpu:
        pos = torch.where(mask.squeeze() > 0)
        if pos[0].numel() == 0:
            boxes.append(torch.tensor([0, 0, 0, 0]))
        else:
            boxes.append(torch.tensor([torch.min(pos[1]), torch.min(pos[0]), torch.max(pos[1]), torch.max(pos[0])]))
            
    # Push the final batch of boxes back to the GPU at once
    return torch.stack(boxes).float().to(masks.device)

def calculate_bbox_iou(pred_boxes, true_boxes):
    """
    Calculates IoU between predicted and ground truth bounding boxes
    Args:
        pred_boxes: predicted bounding boxes
        true_boxes: ground truth bounding boxes
        """
    x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    union_area = pred_area + true_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # Handle completely empty boundin boxes safely
    empty_mask = (pred_area == 0) & (true_area == 0)
    iou[empty_mask] = 0.0 
    return iou

def calculate_mask_metrics(pred_binary, true_mask, smooth=1e-6):
    """
    Calculates Mask IoU and Dice Coefficient per batch
    Args:
        pred_binary: predicted mask
        true_mask: ground truth mask
        smooth: mask smoothing factor

    Returns:
        iou: Intersection over union
        dice: Dice score
        pred_binary: predicted mask
    """
    pred_flat, true_flat = pred_binary.view(pred_binary.size(0), -1), true_mask.view(true_mask.size(0), -1)
    
    intersection = (pred_flat * true_flat).sum(1)
    union = pred_flat.sum(1) + true_flat.sum(1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred_flat.sum(1) + true_flat.sum(1) + smooth)
    return iou, dice, pred_binary

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """
    PLOT CONFUSION MATRIX
    
    Args:
        y_true: ground truth classification lables
        y_pred: predicted classification lables
        """
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 16})
    plt.title(f'Confusion Matrix - {dataset_name.upper()}', fontsize=22, fontweight='bold')
    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cm_{dataset_name}.png")
    plt.close()

def plot_qualitative_overlays(rgb_batch, true_masks, pred_masks, true_boxes, pred_boxes, true_labels, pred_labels, dataset_name, num_samples=4):
    """
    Plots RGB image with True and Predicted Masks & Bounding Boxes overlaid
    Args:
        rgb_batch: images batch
        true_masks: ground truth masks
        pred_masks: predicted masks
        pred_boxes: predicted bounding boxes
        true_boxes: ground truth bounding boxes
        y_true: ground truth classification lables
        y_pred: predicted classification lables
        dataset_name: Name of the dataset
        num_samples: number of rows to stack with sample tests
        """
    num_samples = min(num_samples, rgb_batch.size(0))
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        img = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
        t_mask, p_mask = true_masks[i].squeeze().cpu().numpy(), pred_masks[i].squeeze().cpu().numpy()
        t_box, p_box = true_boxes[i].cpu().numpy(), pred_boxes[i].cpu().numpy()
        
        # RGB Image
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f"Input ({dataset_name})", fontsize=18, fontweight='bold')
        axs[i, 0].axis('off')
        
        # Ground truth overlay
        axs[i, 1].imshow(img)
        axs[i, 1].imshow(t_mask, cmap='Greens', alpha=0.4)
        if np.sum(t_box) > 0:
            rect = patches.Rectangle((t_box[0], t_box[1]), t_box[2]-t_box[0], t_box[3]-t_box[1], linewidth=3, edgecolor='g', facecolor='none')
            axs[i, 1].add_patch(rect)
        axs[i, 1].set_title(f"True: {IDX_TO_CLASS[true_labels[i].item()].upper()}", fontsize=18, fontweight='bold')
        axs[i, 1].axis('off')

        # Prediction overlay
        color = 'Greens' if true_labels[i] == pred_labels[i] else 'Reds'
        box_color = 'g' if true_labels[i] == pred_labels[i] else 'r'
        
        axs[i, 2].imshow(img)
        axs[i, 2].imshow(p_mask, cmap=color, alpha=0.4)
        if np.sum(p_box) > 0:
            rect = patches.Rectangle((p_box[0], p_box[1]), p_box[2]-p_box[0], p_box[3]-p_box[1], linewidth=3, edgecolor=box_color, facecolor='none')
            axs[i, 2].add_patch(rect)
        axs[i, 2].set_title(f"Pred: {IDX_TO_CLASS[pred_labels[i].item()].upper()}", fontsize=18, fontweight='bold')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/qualitative_{dataset_name}.png")
    plt.close()

def evaluate_dataset(model, loader, channels, dataset_name, connected_components):
    model.eval()
    loop = tqdm(loader, desc=f"Evaluating {dataset_name.upper()}")
    
    # Classification Trackers
    all_true_labels = []
    all_pred_labels = []
    
    # Segmentation & Detection Trackers
    total_mask_iou = 0.0
    total_dice = 0.0
    total_bbox_iou = 0.0
    bbox_acc_05_count = 0
    total_samples = 0
    
    success_bucket = []
    cls_fail_bucket = []
    seg_fail_bucket = []

    with torch.no_grad():
        for i, batch in enumerate(loop):
            rgb, depth = batch['rgb'].to(DEVICE), batch['depth'].to(DEVICE)
            masks, labels = batch['mask'].to(DEVICE).float(), batch['label'].to(DEVICE)
            images = torch.cat([rgb, depth], dim=1) if channels == 4 else rgb

            # Forward pass
            with torch.autocast(device_type=DEVICE):
                mask_logits, class_logits = model(images)
            
            # Classification
            _, predicted_classes = torch.max(class_logits, 1)
            all_true_labels.extend(labels.cpu().tolist())
            all_pred_labels.extend(predicted_classes.cpu().tolist())
            total_samples += labels.size(0)

            raw_pred_binary = (torch.sigmoid(mask_logits) > 0.5).float()
            
            if connected_components:
                
                clean_pred_binary = keep_largest_connected_component(raw_pred_binary)

            else:

                clean_pred_binary = raw_pred_binary

            # Segmentation
            batch_mask_iou, batch_dice, pred_masks_binary = calculate_mask_metrics(clean_pred_binary, masks)
            total_mask_iou += batch_mask_iou.sum().item()
            total_dice += batch_dice.sum().item()

            # Detection (Bboxes)
            true_bboxes = extract_bboxes_from_masks(masks)
            pred_bboxes = extract_bboxes_from_masks(clean_pred_binary)
            
            batch_bbox_iou = calculate_bbox_iou(pred_bboxes, true_bboxes)
            total_bbox_iou += batch_bbox_iou.sum().item()
            bbox_acc_05_count += (batch_bbox_iou > 0.5).sum().item()

            for b_idx in range(labels.size(0)):
                lbl_true = labels[b_idx].item()
                lbl_pred = predicted_classes[b_idx].item()
                iou = batch_bbox_iou[b_idx].item()

                # Data loading

                sample_dict = {
                    'rgb': rgb[b_idx].cpu(),
                    'true_masks': masks[b_idx].cpu(),
                    'pred_masks': pred_masks_binary[b_idx].cpu(),
                    'true_boxes': true_bboxes[b_idx].cpu(),
                    'pred_boxes': pred_bboxes[b_idx].cpu(),
                    'true_labels': labels[b_idx].cpu(),
                    'pred_labels': predicted_classes[b_idx].cpu()
                }

                # Perfect Predictions (Need 2)
                if lbl_true == lbl_pred and iou > 0.85 and len(success_bucket) < 2:
                    # Random gesture !=
                    if lbl_true not in [s['true_labels'].item() for s in success_bucket]:
                        success_bucket.append(sample_dict)

                # Classification Failure
                elif lbl_true != lbl_pred and len(cls_fail_bucket) < 1:
                    cls_fail_bucket.append(sample_dict)

                # Segmentation Failure
                elif lbl_true == lbl_pred and iou < 0.4 and len(seg_fail_bucket) < 1:
                    seg_fail_bucket.append(sample_dict)

    final_samples = success_bucket + cls_fail_bucket + seg_fail_bucket
    
    # If perfect samples are chosen
    while len(final_samples) < 4 and len(success_bucket) > 0:
        final_samples.append(success_bucket[0])

    if len(final_samples) > 0:
        plot_qualitative_overlays(
            torch.stack([s['rgb'] for s in final_samples]),
            torch.stack([s['true_masks'] for s in final_samples]),
            torch.stack([s['pred_masks'] for s in final_samples]),
            torch.stack([s['true_boxes'] for s in final_samples]),
            torch.stack([s['pred_boxes'] for s in final_samples]),
            torch.stack([s['true_labels'] for s in final_samples]),
            torch.stack([s['pred_labels'] for s in final_samples]),
            dataset_name,
            num_samples=len(final_samples)
        )

    # Final metrics
    metrics = {}
    
    # Classification
    all_true_labels, all_pred_labels = np.array(all_true_labels), np.array(all_pred_labels)
    metrics['top1_acc'] = 100 * np.mean(all_true_labels == all_pred_labels)
    metrics['macro_f1'] = 100 * f1_score(all_true_labels, all_pred_labels, average='macro')
    plot_confusion_matrix(all_true_labels, all_pred_labels, dataset_name)
    
    # Segmentation
    metrics['mask_miou'] = 100 * total_mask_iou / total_samples
    metrics['dice'] = 100 * total_dice / total_samples
    
    # Detection
    metrics['bbox_miou'] = 100 * total_bbox_iou / total_samples
    metrics['bbox_acc_05'] = 100 * bbox_acc_05_count / total_samples

    # Print Report
    print(f"\n[{dataset_name.upper()} SET RESULTS]")
    print(f"Classification -> Acc: {metrics['top1_acc']:.2f}% | Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"Segmentation   -> mIoU: {metrics['mask_miou']:.2f}% | Dice: {metrics['dice']:.2f}%")
    print(f"Detection      -> Bbox mIoU: {metrics['bbox_miou']:.2f}% | Acc@0.5 IoU: {metrics['bbox_acc_05']:.2f}%\n")
    
    return metrics

def save_metrics_to_file(dataset_name, metrics):
    """
    Appends the numerical metrics to a text file in the results directory
    Args::
        dataset_name: name of the dataset
        metrics: performance metrics
    """
    filepath = f"{RESULTS_DIR}/metrics.txt"
    
    with open(filepath, "a") as f:
        f.write(f"[{dataset_name.upper()} SET RESULTS]\n")
        f.write(f"Classification -> Acc: {metrics['top1_acc']:.2f}% | Macro F1: {metrics['macro_f1']:.2f}%\n")
        f.write(f"Segmentation   -> mIoU: {metrics['mask_miou']:.2f}% | Dice: {metrics['dice']:.2f}%\n")
        f.write(f"Detection      -> Bbox mIoU: {metrics['bbox_miou']:.2f}% | Acc@0.5 IoU: {metrics['bbox_acc_05']:.2f}%\n")
        f.write("-" * 50 + "\n\n")

def main(args):
    cudnn.benchmark = True
    print(f"Full Independent Evaluation on {DEVICE}")
    
    #Save to metrics file
    metrics_path = f"{RESULTS_DIR}/metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("=== FINAL MODEL EVALUATION METRICS ===\n\n")
    
    # Setup paths
    current_file_path = Path(__file__).resolve()
    dataset_root = current_file_path.parent.parent.parent / "rgb_depth"
    
    # Model loading
    model = HandGestureModel(in_channels=args.channels, n_classes=10).to(DEVICE)
    weights_path = f"{WEIGHTS_DIR}/finetuned_model_{args.channels}ch.pth"
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # Evaluation
    if dataset_root.exists():
        train_students, val_students, test_students = get_student_split(str(dataset_root))
        
        # Val Set
        val_dataset = HandGestureDataset(str(dataset_root), val_students, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)
        val_metrics = evaluate_dataset(model, val_loader, args.channels, "val", connected_components=args.connected_components)
        save_metrics_to_file("val", val_metrics) 
        
        # Train Set
        train_dataset = HandGestureDataset(str(dataset_root), train_students, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        train_metrics = evaluate_dataset(model, train_loader, args.channels, "train", connected_components=args.connected_components)
        save_metrics_to_file("train", train_metrics) 

        # Internal Test Set
        test_dataset = HandGestureDataset(str(dataset_root), test_students, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)
        test_internal_metrics = evaluate_dataset(model, test_loader, args.channels, "test_internal", connected_components=args.connected_components)
        save_metrics_to_file("test_internal", test_internal_metrics) 

    # Evaluate on the official test
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = OfficialTestDataset(root_dir=args.test_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)
        test_official_metrics = evaluate_dataset(model, test_loader, args.channels, "test_official", connected_components=args.connected_components)
        save_metrics_to_file("test_official", test_official_metrics) 
    else:
        print("Skipping Official Test Set (Provide valid --test_dir)")

    print(f"\nAll charts, matrices, and metrics saved to '{RESULTS_DIR}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_dir', type=str, required=False, help="Path to the official test dataset directory")
    parser.add_argument('--connected_components', type=bool, required=False, default=False, help="Perform connected comopnents post processing")
    args = parser.parse_args()
    
    main(args)
