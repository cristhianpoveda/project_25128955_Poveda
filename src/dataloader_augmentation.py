import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF # Crucial for manual sync
from .utils import get_dataset_samples, load_depth_map

CLASS_MAP = {
    'call': 0, 'dislike': 1, 'like': 2, 'ok': 3, 'one': 4,
    'palm': 5, 'peace': 6, 'rock': 7, 'stop': 8, 'three': 9
}

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, student_list=None, augment=False):
        """
        Args:
            root_dir (str): Path to dataset.
            student_list (list): List of students for this split.
            augment (bool): If True, applies random rotation, shear, noise, etc.
        """
        self.samples = get_dataset_samples(root_dir, student_list)
        self.augment = augment
        
        # Standard Resize for all inputs to ensure 256x256 output
        self.resize_dims = (256, 256)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # 1. Load Data
        rgb_img = Image.open(sample_info['rgb_path']).convert('RGB')
        mask_img = Image.open(sample_info['mask_path']).convert('L')
        depth_map = load_depth_map(sample_info['depth_path'], sample_info['meta_path'])

        # 2. Depth Pre-processing (Sanitize & Normalize)
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        max_dist = 1.5 
        depth_map = np.clip(depth_map, 0, max_dist)
        depth_map = depth_map / max_dist
        
        # Convert Depth to Tensor [1, H, W] immediately for consistent processing
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)

        # 3. AUGMENTATION (Train Only)
        if self.augment:
            # --- A. Synchronized Geometric Transform (RGB + Mask + Depth) ---
            # Random Affine: Rotation, Scale, Shift, Shear
            # Shear is used instead of Horizontal Flip because dataset is Right-Hand only
            angle = random.uniform(-20, 20)
            translate = (random.uniform(-0.05, 0.05) * rgb_img.width, 
                         random.uniform(-0.05, 0.05) * rgb_img.height)
            scale = random.uniform(0.85, 1.15)
            shear = random.uniform(-10, 10) # Slant the hand

            # Apply same parameters to all 3
            rgb_img = TF.affine(rgb_img, angle, translate, scale, shear)
            mask_img = TF.affine(mask_img, angle, translate, scale, shear)
            depth_tensor = TF.affine(depth_tensor, angle, translate, scale, shear)

            # --- B. Photometric Transform (RGB Only) ---
            # Randomly change brightness/contrast/saturation
            # color_params = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            color_params = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            rgb_img = color_params(rgb_img)

            # --- C. Sensor Noise (Depth Only) ---
            # 50% chance to add noise
            if random.random() > 0.5:
                noise = torch.randn_like(depth_tensor) * 0.02 # 2% Gaussian noise
                depth_tensor = depth_tensor + noise
                
                # Random "Holes" (Simulate RealSense pixel dropout)
                # 1% of pixels drop to zero
                mask_dropout = (torch.rand_like(depth_tensor) > 0.99).float()
                depth_tensor = depth_tensor * (1 - mask_dropout)
                
                # Re-clamp to ensure valid range [0, 1]
                depth_tensor = torch.clamp(depth_tensor, 0, 1)

        # 4. Final Formatting (Resize & ToTensor)
        # We apply resize manually here to ensure everything aligns perfectly
        rgb_img = TF.resize(rgb_img, self.resize_dims)
        mask_img = TF.resize(mask_img, self.resize_dims, interpolation=transforms.InterpolationMode.NEAREST)
        depth_tensor = TF.resize(depth_tensor, self.resize_dims, interpolation=transforms.InterpolationMode.NEAREST)

        # Convert RGB/Mask to Tensors (Depth is already tensor)
        rgb_tensor = TF.to_tensor(rgb_img)
        mask_tensor = TF.to_tensor(mask_img)

        # 5. Label Handling
        raw_label = sample_info['gesture'].strip().lower()
        label = CLASS_MAP.get(raw_label, -1)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }