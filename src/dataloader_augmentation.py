import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF # Needed for manual sync
from .utils import get_dataset_samples, load_depth_map

CLASS_MAP = {
    'call': 0, 'dislike': 1, 'like': 2, 'ok': 3, 'one': 4,
    'palm': 5, 'peace': 6, 'rock': 7, 'stop': 8, 'three': 9
}

# Dataset class to handle data loading for the model

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, student_list=None, augment=False):
        """
        DATASET DATA LOADER
        Args:
            root_dir (str): Path to dataset.
            student_list (list): List of students for this split.
            augment (bool): If True, applies random rotation, shear, noise, etc.
        """
        self.samples = get_dataset_samples(root_dir, student_list)
        self.augment = augment
        
        self.resize_dims = (256, 256) # Resize to a fixed input size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        sample_info = self.samples[idx]

        # Load Data
        rgb_img = Image.open(sample_info['rgb_path']).convert('RGB')
        mask_img = Image.open(sample_info['mask_path']).convert('L')
        depth_map = load_depth_map(sample_info['depth_path'], sample_info['meta_path'])

        # Depth normalisation
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        max_dist = 1.5 
        depth_map = np.clip(depth_map, 0, max_dist)
        depth_map = depth_map / max_dist
        
        # Depth to Tensor [1, H, W]
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)

        if self.augment:
            """
            DATA ON-THE-FLY AUGMENTATION
            Only for training
            """
            
            # Random affine transformation
            angle = random.uniform(-20, 20)
            translate = (random.uniform(-0.05, 0.05) * rgb_img.width, 
                         random.uniform(-0.05, 0.05) * rgb_img.height)
            scale = random.uniform(0.85, 1.15)
            shear = random.uniform(-10, 10) # Slant the hand

            # Apply to all channels
            rgb_img = TF.affine(rgb_img, angle, translate, scale, shear)
            mask_img = TF.affine(mask_img, angle, translate, scale, shear)
            depth_tensor = TF.affine(depth_tensor, angle, translate, scale, shear)

            # Photometric transform: brightness/contrast/saturation
            color_params = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            rgb_img = color_params(rgb_img)

            # Depth noise
            if random.random() > 0.5: # Probability of adding noise
                noise = torch.randn_like(depth_tensor) * 0.02 # 2% Gaussian noise
                depth_tensor = depth_tensor + noise
                
                # Random dropout
                mask_dropout = (torch.rand_like(depth_tensor) > 0.99).float()
                depth_tensor = depth_tensor * (1 - mask_dropout) # 1% pixel dropout
                depth_tensor = torch.clamp(depth_tensor, 0, 1) # Clamp within limits

        # Tensor resize for the model(All channels consistent)
        rgb_img = TF.resize(rgb_img, self.resize_dims)
        mask_img = TF.resize(mask_img, self.resize_dims, interpolation=transforms.InterpolationMode.NEAREST)
        depth_tensor = TF.resize(depth_tensor, self.resize_dims, interpolation=transforms.InterpolationMode.NEAREST)

        # RGB + mask to tensors
        rgb_tensor = TF.to_tensor(rgb_img)
        mask_tensor = TF.to_tensor(mask_img)

        # label to tensor 
        raw_label = sample_info['gesture'].strip().lower()
        label = CLASS_MAP.get(raw_label, -1)
        
        # Dictionary with the full loaded sample
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }