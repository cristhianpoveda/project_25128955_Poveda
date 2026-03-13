import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import json
from functools import lru_cache

def get_dataset_samples(root_dir, split_students=None):
    """
    CREATES DIRECTORY PATHS TO ACCESS THE ANNOTATED SAMPLES
    
    Args:
        root_dir (str): Path to the main dataset folder.
        split_students (list): List of student folder names to search
    
    Returns:
        list: A list of dicts with the path a metadata of a sample
    """
    samples = []
    root = Path(root_dir)
    
    # Iterate over students
    for student_dir in root.iterdir():
        if not student_dir.is_dir(): continue
        if split_students and student_dir.name not in split_students: continue

        subdirs = [d for d in student_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not subdirs: #skip empty folder
            continue
        
        # Iterate over the gesture folders
        nested_student_root = subdirs[0]

        for gesture_dir in nested_student_root.iterdir():
            if not gesture_dir.is_dir(): continue
            
            # Get the label from the directory name
            try:
                
                parts = gesture_dir.name.split('_') # Splitting character
                if len(parts) < 2: continue
                label_str = parts[1] 
            except IndexError:
                continue
            
            # Iterate over clips
            for clip_dir in gesture_dir.iterdir():
                if not clip_dir.is_dir(): continue

                # Paths to specific subfolders
                annot_dir = clip_dir / 'annotation'
                rgb_dir = clip_dir / 'rgb'
                depth_raw_dir = clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'

                # Check if the required folders exist
                if not (annot_dir.exists() and rgb_dir.exists() and meta_path.exists()):
                    continue

                # Ground truth
                for mask_file in annot_dir.glob('*.png'):
                    frame_name = mask_file.stem
                    
                    # Construct path
                    rgb_file = rgb_dir / f"{frame_name}.png"
                    depth_file = depth_raw_dir / f"{frame_name}.npy"
                    
                    if rgb_file.exists() and depth_file.exists():
                        samples.append({
                            
                            'student': student_dir.name, # Used to slipt stuents only
                            'gesture': label_str,
                            'clip': clip_dir.name,
                            'rgb_path': str(rgb_file),
                            'mask_path': str(mask_file),
                            'depth_path': str(depth_file),
                            'meta_path': str(meta_path)
                        })
    return samples

"""
SEND SAMPLE METADATA TO GPU CHACHE
Avoid metadata repeated loading
"""
@lru_cache(maxsize=None)
def get_cached_metadata(meta_path_str):
    if meta_path_str is None or str(meta_path_str) == 'None':
        return {}
    with open(meta_path_str, 'r') as f:
        return json.load(f)

def load_depth_map(npy_path, meta_path):
    """
    Loads raw depth and scales from cached metadata.
    Args:
        npy_path: raw depth path
        meta_path: metadata path
    
    Returns:
        depth_metric: scaled depth map
    """
    depth_path_str = str(npy_path)

    depth_raw = np.load(depth_path_str, allow_pickle=True)

    meta_data = {}
    if meta_path is not None:
        meta_data = get_cached_metadata(str(meta_path))

    scale = meta_data.get('scale', 1.0) 
    
    depth_metric = depth_raw.astype(np.float32) * scale
    return depth_metric