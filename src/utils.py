import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import json
from functools import lru_cache

def get_dataset_samples(root_dir, split_students=None):
    """
    Crawls the directory to find valid samples based on existing annotations.
    
    Args:
        root_dir (str): Path to the main dataset folder.
        split_students (list): List of student folder names to include (e.g., for train/val split).
                               If None, includes all students.
    
    Returns:
        list: A list of dicts, where each dict contains paths and metadata for one frame.
    """
    samples = []
    root = Path(root_dir)
    
    # 1. Iterate over Students (e.g., COMP0248_Poveda)
    for student_dir in root.iterdir():
        if not student_dir.is_dir(): continue
        if split_students and student_dir.name not in split_students: continue

        # --- NEW STEP: Enter the nested folder ---
        subdirs = [d for d in student_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Skip if student folder is empty
        if not subdirs: 
            continue
            
        # We assume the exact structure exists, so we take the first folder found
        nested_student_root = subdirs[0]
        # -----------------------------------------

        # 2. Iterate over Gestures (e.g., G01_call) INSIDE the nested root
        for gesture_dir in nested_student_root.iterdir():
            if not gesture_dir.is_dir(): continue
            
            # Extract label from folder name (e.g., "G01_call" -> "call")
            try:
                # Splitting on first underscore to get label
                parts = gesture_dir.name.split('_')
                if len(parts) < 2: continue
                label_str = parts[1] 
            except IndexError:
                continue
            
            # 3. Iterate over Clips (e.g., clip01)
            for clip_dir in gesture_dir.iterdir():
                if not clip_dir.is_dir(): continue

                # Paths to specific subfolders
                annot_dir = clip_dir / 'annotation'
                rgb_dir = clip_dir / 'rgb'
                depth_raw_dir = clip_dir / 'depth_raw'
                meta_path = clip_dir / 'depth_metadata.json'

                # Check if essential folders exist
                if not (annot_dir.exists() and rgb_dir.exists() and meta_path.exists()):
                    continue

                # 4. The Source of Truth: The Annotation Folder
                # We only take frames that have an annotation mask.
                for mask_file in annot_dir.glob('*.png'):
                    frame_name = mask_file.stem # e.g., "frame_005"
                    
                    # Construct matching paths
                    rgb_file = rgb_dir / f"{frame_name}.png"
                    depth_file = depth_raw_dir / f"{frame_name}.npy"
                    
                    if rgb_file.exists() and depth_file.exists():
                        samples.append({
                            # We keep the top-level student dir name for the split_students logic
                            'student': student_dir.name, 
                            'gesture': label_str,
                            'clip': clip_dir.name,
                            'rgb_path': str(rgb_file),
                            'mask_path': str(mask_file),
                            'depth_path': str(depth_file),
                            'meta_path': str(meta_path)
                        })
    return samples

@lru_cache(maxsize=None)
def get_cached_metadata(meta_path_str):
    if meta_path_str is None or str(meta_path_str) == 'None':
        return {}
    with open(meta_path_str, 'r') as f:
        return json.load(f)

def load_depth_map(npy_path, meta_path):
    """
    Loads raw depth (.npy) and applies scaling from cached metadata.
    """
    depth_path_str = str(npy_path)

    # 1. Cargar el array crudo directamente
    depth_raw = np.load(depth_path_str, allow_pickle=True)

    # 2. Obtener Metadata (servido instantáneamente desde la RAM gracias a lru_cache)
    meta_data = {}
    if meta_path is not None:
        meta_data = get_cached_metadata(str(meta_path))

    # 3. Extraer la escala y aplicarla
    # (Ajusta 'scale' si tu JSON usa otra llave, ej: 'depth_scale')
    scale = meta_data.get('scale', 1.0) 
    
    depth_metric = depth_raw.astype(np.float32) * scale
    return depth_metric