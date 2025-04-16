"""
Generate pseudo masks from classifier models using CAM methods
"""
import os
import torch
import json
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
from tqdm import tqdm
from PIL import Image
import data
from models.classifier import create_classifier
from models.cam import GradCAM, create_cam_model, GradCAMForMask
from utils.visualization import visualize_cam

logger = logging.getLogger(__name__)

def generate_masks(config, method='gradcam', classifier_path=None, output_dir=None,
                  visualize=False, threshold=0.4, args=None):
    """
    Generate pseudo masks for all images using CAM
    
    Args:
        config_path: Path to configuration file
        method: CAM method ('gradcam' or 'cam')
        classifier_path: Path to trained classifier checkpoint
        output_dir: Directory to save generated masks
        visualize: Whether to save visualizations
        threshold: Threshold for mask binarization
        
    Returns:
        Path to generated masks directory
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config['paths']['masks']) / method
    else:
        output_dir = Path(output_dir)
    
    masks_dir = output_dir
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger("Generate Masks")
    
    logger.info(f"Generating masks using {method}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create CAM model
    if method == 'gradcam':
        exp_name = f"{args.backbone}_{args.init}"
        classifier = create_classifier(config, exp_name)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier = classifier.to(device)
        cam_model = GradCAMForMask(classifier)

    elif method == 'cam':
        raise NotImplementedError("CAM method is not implemented yet.")
    else:
        raise ValueError(f"Unsupported CAM method: {method}")


    # Create dataloaders
    all_dataloader = data.data_loaders(split='trainval', batch_size=1, shuffle=False)
    
    # Generate masks for all images
    for image, label, fname in tqdm(all_dataloader, desc=f"Generating masks with {method}"):
        mask_name = fname[0].split(".")[0] + ".png"
        mask_path = masks_dir / f"{mask_name}"

        try:
            # Generate CAM and mask
            cam, binary_mask = cam_model.generate_mask(
                image.to(device),
                target_class=label,
                orig_size=(224, 224),  # or actual image size if needed
                threshold=threshold
            )

            # Save binary mask
            cam_model.save_mask(binary_mask, mask_path)

        except Exception as e:
            raise ValueError(f"Error processing {fname}: {e}")

    
    logger.info(f"Generated {method} masks for all images. Saved to {masks_dir}")
    
    return masks_dir
