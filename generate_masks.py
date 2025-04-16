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

def generate_mask_for_image(cam_model, image, target_class=None, threshold=0.4):
    """
    Generate binary mask for a single image using CAM
    
    Args:
        cam_model: CAM model to use
        image: Input image tensor
        target_class: Target class index
        threshold: Threshold for binarization
        
    Returns:
        Binary mask tensor
    """
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Generate CAM
    if isinstance(cam_model, GradCAM):
        cam = cam_model.generate_cam(image, target_class)
    else:
        raise ValueError(f"Unsupported CAM model type: {type(cam_model)}")
    
    # Binarize using threshold
    mask = (cam > threshold).astype(np.uint8)

    bg_mask = 1.0 - mask

    return torch.tensor(mask.squeeze())

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
    
    masks_dir = output_dir / "masks"
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
    all_dataloader = data.data_loaders(split='all', batch_size=1, shuffle=False)
    
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

def generate_all_masks(config_path, output_base_dir=None):
    """
    Generate masks using all CAM methods
    
    Args:
        config_path: Path to configuration file
        output_base_dir: Base directory for all masks
        
    Returns:
        Dictionary of paths to generated masks
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if output_base_dir is None:
        output_base_dir = Path(config['paths']['masks'])
    else:
        output_base_dir = Path(output_base_dir)
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_base_dir / "generation.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Generating masks using all CAM methods")
    
    # Get trained classifier checkpoints
    classifier_dir = Path(config['paths']['outputs']) / "classifier"
    classifier_paths = {}
    
    for exp in config['experiments']['classifier']:
        model_name = exp['name']
        model_dir = classifier_dir / model_name
        
        if model_dir.exists():
            checkpoint_paths = list(model_dir.glob("*.pth"))
            if checkpoint_paths:
                # Use the most recent checkpoint
                classifier_paths[model_name] = str(max(checkpoint_paths, key=os.path.getmtime))
    
    if not classifier_paths:
        logger.warning("No trained classifiers found. Using random weights.")
    
    # Generate masks for each method
    mask_dirs = {}
    
    for method in config['models']['cam']['methods']:
        # Select appropriate classifier for method
        if method == 'gradcam':
            # For GradCAM, use ResNet18 with ImageNet weights
            classifier_path = classifier_paths.get('resnet18_imagenet', list(classifier_paths.values())[0] if classifier_paths else None)
        else:  # CCAM
            classifier_path = classifier_paths.get('resnet18_imagenet', list(classifier_paths.values())[0] if classifier_paths else None)
        
        logger.info(f"Generating {method} masks using classifier: {classifier_path}")
        
        output_dir = output_base_dir / method
        mask_dirs[method] = generate_masks(
            config_path=config_path,
            method=method,
            classifier_path=classifier_path,
            output_dir=output_dir,
            visualize=True,
            threshold=config['models']['cam']['threshold']
        )
    
    return mask_dirs