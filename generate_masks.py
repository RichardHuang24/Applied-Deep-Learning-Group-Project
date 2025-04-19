"""
Generate pseudo masks from classifier models using CAM methods and CCAM
"""
import os
import torch
import json
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from PIL import Image
import time

# Import our models and functions
from models.classifier import create_classifier
from models.cam import GradCAMForMask, CAMForMask, CCAMForMask
from utils.visualization import visualize_cam

# Import CCAM components
from models.train_ccam import SimMaxLoss, SimMinLoss, SupervisionLoss
from models.train_ccam import train_ccam, CCamModel

logger = logging.getLogger(__name__)


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def generate_masks(config, method='gradcam', classifier_path=None, output_dir=None,
                  visualize=False, threshold=0.4, args=None):
    """
    Generate pseudo masks for all images using CAM, GradCAM or CCAM
    
    Args:
        config: Configuration dictionary
        method: CAM method ('gradcam', 'cam', 'ccam', 'gradcam+ccam', 'cam+ccam')
        classifier_path: Path to trained classifier checkpoint
        output_dir: Directory to save generated masks
        visualize: Whether to save visualizations
        threshold: Threshold for mask binarization
        args: Additional arguments
        
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
    logger.info(f"Generating masks using {method}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Handle different methods
    if method in ['gradcam', 'cam']:
        return generate_cam_masks(config, method, classifier_path, output_dir, 
                                 visualize, threshold, args, device)[1]
    elif method == 'ccam':
        return generate_ccam_masks(config, None, classifier_path, output_dir, 
                                  visualize, threshold, args, device)
    elif method in ['gradcam+ccam', 'cam+ccam']:
        # First, generate initial CAMs
        initial_method = method.split('+')[0]  # 'gradcam' or 'cam'
        initial_cams_dir = generate_cam_masks(config, initial_method, classifier_path, 
                                             output_dir / "initial", visualize, threshold, args, device)[0]
        
        # Then, use these CAMs to train CCAM
        return generate_ccam_masks(config, initial_cams_dir, classifier_path, 
                                  output_dir, visualize, threshold, args, device)
    else:
        raise ValueError(f"Unsupported method: {method}")

def generate_cam_masks(config, method, classifier_path, output_dir, 
                      visualize, threshold, args, device):
    """
    Generate masks using CAM or GradCAM
    """
    # Create model based on method
    if method == 'gradcam':
        exp_name = f"{args.backbone}_{args.init}_{method}"
        classifier = create_classifier(config, exp_name)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier = classifier.to(device)
        cam_model = GradCAMForMask(classifier)
    elif method == 'cam':
        exp_name = f"{args.backbone}_{args.init}_{method}"
        classifier = create_classifier(config, exp_name)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier = classifier.to(device)
        cam_model = CAMForMask(classifier)
    else:
        raise ValueError(f"Unsupported CAM method: {method}")

    # Create dataloaders
    import data
    all_dataloader = data.data_loaders(split='trainval', batch_size=1, shuffle=False, keep_large=True)
    
    # Create output directories
    masks_dir = output_dir / "masks"
    cams_dir = output_dir / "cams"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate masks for all images
    for image, label, fname in tqdm(all_dataloader, desc=f"Generating masks with {method}"):
        mask_name = fname[0].split(".")[0] + ".png"
        cam_name = fname[0].split(".")[0] + ".npy"
        
        mask_path = masks_dir / mask_name
        cam_path = cams_dir / cam_name

        try:
            # Generate CAM and mask
            cam, binary_mask = cam_model.generate_mask(
                image.to(device),
                target_class=label,
                orig_size=(448, 448),  # or actual image size if needed
                threshold=threshold
            )

            # Save binary mask
            cam_model.save_mask(binary_mask, mask_path)
            # print(cam.shape)
            # print(cam.min(), cam.max(), cam.mean())
            # Save CAM as numpy array (for CCAM training)
            np.save(cam_path, cam)
            
            # Save visualization if requested
            if visualize:
                vis_path = vis_dir / f"{fname[0].split('.')[0]}_cam.png"
                visualize_cam(image, cam, binary_mask, vis_path)

        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            raise

    logger.info(f"Generated {method} masks for all images. Saved to {masks_dir}")
    
    return cams_dir, masks_dir

def generate_ccam_masks(config, initial_cams_dir, classifier_path, output_dir, 
                       visualize, threshold, args, device):
    """
    Generate masks using CCAM, optionally guided by initial CAMs
    """
    # Import data module
    import data
    
    # Get configuration parameters
    num_epochs = config.get('models', {}).get('ccam', {}).get('epochs', 15)
    batch_size = config.get('models', {}).get('ccam', {}).get('batch_size', 32)
    lr = config.get('models', {}).get('ccam', {}).get('lr', 0.0001)
    alpha = config.get('models', {}).get('ccam', {}).get('alpha', 0.05)
    num_classes = config['dataset']['num_classes']
    
    # Create CCAM model
    ccam_model = CCamModel(
        backbone=args.backbone,
        initialization=args.init
    ).to(device)
    
    # Initialize from classifier if provided
    if classifier_path and initial_cams_dir is not None:  # For 'gradcam+ccam' or 'cam+ccam'
        # Load classifier weights to initialize backbone
        classifier_dict = torch.load(classifier_path, map_location=device)
        if 'state_dict' in classifier_dict:
            classifier_dict = classifier_dict['state_dict']
            
        # Filter backbone keys and map to CCAM model structure
        backbone_dict = {}
        for k, v in classifier_dict.items():
            if k.startswith('model.'):
                # Map classifier's model.X to CCAM's backbone.X
                new_key = k.replace('model.', 'backbone.model.')
                backbone_dict[new_key] = v
        
        # Load backbone weights
        missing, unexpected = ccam_model.load_state_dict(backbone_dict, strict=False)
        logger.info(f"Initialized CCAM backbone from classifier")
        logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    
    # Create loss functions
    criterion = [
        SimMaxLoss(metric='cos', alpha=alpha).to(device),  # FG-FG (maximize similarity)
        SimMinLoss(metric='cos').to(device),               # FG-BG (minimize similarity)
        SimMaxLoss(metric='cos', alpha=alpha).to(device)   # BG-BG (maximize similarity)
    ]
    
    # Add supervision loss if initial CAMs are provided
    if initial_cams_dir is not None:
        criterion.append(SupervisionLoss(high_threshold=0.7, low_threshold=0.2).to(device))
    
    # Get parameter groups for optimizer
    param_groups = ccam_model.get_parameter_groups()
    
    # Create scheduler
    train_loader = data.data_loaders(split='trainval', batch_size=batch_size, shuffle=True)
    num_steps = len(train_loader) * num_epochs
    
    # Create optimizer
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': 0.0001},  # Backbone
        {'params': param_groups[1], 'lr': 2 * lr, 'weight_decay': 0},   # Backbone biases
        {'params': param_groups[2], 'lr': 10 * lr, 'weight_decay': 0.0001},  # Disentangler
        {'params': param_groups[3], 'lr': 20 * lr, 'weight_decay': 0}   # Disentangler biases
    ], lr=lr, weight_decay=0.0001, max_step=num_steps)
    
    # Train CCAM
    logger.info(f"Training CCAM for {num_epochs} epochs with batch size {batch_size}")
    trained_model = train_ccam(
        config=config,
        model=ccam_model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=num_epochs,
        output_dir=output_dir,
        initial_cams_dir=initial_cams_dir,
        device=device
    )
    
    # Create CAM model for mask generation using the same interface as other methods
    cam_model = CCAMForMask(trained_model)
    
    # Create dataloaders for final mask generation
    all_dataloader = data.data_loaders(split='trainval', batch_size=1, shuffle=False, keep_large=True)
    
    # Create output directories for final masks
    masks_dir = output_dir / "masks"
    cams_dir = output_dir / "cams"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate final masks using the trained CCAM model
    logger.info("Generating final CCAM masks")
    for image, label, fname in tqdm(all_dataloader, desc="Generating CCAM masks"):
        mask_name = fname[0].split(".")[0] + ".png"
        cam_name = fname[0].split(".")[0] + ".npy"
        
        mask_path = masks_dir / mask_name
        cam_path = cams_dir / cam_name
        
        try:
            # Generate CAM and mask
            cam, binary_mask = cam_model.generate_mask(
                image.to(device),
                target_class=None,  # CCAM is class-agnostic
                orig_size=(448, 448),
                threshold=threshold
            )
            
            # Save binary mask
            cam_model.save_mask(binary_mask, mask_path)
            
            # Save CAM as numpy array
            np.save(cam_path, cam)
            
            # Save visualization if requested
            if visualize:
                vis_path = vis_dir / f"{fname[0].split('.')[0]}_ccam.png"
                visualize_cam(image, cam, binary_mask, vis_path)
        
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            raise
    
    logger.info(f"Generated CCAM masks for all images. Saved to {masks_dir}")
    
    return masks_dir