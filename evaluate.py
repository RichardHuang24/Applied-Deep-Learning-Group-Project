"""
Evaluation pipeline for segmentation models
"""
import os
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from models.pspnet import create_segmentation_model
from data import create_dataloaders
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import visualize_prediction, visualize_results_grid

logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device, num_classes, visualize=False, output_dir=None):
    """
    Evaluate a segmentation model on a dataset
    
    Args:
        model: Segmentation model to evaluate
        dataloader: Validation dataloader
        device: Device to run evaluation on
        num_classes: Number of classes
        visualize: Whether to save visualizations
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_images = []
    
    # Track metrics
    total_pixel_acc = 0
    total_miou = 0
    total_samples = 0
    
    logger.info("Evaluating model...")
    
    with torch.no_grad():
        for images, masks, paths, _ in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Convert to class indices (B, H, W)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for batch
            metrics = calculate_metrics(preds, masks, num_classes)
            
            # Update totals
            batch_size = images.size(0)
            total_pixel_acc += metrics['pixel_acc'] * batch_size
            total_miou += metrics['miou'] * batch_size
            total_samples += batch_size
            
            # Store for visualization
            if visualize and output_dir:
                all_preds.extend([p for p in preds.cpu()])
                all_targets.extend([m for m in masks.cpu()])
                all_images.extend([i for i in images.cpu()])
                
                # Save individual predictions
                for i in range(batch_size):
                    img_path = Path(paths[i])
                    img_name = img_path.stem
                    
                    # Create visualization
                    vis_img = visualize_prediction(
                        images[i].cpu(),
                        preds[i].cpu(),
                        masks[i].cpu(),
                        save_path=None
                    )
                    
                    # Save visualization
                    output_path = output_dir / f"{img_name}_pred.png"
                    vis_img.save(output_path)
    
    # Calculate final metrics
    final_metrics = {
        'pixel_acc': total_pixel_acc / total_samples,
        'miou': total_miou / total_samples
    }
    
    # Print metrics
    logger.info(f"Pixel Accuracy: {final_metrics['pixel_acc']:.4f}")
    logger.info(f"Mean IoU: {final_metrics['miou']:.4f}")
    
    # Save grid visualization
    if visualize and output_dir and all_images:
        # Create visualization grid
        grid = visualize_results_grid(
            all_images[:16],  # Limit to 16 samples
            all_preds[:16],
            all_targets[:16],
            save_path=output_dir / "results_grid.png"
        )
    
    return final_metrics

def evaluate(config_path, supervision='full', checkpoint_path=None, output_dir=None, visualize=True):
    """
    Evaluate segmentation model
    
    Args:
        config_path: Path to configuration file
        supervision: Supervision type ('full', 'weak_gradcam', 'weak_ccam')
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
        visualize: Whether to save visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config['paths']['results']) / f"eval_{supervision}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "evaluation.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Evaluating model with {supervision} supervision")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    pseudo_masks_dir = None
    if supervision.startswith('weak_'):
        cam_method = supervision.split('_')[1]
        pseudo_masks_dir = Path(config['paths']['masks']) / cam_method
    
    _, val_loader = create_dataloaders(
        config=config,
        supervision=supervision,
        pseudo_masks_dir=pseudo_masks_dir
    )
    
    # Create model
    model = create_segmentation_model(config_path)
    model = model.to(device)
    
    # Load checkpoint
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("No checkpoint provided. Using random weights.")
    
    # Set number of classes for segmentation (Foreground/Background)
    num_classes = 2  # Always 2 classes for this segmentation task
    
    # Create output directory for visualizations
    vis_dir = output_dir / "visualizations"
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        visualize=visualize,
        output_dir=vis_dir
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    
    return metrics

def evaluate_all_models(config_path, output_dir=None):
    """
    Evaluate all segmentation models defined in config
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary of metrics for all models
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config['paths']['results']) / "comparison"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "evaluation.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Evaluating all segmentation models")
    
    # Get all experiment configurations
    experiments = config['experiments']['segmentation']
    
    all_metrics = {}
    
    # Evaluate each model
    for exp in experiments:
        model_name = exp['name']
        supervision = exp['supervision']
        
        logger.info(f"Evaluating model: {model_name}")
        
        # Find checkpoint path
        model_dir = Path(config['paths']['outputs']) / "segmentation" / model_name
        checkpoint_paths = list(model_dir.glob("*.pth"))
        
        if not checkpoint_paths:
            logger.warning(f"No checkpoint found for {model_name}")
            continue
        
        # Use the most recent checkpoint
        checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)
        
        # Create output directory for this model
        model_output_dir = output_dir / model_name
        
        # Evaluate model
        metrics = evaluate(
            config_path=config_path,
            supervision=supervision,
            checkpoint_path=checkpoint_path,
            output_dir=model_output_dir,
            visualize=True
        )
        
        all_metrics[model_name] = metrics
    
    # Save comparison results
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Create comparison visualization
    models = list(all_metrics.keys())
    pixel_accuracies = [all_metrics[m]['pixel_acc'] for m in models]
    mious = [all_metrics[m]['miou'] for m in models]
    
    # Create bar chart image using PIL
    from PIL import Image, ImageDraw
    
    # Set image dimensions
    width, height = 800, 400
    padding = 60
    bar_width = (width - 2*padding) / (len(models) * 2 + 1)
    max_bar_height = height - 2*padding
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw bars
    for i, model in enumerate(models):
        # Pixel accuracy bar
        x1 = padding + (2*i) * bar_width
        y1 = height - padding - pixel_accuracies[i] * max_bar_height
        x2 = x1 + bar_width
        y2 = height - padding
        draw.rectangle([x1, y1, x2, y2], fill='blue')
        
        # mIoU bar
        x1 = padding + (2*i+1) * bar_width
        y1 = height - padding - mious[i] * max_bar_height
        x2 = x1 + bar_width
        y2 = height - padding
        draw.rectangle([x1, y1, x2, y2], fill='green')
        
        # Label
        draw.text((padding + (2*i+0.5) * bar_width, height - padding + 10), 
                  model, fill='black')
    
    # Draw axes
    draw.line([padding, padding, padding, height-padding], fill='black', width=2)
    draw.line([padding, height-padding, width-padding, height-padding], 
              fill='black', width=2)
    
    # Draw legend
    draw.rectangle([width-150, 50, width-130, 70], fill='blue')
    draw.text([width-120, 50], "Pixel Accuracy", fill='black')
    draw.rectangle([width-150, 80, width-130, 100], fill='green')
    draw.text([width-120, 80], "mIoU", fill='black')
    
    # Save plot
    img.save(output_dir / "metrics_comparison.png")
    
    logger.info(f"Comparison results saved to {output_dir}")
    
    return all_metrics