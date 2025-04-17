"""
Training functions for all models (classifier, CAM, segmentation)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import logging
from pathlib import Path
import time
from datetime import datetime
import data
from models.classifier import create_classifier
from models.pspnet import create_segmentation_model
from utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def setup_device(seed):
    """Setup device and seeds"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Using device: {device}")
    return device

def train_classifier(config, experiment, output_dir=None):
    """
    Train a classifier with the specified configuration
    
    Args:
        config: Path to configuration file
        experiment: Experiment name (e.g., 'resnet18_imagenet')
        output_dir: Output directory
        
    Returns:
        str: Path to saved checkpoint
    """
    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger("Train Classifier")
    
    # Set up device, set seed
    device = setup_device(config.get('training', {}).get('seed', 42))
    logger.info(f"Using device: {device}")
    
    # Extract training parameters from config
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 32)
    num_epochs = training_config.get('num_epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    # Parse experiment name to get backbone and initialization
    parts = experiment.split('_')
    if len(parts) >= 2:
        backbone_name = parts[0]    # resnet50
        initialization = parts[1]   # random
    else:
        backbone_name = 'resnet18'  # Default backbone
        initialization = 'imagenet'  # Default initialization

    try:
        # Create dataloaders
        train_loader = data.data_loaders(split='train', batch_size=batch_size)
        val_loader = data.data_loaders(split='val', batch_size=batch_size, shuffle=False)
        
        # Create model
        logger.info(f"Creating {backbone_name} model with {initialization} initialization")
        model = create_classifier(config, experiment)
        model = model.to(device)
        
        # Set up criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training loop
        best_acc = 0.0
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training step
            for batch_idx, (imgs, labels, _) in enumerate(tqdm(train_loader, desc="Training Classifier")):
                # Extract data
                images = imgs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                try:
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.detach().item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Calculate training statistics
            epoch_loss = running_loss / total if total > 0 else float('inf')
            epoch_acc = correct / total if total > 0 else 0
            logger.info(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            # Validation step
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Extract data
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                    
                    # Forward pass
                    try:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        # Statistics
                        val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                    except Exception as e:
                        logger.error(f"Error in validation batch: {e}")
                        continue
            
            # Calculate validation statistics
            val_epoch_loss = val_loss / val_total if val_total > 0 else float('inf')
            val_epoch_acc = val_correct / val_total if val_total > 0 else 0
            logger.info(f"Validation Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")
            
            # Save best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model with acc: {best_acc:.4f}")
        
        # # Save final model
        # final_checkpoint_path = os.path.join(output_dir, 'final_model.pth')
        # torch.save(model.state_dict(), final_checkpoint_path)
        logger.info(f"Training completed. Best accuracy: {best_acc:.4f}")
        
        return checkpoint_path
    
    except Exception as e:
        logger.error(f"Error in train_classifier: {str(e)}")
        raise


def train_segmentation(config, supervision='full', experiment_name=None, pseudo_masks_dir=None, num_epochs=None, output_dir=None):
    """Train a segmentation model with specified supervision type"""

    logger = logging.getLogger("Train Segmentation")
    logger.info(f"Training segmentation model with supervision: {supervision}")
    
    output_dir = Path(output_dir) if output_dir else Path(config['paths']['outputs']) / "segmentation" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging and device
    device = setup_device(config['training']['seed'])
    if supervision == 'full':
        return_trimaps = True
        return_pseudomask = False
    elif supervision.startswith('weak'):
        return_trimaps = True
        return_pseudomask = True
    else:
        return_trimaps = False
        return_pseudomask = False
    logger.info(f"Return trimaps: {return_trimaps}, Return pseudomask: {return_pseudomask}")
    
    try:
        # Setup training components
        num_epochs = num_epochs or config['training']['epochs']['pspnet']
        backbone = config['models']['pspnet']['backbone']
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 32)
        learning_rate = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.0001)
        
        # Initialize model and data
        model = create_segmentation_model(backbone=backbone).to(device)

        train_loader = data.data_loaders(
            split='train',
            batch_size=batch_size,
            return_pseudomask=return_pseudomask,
            return_trimaps=return_trimaps,
            mask_dir=pseudo_masks_dir
        )

        val_loader = data.data_loaders(
            split='val',
            batch_size=batch_size,
            return_pseudomask=False,
            return_trimaps=return_trimaps,
            shuffle=False
        )
        
        # Setup training tools
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Training loop
        best_miou = 0.0
        checkpoint_path = output_dir / "segmentation_best.pth"
        
        for epoch in range(num_epochs):
            train_metrics = train_segmentation_epoch(model, train_loader, criterion, optimizer, device, config)
            val_metrics = validate_segmentation_epoch(model, val_loader, criterion, device, config)
            
            scheduler.step(val_metrics['miou'])
            log_metrics(epoch, num_epochs, train_metrics, val_metrics)
            
            if val_metrics['miou'] > best_miou:
                best_miou = val_metrics['miou']
                torch.save(model.state_dict(), checkpoint_path)
        
        logger.info(f"Training completed. Best mIoU: {best_miou:.4f}")
        return str(checkpoint_path)
        
    except Exception as e:
        logger.error(f"Error in train_segmentation: {str(e)}")
        raise

def get_masks_dir(supervision, pseudo_masks_dir, config):
    """Get masks directory based on supervision type"""
    if supervision.startswith('weak_') and pseudo_masks_dir is None:
        cam_method = supervision.split('_')[1]
        return Path(config['paths']['masks']) / cam_method / "masks"
    return pseudo_masks_dir

def train_segmentation_epoch(model, dataloader, criterion, optimizer, device, config):
    """Run one training epoch for segmentation"""
    model.train()
    running_loss = 0.0
    running_pixel_acc = 0.0
    running_miou = 0.0
    total_samples = 0
    num_classes = 2
    
    for images, masks, _ in tqdm(dataloader, desc="Training"):

        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle auxiliary loss if present
        if isinstance(outputs, tuple):
            outputs, aux_outputs = outputs
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long()
            loss = criterion(outputs, masks) + 0.4 * criterion(aux_outputs, masks)
        else:
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long()
            loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        
        preds = torch.argmax(outputs, dim=1)
        metrics = calculate_metrics(preds, masks, num_classes)
        
        running_pixel_acc += metrics['pixel_acc'] * batch_size
        running_miou += metrics['miou'] * batch_size
        total_samples += batch_size
    
    if total_samples == 0:
        return {'loss': float('inf'), 'pixel_acc': 0.0, 'miou': 0.0}
    
    return {
        'loss': running_loss / total_samples,
        'pixel_acc': running_pixel_acc / total_samples,
        'miou': running_miou / total_samples
    }

def validate_segmentation_epoch(model, dataloader, criterion, device, config):
    """Run one validation epoch for segmentation"""
    model.eval()
    running_loss = 0.0
    running_pixel_acc = 0.0
    running_miou = 0.0
    total_samples = 0
    num_classes = 2 # Always 2 classes (Foreground/Background) for segmentation validation
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            # Use only main output for validation if tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)

            masks = masks.long()
            loss = criterion(outputs, masks)

            
            # Calculate metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            
            preds = torch.argmax(outputs, dim=1)
            metrics = calculate_metrics(preds, masks, num_classes)
            
            running_pixel_acc += metrics['pixel_acc'] * batch_size
            running_miou += metrics['miou'] * batch_size
            total_samples += batch_size
    
    if total_samples == 0:
        return {'loss': float('inf'), 'pixel_acc': 0.0, 'miou': 0.0}
    
    return {
        'loss': running_loss / total_samples,
        'pixel_acc': running_pixel_acc / total_samples,
        'miou': running_miou / total_samples
    }

def log_metrics(epoch, num_epochs, train_metrics, val_metrics):
    """Log training and validation metrics"""
    logger.info(
        f"Epoch {epoch+1}/{num_epochs} - "
        f"Train Loss: {train_metrics['loss']:.4f}, "
        f"Train Acc: {train_metrics['pixel_acc']:.4f}, "
        f"Train mIoU: {train_metrics['miou']:.4f}, "
        f"Val Loss: {val_metrics['loss']:.4f}, "
        f"Val Acc: {val_metrics['pixel_acc']:.4f}, "
        f"Val mIoU: {val_metrics['miou']:.4f}"
    )