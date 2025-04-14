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

from models.classifier import create_classifier
from models.cam import create_cam_model
from models.pspnet import create_segmentation_model
from data import create_dataloaders
from utils.metrics import calculate_metrics

def train_classifier(config_path, experiment, output_dir=None):
    """
    Train a classifier with the specified configuration
    
    Args:
        config_path: Path to configuration file
        experiment: Experiment name (e.g., 'resnet18_imagenet')
        output_dir: Output directory
        
    Returns:
        str: Path to saved checkpoint
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(config.get('paths', {}).get('outputs', 'outputs'), 'classifiers', experiment)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging FIRST
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('train_classifier')
    
    # THEN set up device - now the logger exists
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Extract training parameters from config - use safer access with defaults
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 32)
    num_epochs = training_config.get('num_epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    # Parse experiment name to get backbone and initialization
    parts = experiment.split('_')
    if len(parts) >= 2:
        backbone_name = parts[0]
        initialization = parts[1]
    else:
        backbone_name = 'resnet18'  # Default backbone
        initialization = 'imagenet'  # Default initialization
    
    # Extract dataset information
    dataset_config = config.get('dataset', {})
    num_classes = dataset_config.get('num_classes', 37)  # Default to Pet dataset classes
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config=config,
        split='train', 
        batch_size=batch_size
    )
    
    # Create model
    logger.info(f"Creating {backbone_name} model with {initialization} initialization")
    model = create_classifier(config_path, experiment)
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
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training step
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Extract data - now we know exactly what's in the batch
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Debugging info (first batch only)
            if batch_idx == 0 and epoch == 0:
                logger.info(f"Batch shape: {images.shape}")
                logger.info(f"Labels shape: {labels.shape}")
                logger.info(f"Label values: {labels}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate training statistics
        epoch_loss = running_loss / total
        epoch_acc = correct / total
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
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model with acc: {best_acc:.4f}")
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    logger.info(f"Training completed. Best accuracy: {best_acc:.4f}")
    
    return final_checkpoint_path

def train_cam(config_path, method='ccam', backbone='resnet18', num_epochs=None, output_dir=None):
    """
    Train a CAM model (Note: GradCAM doesn't require training)
    
    Args:
        config_path: Path to configuration file
        method: CAM method ('gradcam' or 'ccam')
        backbone: Backbone model name
        num_epochs: Number of epochs to train (override config)
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved checkpoint or None if GradCAM
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get training parameters
    if num_epochs is None:
        num_epochs = config['training']['epochs']['cam']
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config['paths']['outputs']) / "cam" / method
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Training CAM model: {method}")
    logger.info(f"Backbone: {backbone}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])
    
    # Create model
    model = create_cam_model(config_path, method, backbone)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Check if model requires training
    if method == 'ccam':
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            config=config,
            split='train',
            batch_size=config['training']['batch_size']
        )
        
        # Create loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr']['cam']
        )
        
        # Create learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Training loop
        best_acc = 0.0
        checkpoint_path = output_dir / f"{method}_{backbone}_best.pth"
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(images)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate epoch training metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    # Move data to device
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate epoch validation metrics
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'method': method,
                    'backbone': backbone
                }, checkpoint_path)
                logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
        
        logger.info(f"Training completed. Best validation accuracy: {best_acc:.4f}")
        return str(checkpoint_path)
    else:
        logger.info(f"GradCAM doesn't require training as it uses a trained classifier")
        return None

def train_segmentation(config_path, supervision='full', pseudo_masks_dir=None, 
                       num_epochs=None, output_dir=None):
    """
    Train a segmentation model
    
    Args:
        config_path: Path to configuration file
        supervision: Supervision type ('full', 'weak_gradcam', 'weak_ccam')
        pseudo_masks_dir: Directory with pseudo masks
        num_epochs: Number of epochs to train (override config)
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved checkpoint
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get training parameters
    if num_epochs is None:
        num_epochs = config['training']['epochs']['pspnet']
    
    # Find experiment configuration
    for exp in config['experiments']['segmentation']:
        if exp['supervision'] == supervision:
            model_name = exp['name']
            break
    else:
        model_name = supervision
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config['paths']['outputs']) / "segmentation" / model_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Training segmentation model: {model_name}")
    logger.info(f"Supervision: {supervision}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['training']['seed'])
    
    # Create dataloaders
    if supervision.startswith('weak_') and pseudo_masks_dir is None:
        # Determine pseudo masks directory from supervision type
        cam_method = supervision.split('_')[1]
        pseudo_masks_dir = Path(config['paths']['masks']) / cam_method / "masks"
    
    train_loader, val_loader = create_dataloaders(
        config=config,
        split='train',
        batch_size=config['training']['batch_size'],
        supervision=supervision,
        pseudo_masks_dir=pseudo_masks_dir
    )
    
    # Create model
    backbone = config['models']['pspnet']['backbone']
    model = create_segmentation_model(config_path, backbone)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr']['pspnet']
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_miou = 0.0
    checkpoint_path = output_dir / f"{model_name}_best.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pixel_acc = 0.0
        train_miou = 0.0
        train_samples = 0
        
        for images, masks, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Handle auxiliary loss if training
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss = criterion(outputs, masks) + 0.4 * criterion(aux_outputs, masks)
            else:
                loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=1)
            metrics = calculate_metrics(preds, masks, config['dataset']['num_classes'] + 1)  # Add 1 for background
            
            train_pixel_acc += metrics['pixel_acc'] * batch_size
            train_miou += metrics['miou'] * batch_size
            train_samples += batch_size
        
        # Calculate epoch training metrics
        train_loss = train_loss / train_samples
        train_pixel_acc = train_pixel_acc / train_samples
        train_miou = train_miou / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pixel_acc = 0.0
        val_miou = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, masks, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use only main output for validation
                
                loss = criterion(outputs, masks)
                
                # Track statistics
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                
                # Calculate metrics
                preds = torch.argmax(outputs, dim=1)
                metrics = calculate_metrics(preds, masks, config['dataset']['num_classes'] + 1)  # Add 1 for background
                
                val_pixel_acc += metrics['pixel_acc'] * batch_size
                val_miou += metrics['miou'] * batch_size
                val_samples += batch_size
        
        # Calculate epoch validation metrics
        val_loss = val_loss / val_samples
        val_pixel_acc = val_pixel_acc / val_samples
        val_miou = val_miou / val_samples
        
        # Update learning rate scheduler
        scheduler.step(val_miou)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_pixel_acc:.4f}, Train mIoU: {train_miou:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_pixel_acc:.4f}, Val mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_pixel_acc': val_pixel_acc,
                'supervision': supervision,
                'backbone': backbone
            }, checkpoint_path)
            logger.info(f"Saved best model with val_miou: {val_miou:.4f}")
    
    logger.info(f"Training completed. Best validation mIoU: {best_miou:.4f}")
    return str(checkpoint_path)