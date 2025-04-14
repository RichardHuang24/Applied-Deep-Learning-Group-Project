"""
Evaluation metrics for semantic segmentation implemented in PyTorch
"""
import torch
import logging

logger = logging.getLogger(__name__)

def calculate_pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy between predicted and target masks
    
    Args:
        pred: Predicted segmentation mask (B, H, W) - class indices
        target: Target segmentation mask (B, H, W) - class indices
        
    Returns:
        float: Pixel accuracy
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    else:
        pred = torch.tensor(pred)
        
    if isinstance(target, torch.Tensor):
        target = target.detach()
    else:
        target = torch.tensor(target)
    
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate accuracy
    correct = torch.sum(pred_flat == target_flat).item()
    total = target_flat.size(0)
    
    return correct / total

def calculate_iou(pred, target, cls_idx):
    """
    Calculate IoU for a specific class
    
    Args:
        pred: Predicted segmentation mask (B, H, W) - class indices
        target: Target segmentation mask (B, H, W) - class indices
        cls_idx: Class index to calculate IoU for
        
    Returns:
        float: IoU score for the class
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    else:
        pred = torch.tensor(pred)
        
    if isinstance(target, torch.Tensor):
        target = target.detach()
    else:
        target = torch.tensor(target)
    
    # Create binary masks for the class
    pred_mask = (pred == cls_idx)
    target_mask = (target == cls_idx)
    
    # Calculate intersection and union
    intersection = (pred_mask & target_mask).sum().float()
    union = (pred_mask | target_mask).sum().float()
    
    # Handle division by zero
    if union == 0:
        return 0.0
    
    return (intersection / union).item()

def calculate_miou(pred, target, num_classes):
    """
    Calculate mean IoU across all classes
    
    Args:
        pred: Predicted segmentation mask (B, H, W) - class indices
        target: Target segmentation mask (B, H, W) - class indices
        num_classes: Number of classes
        
    Returns:
        tuple: (mean IoU, list of class IoUs)
    """
    class_ious = []
    
    for cls_idx in range(num_classes):
        iou = calculate_iou(pred, target, cls_idx)
        class_ious.append(iou)
    
    # Filter out classes that are not present in the target
    valid_ious = [iou for iou in class_ious if iou > 0 or iou == 0]
    
    if len(valid_ious) == 0:
        return 0.0, class_ious
    
    return sum(valid_ious) / len(valid_ious), class_ious

def calculate_metrics(pred, target, num_classes):
    """
    Calculate all metrics for segmentation evaluation
    
    Args:
        pred: Predicted segmentation mask (B, H, W) - class indices
        target: Target segmentation mask (B, H, W) - class indices
        num_classes: Number of classes
        
    Returns:
        dict: Dictionary with all metrics
    """
    # Convert logits to class indices if needed
    if pred.dim() == 4:  # (B, C, H, W)
        pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    if target.dim() == 4:  # (B, C, H, W)
        target = torch.argmax(target, dim=1)  # (B, H, W)
    
    # Calculate pixel accuracy
    pixel_acc = calculate_pixel_accuracy(pred, target)
    
    # Calculate mIoU
    miou, class_ious = calculate_miou(pred, target, num_classes)
    
    return {
        'pixel_acc': pixel_acc,
        'miou': miou,
        'class_ious': class_ious
    }

def print_metrics(metrics, class_names=None):
    """
    Print metrics in a readable format
    
    Args:
        metrics: Dictionary with metrics
        class_names: List of class names (optional)
    """
    logger.info(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")
    logger.info(f"Mean IoU: {metrics['miou']:.4f}")
    
    if class_names is not None and 'class_ious' in metrics:
        logger.info("Per-class IoU:")
        for i, iou in enumerate(metrics['class_ious']):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            logger.info(f"  {class_name}: {iou:.4f}")