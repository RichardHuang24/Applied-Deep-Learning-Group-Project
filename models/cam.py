"""
Class Activation Map (CAM) implementations:
- Grad-CAM
- CCAM (Cross-Channel Class Activation Mapping)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation for visualizing class activation maps
    
    Reference: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    """
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM
        
        Args:
            model: Trained classifier model
            target_layer: Target layer for CAM generation (if None, use last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Register hooks
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Get target layer (default: last conv layer)
        if target_layer is None:
            if hasattr(model, 'model'):  # For ResNetClassifier wrapper
                target_layer = model.model.layer4[-1]
            else:
                target_layer = model.layer4[-1]
        
        # Register forward and backward hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
        
    def __del__(self):
        # Remove hooks when object is deleted
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None, relu=True):
        """
        Generate class activation map
        
        Args:
            input_image: Input tensor image [B, C, H, W]
            target_class: Target class index (if None, use predicted class)
            relu: Whether to apply ReLU to CAM output
            
        Returns:
            numpy.ndarray: Class activation map
        """
        # Forward pass
        input_image = input_image.clone().requires_grad_(True)
        
        if hasattr(self.model, 'forward'):
            model_output = self.model(input_image)
        else:
            # Handle case where model might be a wrapper with a different forward method
            model_output = self.model.forward(input_image)
        
        if isinstance(model_output, dict):
            # Handle the case where the model returns a dict
            model_output = model_output['logits']
        
        # If target class is not specified, use the predicted class
        batch_size = input_image.size(0)
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)
        elif not isinstance(target_class, torch.Tensor):
            target_class = torch.tensor([target_class] * batch_size, 
                                       device=input_image.device)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        for idx, cls in enumerate(target_class):
            one_hot[idx, cls] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights using global average pooling (GAP) on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM by weighting activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU if requested
        if relu:
            cam = F.relu(cam)
        
        # Normalize each CAM
        batch_cams = []
        for i in range(batch_size):
            cam_i = cam[i].squeeze().cpu().detach().numpy()
            cam_i = cam_i - np.min(cam_i)
            cam_max = np.max(cam_i)
            if cam_max != 0:
                cam_i = cam_i / cam_max
            batch_cams.append(cam_i)
        
        return np.array(batch_cams)


class CCAM(nn.Module):
    """
    Cross-Channel Class Activation Mapping (CCAM)
    
    A more effective CAM method that captures cross-channel interactions 
    for better localization in weakly-supervised semantic segmentation.
    """
    def __init__(self, backbone='resnet18', num_classes=37):
        """
        Initialize CCAM
        
        Args:
            backbone: Backbone model name
            num_classes: Number of classes
        """
        super(CCAM, self).__init__()
        
        # Create backbone network
        from .classifier import ResNetClassifier
        self.backbone = ResNetClassifier(backbone=backbone, num_classes=num_classes)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone.get_features(dummy)
            _, C, H, W = features.shape
        
        # Class-specific attention branch
        # This generates attention maps for each class
        self.attention_branch = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, num_classes, kernel_size=1)
        )
        
        # Classification branch
        # This generates classification scores
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classification_branch = nn.Linear(C, num_classes)
        
    def forward(self, x):
        """
        Forward pass through CCAM
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            dict: Dictionary with features, attention maps and classification scores
        """
        # Extract features from backbone
        features = self.backbone.get_features(x)
        
        # Generate attention maps (one per class)
        attention_maps = self.attention_branch(features)
        
        # Classification using global average pooling
        gap_features = self.gap(features).view(features.size(0), -1)
        classification_logits = self.classification_branch(gap_features)
        
        return {
            'features': features,
            'attention_maps': attention_maps,
            'logits': classification_logits
        }
    
    def get_cam(self, x, target_class=None):
        """
        Generate class activation map
        
        Args:
            x: Input tensor [B, 3, H, W]
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            numpy.ndarray: Class activation map
        """
        # Forward pass
        output = self.forward(x)
        attention_maps = output['attention_maps']
        logits = output['logits']
        
        # If target class is not specified, use predicted class
        batch_size = x.size(0)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        elif not isinstance(target_class, torch.Tensor):
            target_class = torch.tensor([target_class] * batch_size, 
                                       device=x.device)
        
        # Extract CAM for target classes
        batch_cams = []
        for i in range(batch_size):
            cls_idx = target_class[i].item()
            cam = attention_maps[i, cls_idx].cpu().detach().numpy()
            
            # Normalize
            cam = cam - np.min(cam)
            cam_max = np.max(cam)
            if cam_max != 0:
                cam = cam / cam_max
                
            batch_cams.append(cam)
        
        return np.array(batch_cams)


def create_cam_model(config_path='config.yaml', method='ccam', backbone='resnet18'):
    """
    Factory function to create CAM models
    
    Args:
        config_path: Path to configuration file
        method: CAM method ('gradcam' or 'ccam')
        backbone: Backbone model name
        
    Returns:
        CAM model (GradCAM or CCAM instance)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['dataset']['num_classes']
    
    if method == 'gradcam':
        # For GradCAM we need a classifier model first
        from .classifier import ResNetClassifier
        classifier = ResNetClassifier(backbone=backbone, num_classes=num_classes)
        return GradCAM(classifier)
    elif method == 'ccam':
        return CCAM(backbone=backbone, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported CAM method: {method}")