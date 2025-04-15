"""
Class Activation Map (CAM) implementations:
- Grad-CAM
- CCAM (Cross-Channel Class Activation Mapping)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
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


def create_cam_model(config_path='config.json', method='ccam', backbone='resnet18'):
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
        config = json.load(f)
    
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