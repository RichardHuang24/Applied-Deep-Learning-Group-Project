"""
Class Activation Map (CAM) implementations:
- Grad-CAM
- CAM (Zhou et al. 2016)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image
from pathlib import Path

from .classifier import ResNetCAM

logger = logging.getLogger(__name__)

class GradCAMForMask:
    """
    Grad-CAM wrapper for generating pseudo-masks for weakly-supervised segmentation.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None
        self.hooks = []

        # Automatically find the last conv layer if not specified
        if target_layer is None:
            if hasattr(model, 'model'):  # Custom wrapper case
                target_layer = model.model.layer4[-1]
            else:
                target_layer = model.layer4[-1]

        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(self._backward_hook))

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_mask(self, image_tensor, target_class=None, orig_size=None, threshold=0.4, relu=True):
        """
        Generate a pseudo-mask from an input image tensor.

        Args:
            image_tensor (Tensor): Image tensor [1, C, H, W]
            target_class (int or None): If None, use predicted class
            orig_size (tuple): Resize CAM to this (H, W) if given
            threshold (float): Value in [0, 1] to threshold the CAM
            relu (bool): Apply ReLU before thresholding

        Returns:
            cam_np: Raw CAM (H, W) as float
            binary_mask: Thresholded mask (H, W) as uint8
        """
        image_tensor = image_tensor.clone().requires_grad_(True)

        # Forward pass
        outputs = self.model(image_tensor)
        if isinstance(outputs, dict):
            outputs = outputs['logits']

        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()
        elif isinstance(target_class, torch.Tensor):
            target_class = target_class.item()

        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM weights
        weights = torch.sum(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)  # shape [1, H, W]

        # print(f"min", cam.min(), "max", cam.max(), "mean", cam.mean(), "10% quantiles", cam.quantile(0.1), "30% quantiles", cam.quantile(0.3), "50% quantiles", cam.quantile(0.5), "70% quantiles", cam.quantile(0.7), "90% quantiles", cam.quantile(0.9))
        # print("\n\n")
        if relu:
            cam = F.relu(cam)
        # print(f"min", cam.min(), "max", cam.max(), "mean", cam.mean(), "quantiles", cam.quantile(0.1), cam.quantile(0.3), cam.quantile(0.5), cam.quantile(0.7), cam.quantile(0.9))
        # print("\n\n")

        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()
        # cam -= cam.min()
        # cam_max = cam.max()
        # if cam_max > 0:
            # cam /= cam_max
        # cam = (cam - cam.mean()) / cam.std()
        # print(f"min", cam.min(), "max", cam.max(), "mean", cam.mean(), "10% quantiles", cam.quantile(0.1), "30% quantiles", cam.quantile(0.3), "50% quantiles", cam.quantile(0.5), "70% quantiles", cam.quantile(0.7), "90% quantiles", cam.quantile(0.9))
        # print("\n\n")

        # Resize if needed
        if orig_size is not None:
            cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

            # Resize using bilinear interpolation
            resized_cam = F.interpolate(cam_tensor, size=orig_size, mode='bilinear', align_corners=False)

            # Convert back to numpy [H, W]
            cam = resized_cam.squeeze().cpu().numpy()

        # Threshold to get binary mask
        binary_mask = (cam >= threshold).astype(np.uint8)

        return cam, binary_mask

    def save_mask(self, mask, save_path):
        """
        Save binary mask to a .png file using pathlib.Path.

        Args:
            mask (np.ndarray): Binary mask (0 or 1)
            save_path (Path): Pathlib Path to save the .png mask
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir exists
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)


class CAMForMask:
    """
    Class Activation Map generator for ResNetCAM models.
    
    This class generates class activation maps for a model with
    Global Average Pooling architecture (following the original CAM paper).
    """
    
    def __init__(self, model):
        """
        Initialize CAM generator with a trained model.
        
        Args:
            model (ResNetCAM): A trained ResNetCAM model
        """
        if not isinstance(model, ResNetCAM):
            raise ValueError("Model must be an instance of ResNetCAM")
        
        self.model = model
        self.model.eval()
    
    def generate_mask(self, image_tensor, target_class=None, orig_size=None, threshold=0.4):
        """
        Generate a class activation map and binary mask from an input image.
        
        Args:
            image_tensor (Tensor): Input image tensor [1, C, H, W]
            target_class (int or None): Target class index. If None, use predicted class
            orig_size (tuple): Resize CAM to this (H, W) if given
            threshold (float): Value in [0, 1] to threshold the CAM
            
        Returns:
            cam_np: Raw CAM (H, W) as float
            binary_mask: Thresholded mask (H, W) as uint8
        """
        # Forward pass
        with torch.no_grad():
            # Get model prediction if target class not specified
            if target_class is None:
                outputs = self.model(image_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
            elif isinstance(target_class, torch.Tensor):
                target_class = target_class.item()
            
            # Get feature maps from the last convolutional layer
            feature_maps = self.model.get_activations(image_tensor)  # [1, C, H, W]
            
            # Get class-specific weights
            class_weights = self.model.get_cam_weights()[target_class]  # [C]
            
            # Compute weighted sum of feature maps
            batch_size, num_channels, height, width = feature_maps.shape
            cam = torch.zeros(height, width, device=feature_maps.device)
            
            for i, w in enumerate(class_weights):
                cam += w * feature_maps[0, i]  # Add weighted activation maps
            # Apply ReLU to focus on positive activations
            cam = F.relu(cam)
            
            # Convert to numpy and normalize
            cam_np = cam.cpu().numpy()
            # cam_np -= cam_np.min()
            # cam_max = cam_np.max()
            # if cam_max > 0:
            #     cam_np /= cam_max
            # print(f"min", cam.min(), "max", cam.max(), "mean", cam.mean(), "10% quantiles", cam.quantile(0.1), "30% quantiles", cam.quantile(0.3), "50% quantiles", cam.quantile(0.5), "70% quantiles", cam.quantile(0.7), "90% quantiles", cam.quantile(0.9))
            # print("\n\n")
            # Resize if needed
            if orig_size is not None:
                cam_tensor = torch.tensor(cam_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                resized_cam = F.interpolate(cam_tensor, size=orig_size, mode='bilinear', align_corners=False)
                cam_np = resized_cam.squeeze().cpu().numpy()
            
            # Threshold to get binary mask
            binary_mask = (cam_np >= threshold).astype(np.uint8)
            
            return cam_np, binary_mask
    
    def save_mask(self, mask, save_path):
        """
        Save binary mask to a .png file using pathlib.Path.
        
        Args:
            mask (np.ndarray): Binary mask (0 or 1)
            save_path (Path): Pathlib Path to save the .png mask
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)


class CCAMForMask:
    """
    CCAM wrapper for generating pseudo-masks for weakly-supervised segmentation.
    Follows the same interface as GradCAMForMask and CAMForMask to ensure compatibility.
    """
    def __init__(self, model):
        """
        Initialize CCAMForMask with a trained CCAM model
        
        Args:
            model: Trained CCamModel instance
        """
        self.model = model
        self.model.eval()
    
    def generate_mask(self, image_tensor, target_class=None, orig_size=None, threshold=0.4, relu=True):
        """
        Generate a pseudo-mask from an input image tensor using CCAM.
        Maintains the same interface as GradCAMForMask for compatibility.
        
        Args:
            image_tensor (Tensor): Image tensor [1, C, H, W]
            target_class (int or None): Not used in CCAM (class-agnostic)
            orig_size (tuple): Resize CAM to this (H, W) if given
            threshold (float): Value in [0, 1] to threshold the CAM
            relu (bool): Not used in CCAM (always applies sigmoid)
            
        Returns:
            cam_np: Raw CAM (H, W) as float
            binary_mask: Thresholded mask (H, W) as uint8
        """
        # Forward pass through CCAM model
        with torch.no_grad():
            fg_feats, bg_feats, ccam = self.model(image_tensor)
            
            # Get CAM
            cam = ccam.squeeze().cpu().numpy()
            
            # Normalize CAM (already normalized by sigmoid, but let's ensure ranges)
            cam = cam - cam.min()
            # cam_max = cam.max()
            # cam /= cam_max + (1e-6)
            
            # Resize if needed
            if orig_size is not None:
                cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                resized_cam = F.interpolate(cam_tensor, size=orig_size, mode='bilinear', align_corners=False)
                cam = resized_cam.squeeze().cpu().numpy()
            
            # Threshold to get binary mask
            binary_mask = (cam >= threshold).astype(np.uint8)
            
            return cam, binary_mask
    
    def save_mask(self, mask, save_path):
        """
        Save binary mask to a .png file using pathlib.Path
        
        Args:
            mask (np.ndarray): Binary mask (0 or 1)
            save_path (Path): Pathlib Path to save the .png mask
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)

# class CAMForMask:
#     def __init__(self, model, target_layer=None):
#         self.model = model.eval()
#         # 1) find target layer
#         if target_layer is None:
#             backbone = model.model if hasattr(model, "model") else model
#             target_layer = backbone.layer4[-1] 

#         self.fmaps = None
#         target_layer.register_forward_hook(self._save_fmap)

#         # 2) get weights
#         if   hasattr(model, "fc"):            
#             self.fc_weights = model.fc.weight.data
#         elif hasattr(model, "classifier"):
#             self.fc_weights = model.classifier.weight.data
#         elif hasattr(model, "model") and hasattr(model.model, "fc"):
#             self.fc_weights = model.model.fc.weight.data
#         else:
#             raise RuntimeError("❌ Cannot locate final linear layer (fc) in model")

#         self.fc_weights = self.fc_weights.cpu()

#     def _save_fmap(self, m, x, y):
#         # y: [B, C, H, W]
#         self.fmaps = y.detach()

#     @torch.no_grad()
#     def generate_mask(self, img_tensor, target_class=None,
#                       orig_size=None, threshold=0.4):
#         """
#         Args:
#             img_tensor: [1,C,H,W] 
#             target_class: None -> predicted class
#         Returns:
#             cam_np, bin_mask (H,W)
#         """
#         logits = self.model(img_tensor)
#         if target_class is None:
#             target_class = torch.argmax(logits, 1).item()
#         # 3) calculate CAM = w*F
#         weights = self.fc_weights[target_class].to(self.fmaps.device).view(-1)
#         cam     = torch.einsum("c,chw->hw", weights, self.fmaps[0])

#         cam = cam.clamp(min=0)                         # ReLU
#         cam -= cam.min()
#         if cam.max() > 0:
#             cam /= cam.max()

#         if orig_size is not None:
#             cam = F.interpolate(cam[None, None, ...], size=orig_size,
#                                 mode="bilinear", align_corners=False)[0,0]

#         cam_np = cam.cpu().numpy()
#         bin_mask = (cam_np >= threshold).astype(np.uint8)
#         return cam_np, bin_mask

#     def save_mask(self, mask, save_path: Path):
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         Image.fromarray(mask * 255).save(save_path)