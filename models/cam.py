"""
Class Activation Map (CAM) implementations:
- Grad-CAM
- CCAM (Cross-Channel Class Activation Mapping)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from PIL import Image
from pathlib import Path

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
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)  # shape [1, H, W]

        if relu:
            cam = F.relu(cam)

        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max

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
    Vanilla CAM (Zhou et al. 2016) 
    “Conv backbone + Global‑Pooling + FC” 
    """
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        # 1) find target layer
        if target_layer is None:
            backbone = model.model if hasattr(model, "model") else model
            target_layer = backbone.layer4[-1] 

        self.fmaps = None
        target_layer.register_forward_hook(self._save_fmap)

        # 2) get weights
        if   hasattr(model, "fc"):            
            self.fc_weights = model.fc.weight.data
        elif hasattr(model, "classifier"):
            self.fc_weights = model.classifier.weight.data
        elif hasattr(model, "model") and hasattr(model.model, "fc"):
            self.fc_weights = model.model.fc.weight.data
        else:
            raise RuntimeError("❌ Cannot locate final linear layer (fc) in model")

        self.fc_weights = self.fc_weights.cpu()

    def _save_fmap(self, m, x, y):
        # y: [B, C, H, W]
        self.fmaps = y.detach()

    @torch.no_grad()
    def generate_mask(self, img_tensor, target_class=None,
                      orig_size=None, threshold=0.4):
        """
        Args:
            img_tensor: [1,C,H,W] 
            target_class: None -> predicted class
        Returns:
            cam_np, bin_mask (H,W)
        """
        logits = self.model(img_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, 1).item()
        # 3) calculate CAM = w*F
        weights = self.fc_weights[target_class].to(self.fmaps.device).view(-1)
        cam     = torch.einsum("c,chw->hw", weights, self.fmaps[0])

        cam = cam.clamp(min=0)                         # ReLU
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        if orig_size is not None:
            cam = F.interpolate(cam[None, None, ...], size=orig_size,
                                mode="bilinear", align_corners=False)[0,0]

        cam_np = cam.cpu().numpy()
        bin_mask = (cam_np >= threshold).astype(np.uint8)
        return cam_np, bin_mask

    def save_mask(self, mask, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask * 255).save(save_path)