"""
Visualization tools for weakly-supervised segmentation using PIL
"""
import os
import torch
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

def create_color_palette(num_classes=38):
    """
    Create a color palette for visualization
    
    Args:
        num_classes: Number of classes to generate colors for
        
    Returns:
        List of RGB tuples for each class
    """
    # Start with black for background
    palette = [(0, 0, 0)]
    
    # Generate visually distinct colors for classes
    for i in range(1, num_classes):
        # Use a hue-based approach for color generation
        hue = (i * 37) % 360  # Use prime number to get better distribution
        
        # Convert HSV to RGB (simplified conversion)
        h = hue / 60.0
        s = 0.8
        v = 0.9
        
        sector = int(h) % 6
        f = h - int(h)
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if sector == 0:
            r, g, b = v, t, p
        elif sector == 1:
            r, g, b = q, v, p
        elif sector == 2:
            r, g, b = p, v, t
        elif sector == 3:
            r, g, b = p, q, v
        elif sector == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return palette

# Create a global color palette
COLOR_PALETTE = create_color_palette()

def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image
    
    Args:
        tensor: Input tensor (C,H,W) or (H,W,C)
        
    Returns:
        PIL.Image: Converted image
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor  # Already a PIL image
        
    if tensor.dim() == 4:
        # Take the first sample if a batch is provided
        tensor = tensor[0]
        
    # Handle different tensor formats
    if tensor.dim() == 3:
        # Check if tensor is in (C,H,W) format
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            tensor = tensor.permute(1, 2, 0)  # Convert to (H,W,C)
        
        # Normalize if needed
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
            
        # Convert to PIL
        if tensor.shape[2] == 1:
            # Single channel
            tensor = tensor.squeeze(2)
            return Image.fromarray(tensor.cpu().numpy(), mode='L')
        else:
            # RGB
            return Image.fromarray(tensor.cpu().numpy())
    
    elif tensor.dim() == 2:
        # Single channel (H,W)
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
            
        return Image.fromarray(tensor.cpu().numpy(), mode='L')
    
    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")

def create_mask_overlay(image, mask, color_palette=None):
    """
    Create a colored mask overlay on an image
    
    Args:
        image: PIL Image
        mask: Tensor with class indices
        color_palette: List of RGB tuples for classes
        
    Returns:
        PIL.Image: Image with mask overlay
    """
    if color_palette is None:
        color_palette = COLOR_PALETTE
        
    # Convert to PIL if needed
    image_pil = tensor_to_pil(image)
    width, height = image_pil.size
    
    # Convert mask to numpy array of class indices
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Create a new transparent image for the mask
    mask_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    
    # Draw each class with its color
    for y in range(height):
        for x in range(width):
            cls_idx = mask[y, x]
            if cls_idx > 0:  # Skip background
                color = color_palette[int(cls_idx)] if int(cls_idx) < len(color_palette) else (255, 0, 0)
                mask_draw.point((x, y), fill=(*color, 128))  # 128 for semi-transparency
                
    # Convert original image to RGBA and blend
    if image_pil.mode != "RGBA":
        image_pil = image_pil.convert("RGBA")
        
    # Blend the images
    return Image.alpha_composite(image_pil, mask_image).convert("RGB")

def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None, title=None):
    """
    Visualize prediction using PIL
    
    Args:
        image: Image tensor or PIL image
        pred_mask: Predicted mask tensor or PIL image
        gt_mask: Ground truth mask tensor or PIL image (optional)
        save_path: Path to save visualization
        title: Title for the visualization
        
    Returns:
        PIL.Image: Visualization image
    """
    # Convert to PIL images
    image_pil = tensor_to_pil(image)
    
    # Convert prediction mask
    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.dim() == 4:  # (B, C, H, W)
            pred_mask = torch.argmax(pred_mask[0], dim=0)  # Convert to class indices
        elif pred_mask.dim() == 3 and pred_mask.shape[0] > 1:  # (C, H, W)
            pred_mask = torch.argmax(pred_mask, dim=0)  # Convert to class indices
        else:
            pred_mask = pred_mask.squeeze()  # Already class indices
        
        pred_mask = pred_mask.cpu().numpy()
    
    # Create prediction overlay
    pred_overlay = create_mask_overlay(image_pil, pred_mask)
    
    # Determine layout
    if gt_mask is not None:
        # Convert ground truth mask
        if isinstance(gt_mask, torch.Tensor):
            if gt_mask.dim() == 4:  # (B, C, H, W)
                gt_mask = torch.argmax(gt_mask[0], dim=0)
            elif gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
                gt_mask = torch.argmax(gt_mask, dim=0)
            else:
                gt_mask = gt_mask.squeeze()
                
            gt_mask = gt_mask.cpu().numpy()
            
        # Create ground truth overlay
        gt_overlay = create_mask_overlay(image_pil, gt_mask)
        
        # Create a grid with all images
        width, height = image_pil.size
        result = Image.new("RGB", (width * 3, height))
        result.paste(image_pil, (0, 0))
        result.paste(pred_overlay, (width, 0))
        result.paste(gt_overlay, (width * 2, 0))
        
        # Add labels
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        draw.text((width + 10, 10), "Prediction", fill=(255, 255, 255), font=font)
        draw.text((width * 2 + 10, 10), "Ground Truth", fill=(255, 255, 255), font=font)
        
        if title:
            draw.text((width, height - 30), title, fill=(255, 255, 255), font=font)
            
    else:
        # Just show original and prediction
        width, height = image_pil.size
        result = Image.new("RGB", (width * 2, height))
        result.paste(image_pil, (0, 0))
        result.paste(pred_overlay, (width, 0))
        
        # Add labels
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        draw.text((width + 10, 10), "Prediction", fill=(255, 255, 255), font=font)
        
        if title:
            draw.text((width // 2, height - 30), title, fill=(255, 255, 255), font=font)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)
    
    return result

def visualize_cam(image, cam, mask=None, save_path=None):
    """
    Visualize CAM heatmap overlay on image
    
    Args:
        image: Image tensor or PIL image
        cam: CAM tensor or numpy array
        mask: Binary mask tensor or numpy array (optional)
        save_path: Path to save visualization
        
    Returns:
        PIL.Image: Visualization image
    """
    # Convert to PIL image
    image_pil = tensor_to_pil(image)
    width, height = image_pil.size
    
    # Convert CAM to tensor
    if not isinstance(cam, torch.Tensor):
        cam = torch.tensor(cam)
    
    # Resize CAM to match image size
    if cam.shape[0] != height or cam.shape[1] != width:
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Normalize CAM
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    
    # Create heatmap
    cam_np = (cam * 255).to(torch.uint8).cpu().numpy()
    heatmap = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            # Create a red gradient (value determines intensity)
            val = cam_np[y, x]
            heatmap.putpixel((x, y), (val, 0, 0))
    
    # Blend with original image
    result = Image.blend(image_pil, heatmap, 0.5)
    
    # Show mask overlay if provided
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Create a binary mask image
        mask_img = Image.new("RGB", (width, height))
        for y in range(height):
            for x in range(width):
                if mask[y, x] > 0:
                    mask_img.putpixel((x, y), (0, 255, 0))  # Green for the mask
        
        # Create mask overlay
        mask_overlay = Image.blend(image_pil, mask_img, 0.3)
        
        # Create final composite
        final = Image.new("RGB", (width * 3, height))
        final.paste(image_pil, (0, 0))
        final.paste(result, (width, 0))
        final.paste(mask_overlay, (width * 2, 0))
        
        # Add labels
        draw = ImageDraw.Draw(final)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        draw.text((width + 10, 10), "CAM", fill=(255, 255, 255), font=font)
        draw.text((width * 2 + 10, 10), "Mask", fill=(255, 255, 255), font=font)
        
        result = final
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)
    
    return result

def visualize_results_grid(images, preds, gt_masks=None, save_path=None, show_gt=True):
    """
    Visualize multiple results in a grid
    
    Args:
        images: List of image tensors or PIL images
        preds: List of prediction mask tensors
        gt_masks: List of ground truth mask tensors (optional)
        save_path: Path to save visualization
        show_gt: Whether to show ground truth
        
    Returns:
        PIL.Image: Grid visualization
    """
    num_samples = len(images)
    
    # Determine grid dimensions
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    # Check how many images per sample (1, 2, or 3)
    subcols = 2  # Original and prediction
    if show_gt and gt_masks is not None:
        subcols = 3  # Original, prediction, and ground truth
    
    # Get dimensions of first image
    first_image = tensor_to_pil(images[0])
    img_width, img_height = first_image.size
    
    # Create the grid
    grid_width = cols * subcols * img_width
    grid_height = rows * img_height
    grid = Image.new("RGB", (grid_width, grid_height))
    
    # Fill the grid
    for idx, (image, pred) in enumerate(zip(images, preds)):
        row = idx // cols
        col = idx % cols
        
        # Convert to PIL
        image_pil = tensor_to_pil(image)
        
        # Process prediction
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 4:  # (B, C, H, W)
                pred = torch.argmax(pred[0], dim=0)
            elif pred.dim() == 3 and pred.shape[0] > 1:
                pred = torch.argmax(pred, dim=0)
            else:
                pred = pred.squeeze()
                
            pred = pred.cpu().numpy()
        
        # Create prediction overlay
        pred_overlay = create_mask_overlay(image_pil, pred)
        
        # Place in grid
        x_offset = col * subcols * img_width
        y_offset = row * img_height
        
        # Place original image
        grid.paste(image_pil, (x_offset, y_offset))
        
        # Place prediction
        grid.paste(pred_overlay, (x_offset + img_width, y_offset))
        
        # Place ground truth if available
        if show_gt and gt_masks is not None:
            gt_mask = gt_masks[idx]
            
            # Process ground truth
            if isinstance(gt_mask, torch.Tensor):
                if gt_mask.dim() == 4:
                    gt_mask = torch.argmax(gt_mask[0], dim=0)
                elif gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
                    gt_mask = torch.argmax(gt_mask, dim=0)
                else:
                    gt_mask = gt_mask.squeeze()
                    
                gt_mask = gt_mask.cpu().numpy()
            
            # Create ground truth overlay
            gt_overlay = create_mask_overlay(image_pil, gt_mask)
            
            # Place in grid
            grid.paste(gt_overlay, (x_offset + 2 * img_width, y_offset))
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid.save(save_path)
    
    return grid