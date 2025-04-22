"""
Training functions for CCAM (Contrastive learning of Class-agnostic Activation Map)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

from .classifier import ResNetClassifier


logger = logging.getLogger(__name__)
class Disentangler(nn.Module):
    """
    Disentangler module that generates a class-agnostic activation map
    and uses it to extract foreground and background features
    """
    def __init__(self, channel_in):
        super(Disentangler, self).__init__()
        
        # Simple activation head: 3x3 conv followed by batch norm
        self.activation_head = nn.Conv2d(channel_in, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
    
    def forward(self, x):
        """
        Forward pass through the disentangler
        
        Args:
            x: Input features from backbone layer4
            
        Returns:
            fg_feats: Foreground features
            bg_feats: Background features
            ccam: Class-agnostic activation map
        """
        N, C, H, W = x.size()
        
        # Generate class-agnostic activation map (CCAM)
        ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))
        
        # Reshape for matrix multiplication
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        
        # Weighted pooling to extract foreground and background features
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam


class CCamModel(nn.Module):
    """
    Complete CCAM model for weakly supervised segmentation
    """
    def __init__(self, backbone='resnet50', initialization='imagenet', num_classes=37):
        super(CCamModel, self).__init__()
        self.backbone_name = backbone
        self.initialization = initialization
        
        # Create backbone using the existing ResNetClassifier
        self.backbone = ResNetClassifier(backbone, initialization, num_classes)
        
        # Determine feature dimensions based on backbone
        if backbone == 'resnet50':
            self.channel_dim = 2048 + 1024
        else:  # ResNet18 or ResNet34
            self.channel_dim = 512 + 256
        
        # Create disentangler
        self.disentangler = Disentangler(self.channel_dim)
        
        # Track layers initialized from scratch for parameter groups
        self.from_scratch_layers = [self.disentangler.activation_head, self.disentangler.bn_head]
    
    def forward(self, x):
        """
        Forward pass through CCAM model
        
        Returns:
            fg_feats: Foreground features
            bg_feats: Background features
            ccam: Class-agnostic activation map
        """
        # Get feature maps from backbone's layer4
        features = self.backbone.get_features(x)
        
        # Pass through disentangler to get foreground/background features and activation map
        fg_feats, bg_feats, ccam = self.disentangler(features)
        
        return fg_feats, bg_feats, ccam
    
    def get_parameter_groups(self):
        """
        Get parameter groups for optimizer (following the reference implementation)
        
        Returns:
            groups: Tuple of 4 parameter groups
                0: Backbone conv/norm weights
                1: Backbone conv/norm biases
                2: From-scratch conv/norm weights
                3: From-scratch conv/norm biases
        """
        groups = ([], [], [], [])
        
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or 
                isinstance(m, nn.BatchNorm2d)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        
        return groups

def cos_sim(embedded_fg, embedded_bg):
    """
    Compute cosine similarity with clamping
    """
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)

def cos_distance(embedded_fg, embedded_bg):
    """
    Compute cosine distance (1 - similarity)
    """
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return 1 - sim

def l2_distance(embedded_fg, embedded_bg):
    """
    Compute L2 distance
    """
    N, C = embedded_fg.size()
    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)
    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

class SimMinLoss(nn.Module):
    """
    Minimize Similarity, push representation of foreground and background apart
    """
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction
        
    def forward(self, embedded_bg, embedded_fg):
        """
        Args:
            embedded_bg: Background features [N, C]
            embedded_fg: Foreground features [N, C]
            
        Returns:
            loss: Negative log of (1 - similarity)
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError
            
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class SimMaxLoss(nn.Module):
    """
    Maximize Similarity, pull representation of same type together
    """
    def __init__(self, metric='cos', alpha=0.05, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, embedded_bg):
        """
        Args:
            embedded_bg: Features of the same type (bg or fg) [N, C]
            
        Returns:
            loss: Weighted negative log of similarity
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError
            
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# class SupervisionLoss(nn.Module):
#     """
#     Supervised segmentation loss using initial CAM as guidance for both foreground and background
#     """
#     def __init__(self, high_threshold=0.8, low_threshold=0.2):
#         super(SupervisionLoss, self).__init__()
#         self.high_threshold = high_threshold
#         self.low_threshold = low_threshold
        
#     def forward(self, ccam, initial_cam):
#         """
#         Args:
#             ccam: CCAM output (B, 1, H, W)
#             initial_cam: Initial CAM from CAM/GradCAM (B, 1, H, W)
            
#         Returns:
#             loss: MSE loss for both foreground and background regions
#         """
#         # Create masks for high confidence foreground and background regions
#         fg_mask = (initial_cam > self.high_threshold)
#         bg_mask = (initial_cam < self.low_threshold)
        
#         # Calculate loss for foreground regions
#         fg_loss = 0
#         if torch.sum(fg_mask) > 0:
#             fg_loss = F.mse_loss(ccam[fg_mask], initial_cam[fg_mask])
        
#         # Calculate loss for background regions
#         bg_loss = 0
#         if torch.sum(bg_mask) > 0:
#             bg_loss = F.mse_loss(ccam[bg_mask], initial_cam[bg_mask])
        
#         # Combined loss
#         loss = fg_loss + bg_loss
        
#         return loss

class SupervisionLoss(nn.Module):
    """
    Supervised segmentation loss using initial CAM as guidance
    Uses cross-entropy loss on high-confidence regions (>0.8 as foreground, <0.1 as background)
    """
    def __init__(self, high_threshold=0.8, low_threshold=0.1):
        super(SupervisionLoss, self).__init__()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
    def forward(self, ccam, initial_cam):
        """
        Args:
            ccam: CCAM output (B, 1, H, W) - single channel probability map [0,1]
            initial_cam: Initial CAM from CAM/GradCAM (B, 1, H, W)
            
        Returns:
            loss: Cross-entropy loss for high confidence regions
        """
        # Create binary labels from initial CAM
        # 1 for foreground (initial_cam > high_threshold)
        # 0 for background (initial_cam < low_threshold)
        fg_mask = (initial_cam > self.high_threshold)
        bg_mask = (initial_cam < self.low_threshold)
        
        # Combined mask for regions we want to include in the loss
        valid_mask = fg_mask | bg_mask
        
        # If no high confidence regions, return zero loss
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=ccam.device)
        
        # Create target labels: 1 for foreground, 0 for background
        target = torch.zeros_like(initial_cam)
        target[fg_mask] = 1.0
        
        # Extract valid regions from ccam and target
        ccam_valid = ccam[valid_mask].squeeze()
        target_valid = target[valid_mask].squeeze()
        
        # Apply binary cross-entropy loss
        # We use BCELoss since ccam is already a probability map [0,1]
        loss = F.binary_cross_entropy(ccam_valid, target_valid)
        
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_ccam(config, model, train_loader, criterion, optimizer, scheduler, num_epochs, 
              output_dir, initial_cams_dir=None, device='cuda'):
    """
    Train CCAM model
    
    Args:
        config: Configuration dictionary
        model: CCAM model
        train_loader: DataLoader for training data
        criterion: List of criterion [bg_bg_loss, bg_fg_loss, fg_fg_loss, supervision_loss]
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        output_dir: Directory to save checkpoints and visualizations
        initial_cams_dir: Directory containing initial CAMs (for supervision)
        device: Device to train on
    
    Returns:
        Trained model
    """
    # Create output directories
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "visualizations"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger.info(f"Training CCAM for {num_epochs} epochs")
    logger.info(f"Saving checkpoints to {checkpoint_dir}")
    logger.info(f"Saving visualizations to {vis_dir}")
    
    # Flag to track CAM orientation (foreground bright or dark)
    invert_cam = False
    
    # Training loop
    for epoch in range(num_epochs):
        # Set up averagemeters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_bg_bg = AverageMeter()
        losses_bg_fg = AverageMeter()
        losses_fg_fg = AverageMeter()
        losses_sup = AverageMeter()
        
        # Switch to train mode
        model.train()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Record time
        end = time.time()
        
        # Training step
        for i, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 4:  # If batch contains image, label, class_name, image_name
                inputs, labels, class_names, img_names = batch
            elif len(batch) == 3:  # If batch contains image, label, image_name
                inputs, labels, img_names = batch
                class_names = None
            else:  # Default case
                inputs, labels = batch
                class_names = None
                img_names = None
            
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Load initial CAM if available (for supervision)
            initial_cams = None
            if initial_cams_dir is not None and img_names is not None:
                initial_cams = []
                for img_name in img_names:
                    # Construct path to initial CAM
                    img_name_base = img_name.split('.')[0]
                    cam_path = Path(initial_cams_dir) / f"{img_name_base}.npy"
                    if cam_path.exists():
                        # Load CAM
                        cam = np.load(cam_path)
                        initial_cams.append(cam)
                    else:
                        # Use placeholder
                        cam = np.zeros((inputs.shape[2] // 16, inputs.shape[3] // 16))  # Reduced size
                        initial_cams.append(cam)
                
                # Convert to tensor
                initial_cams = torch.tensor(np.array(initial_cams), dtype=torch.float32).to(device)
                initial_cams = initial_cams.unsqueeze(1)  # Add channel dimension
                
                # Resize to match CCAM output size
                with torch.no_grad():
                    # Get dummy forward pass to determine CCAM size
                    _, _, dummy_ccam = model(inputs[:1])
                    h, w = dummy_ccam.shape[2], dummy_ccam.shape[3]
                    
                    # Resize initial CAMs to match CCAM size
                    initial_cams = F.interpolate(initial_cams, size=(h, w), mode='bilinear', align_corners=False)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            fg_feats, bg_feats, ccam = model(inputs)
            
            # Compute losses
            # 1. Background-Background similarity (maximize)
            loss_bg_bg = criterion[0](bg_feats)
            # 2. Background-Foreground similarity (minimize)
            loss_bg_fg = criterion[1](bg_feats, fg_feats)
            # 3. Foreground-Foreground similarity (maximize)
            loss_fg_fg = criterion[2](fg_feats)
            
            # Sum contrastive losses
            loss = loss_bg_bg + loss_bg_fg + loss_fg_fg
            
            # Add supervision loss if initial CAMs available
            loss_sup = torch.tensor(0.0, device=device)
            if initial_cams is not None and len(criterion) > 3:
                loss_sup = criterion[3](ccam, initial_cams)
                loss += loss_sup * 0.5
            
            # Backward pass and update
            loss.backward()
            optimizer.step()
            
            # Update averagemeters
            losses.update(loss.item(), inputs.size(0))
            losses_bg_bg.update(loss_bg_bg.item(), inputs.size(0))
            losses_bg_fg.update(loss_bg_fg.item(), inputs.size(0))
            losses_fg_fg.update(loss_fg_fg.item(), inputs.size(0))
            if initial_cams is not None and len(criterion) > 3:
                losses_sup.update(loss_sup.item(), inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.val:.4f} ({losses.avg:.4f})",
                'BG-BG': f"{losses_bg_bg.val:.4f}",
                'BG-FG': f"{losses_bg_fg.val:.4f}",
                'FG-FG': f"{losses_fg_fg.val:.4f}",
                'SUP': f"{losses_sup.val:.4f}",
                'time': f"{batch_time.val:.3f}s"
            })
            
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        # torch.save({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict() if scheduler is not None else None,
        #     'invert_cam': invert_cam
        # }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save latest checkpoint (overwrite)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'invert_cam': invert_cam
        }, checkpoint_dir / "checkpoint_latest.pth")
        
        # Log epoch stats
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Loss: {losses.avg:.4f}, "
                   f"BG-BG: {losses_bg_bg.avg:.4f}, "
                   f"BG-FG: {losses_bg_fg.avg:.4f}, "
                   f"FG-FG: {losses_fg_fg.avg:.4f}, "
                   f"SUP: {losses_sup.avg:.4f}")
    
    # Return trained model
    return model