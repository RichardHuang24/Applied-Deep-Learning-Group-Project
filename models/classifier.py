"""
ResNet classifier models with different backbones and initialization options
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import torchvision.models as models
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)

class ResNetClassifier(nn.Module):
    """
    ResNet classifier with configurable backbone and initialization
    
    Supports:
    - Backbones: ResNet18, ResNet34, ResNet50
    - Initializations: ImageNet, Random, SimCLR
    """
    def __init__(self, backbone='resnet50', initialization='imagenet', num_classes=37):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.initialization = initialization.lower()

        self.model = self._create_backbone()
        self._initialize_weights()

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def _create_backbone(self):
        backbones = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }

        if self.backbone_name not in backbones:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Always load ImageNet weights first for compatibility
        if self.initialization == 'imagenet':
            model = backbones[self.backbone_name](weights='DEFAULT')
        else:
            # Random init or simclr will be handled later
            model = backbones[self.backbone_name](weights=None)
        
        # Modify layer4 to use stride=1 (no downsampling)
        self._modify_layer4_stride(model)
        
        return model
    
    def _modify_layer4_stride(self, model):
        """
        Modify the stride of the first conv layer in layer4 to be 1 instead of 2
        This prevents downsampling in layer4
        """
        # For ResNet models, change the stride in the first Bottleneck/BasicBlock of layer4
        if hasattr(model, 'layer4') and len(model.layer4) > 0:
            # Get the first block of layer4
            first_block = model.layer4[0]
            
            # Modify the stride in the first conv layer (for BasicBlock in ResNet18/34)
            if hasattr(first_block, 'conv1'):
                first_block.conv1.stride = (1, 1)
            
            # For Bottleneck architecture (ResNet50 and higher)
            if hasattr(first_block, 'conv2'):
                first_block.conv2.stride = (1, 1)
            
            # Also modify the downsampling layer if it exists
            if hasattr(first_block, 'downsample') and first_block.downsample is not None:
                if len(first_block.downsample) > 0:
                    # The first module in the downsample sequential is usually the conv layer
                    if isinstance(first_block.downsample[0], nn.Conv2d):
                        first_block.downsample[0].stride = (1, 1)

    def _initialize_weights(self):
        if self.initialization == 'simclr':
            raise NotImplementedError("SimCLR initialization not implemented")
        elif self.initialization == 'random':
            self.model.apply(self._init_weights)
        elif self.initialization == 'mocov2':
            self._load_moco_weights()
        elif self.initialization == 'imagenet':
            pass  # Already loaded in _create_backbone
        else:
            raise ValueError(f"Unsupported initialization: {self.initialization}")
    
    def _load_moco_weights(self):
        """
        Load MoCo v2 pretrained weights (ResNet-50 only officially).
        Automatically downloads if not found.
        """
        
        # Checkpoint URL (MoCo v2 ResNet-50, 800 epochs)
        url = "https://dl.fbaipublicfiles.com/moco/moco_v2_800ep_pretrain.pth.tar"
        local_path = Path(f"pretrained/mocov2_{self.backbone_name}.pth")

        if not local_path.exists():
            logger.info("Downloading MoCo v2 pretrained weights...")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Direct download from official Facebook Research
            url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
            urllib.request.urlretrieve(url, local_path)
            logger.info(f"Downloaded MoCo v2 weights to {local_path}")

        if self.backbone_name != "resnet50":
            raise ValueError(f"No official MoCo v2 weights available for {self.backbone_name}. Using random init.")

        # Load checkpoint
        state_dict = torch.load(local_path, map_location="cpu")["state_dict"]

        # Clean state dict: remove `module.encoder_q.` prefix
        cleaned_dict = {
            k.replace("module.encoder_q.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.encoder_q") and "fc" not in k
        }

        # Load weights
        missing, unexpected = self.model.load_state_dict(cleaned_dict, strict=False)
        logger.info(f"Loaded MoCo v2 weights for {self.backbone_name}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x1 = self.model.layer3(x)
        x2 = self.model.layer4(x1)
        
        return torch.cat([x2, x1], dim=-3)

class ResNetCAM(ResNetClassifier):
    """
    ResNet with Global Average Pooling for Class Activation Mapping (CAM)
    
    This architecture modifies the standard ResNet to follow the original CAM paper's
    approach, where the final convolutional layer is directly connected to the 
    classification layer via global average pooling.
    """
    def __init__(self, backbone='resnet50', initialization='imagenet', num_classes=37):
        super().__init__(backbone, initialization, num_classes)
        
        # Replace the average pooling with an explicit global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Extract features
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)
        
        # Global Average Pooling
        pooled = self.global_pool(features).view(features.size(0), -1)
        
        # Classification
        logits = self.model.fc(pooled)
        
        return logits
    
    def get_cam_weights(self):
        """
        Returns the weights of the final classification layer.
        These weights are used to compute the CAM.
        """
        return self.model.fc.weight.data
    
    def get_activations(self, x):
        """
        Extract feature map activations from the last convolutional layer.
        
        Args:
            x (Tensor): Input image tensor
            
        Returns:
            Tensor: Feature maps from the last conv layer
        """
        return self.get_features(x)


def create_classifier(config, experiment=None):
    """
    Factory function to create classifier model
    
    Args:
        config (dict): Configuration dictionary
        experiment (str): Experiment name from config (if None, use default settings)
    
    Returns:
        ResNetClassifier or ResNetCAM: Configured classifier model
    """
    
    num_classes = config['dataset']['num_classes']
    print(f"Experiment: {experiment}")
    # Get backbone and initialization based on experiment or defaults
    if experiment:
        parts = experiment.split('_')
        backbone = parts[0]
        initialization = parts[1] if len(parts) > 1 else 'imagenet'
        method = parts[2] if len(parts) > 2 else 'cam'  # cam, gradcam
        print(f"Using experiment settings: Backbone={backbone}, Initialization={initialization}, Method={method}")
    else:
        # Use defaults
        backbone = config['models']['classifier']['default']['backbone']
        initialization = config['models']['classifier']['default']['initialization']
        method = 'cam'
    
    # Create and return the appropriate classifier based on method
    if method.lower() == 'cam' or method.lower() == 'cam+ccam':
        return ResNetCAM(
            backbone=backbone,
            initialization=initialization,
            num_classes=num_classes
        )
    elif method.lower() == 'gradcam' or method.lower() == 'gradcam+ccam':
        return ResNetClassifier(
            backbone=backbone,
            initialization=initialization,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'cam' or 'gradcam'.")
    

if __name__ == "__main__":
    print(ResNetClassifier())