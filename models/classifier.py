"""
ResNet classifier models with different backbones and initialization options
"""
import os
import torch
import torch.nn as nn
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
    def __init__(self, backbone='resnet18', initialization='imagenet', num_classes=37):
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
            return backbones[self.backbone_name](weights='DEFAULT')
        else:
            # Random init or simclr will be handled later
            return backbones[self.backbone_name](weights=None)

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

    def get_features(self, x, layer='layer4'):
        """
        Extract features from a specific intermediate layer.
        Currently supports 'layer4' only.
        """
        if layer != 'layer4':
            raise ValueError(f"Layer {layer} not supported")

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)
        return features

def create_classifier(config, experiment=None):
    """
    Factory function to create classifier model
    
    Args:
        config_path (str): Path to configuration file
        experiment (str): Experiment name from config (if None, use default settings)
    
    Returns:
        ResNetClassifier: Configured classifier model
    """

    
    num_classes = config['dataset']['num_classes']
    
    # Get backbone and initialization based on experiment or defaults
    if experiment:
        parts = experiment.split('_')
        backbone = parts[0]
        initialization = parts[1] if len(parts) > 1 else 'imagenet'
        print(f"Using experiment settings: Backbone={backbone}, Initialization={initialization}")
    else:
        # Use defaults
        backbone = config['models']['classifier']['default']['backbone']
        initialization = config['models']['classifier']['default']['initialization']
    
    # Create and return the classifier
    return ResNetClassifier(
        backbone=backbone,
        initialization=initialization,
        num_classes=num_classes
    )
