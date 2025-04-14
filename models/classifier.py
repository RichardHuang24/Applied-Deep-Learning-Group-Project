"""
ResNet classifier models with different backbones and initialization options
"""
import os
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ResNetClassifier(nn.Module):
    """
    ResNet classifier with configurable backbone and initialization
    
    Supports:
    - Backbones: ResNet18, ResNet34, ResNet50
    - Initializations: ImageNet, Random, SimCLR
    """
    def __init__(self, backbone='resnet18', initialization='imagenet', num_classes=37):
        """
        Initialize the ResNet classifier
        
        Args:
            backbone (str): ResNet backbone ('resnet18', 'resnet34', 'resnet50')
            initialization (str): Initialization method ('imagenet', 'random', 'simclr')
            num_classes (int): Number of output classes
        """
        super(ResNetClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.initialization = initialization
        
        # Create backbone
        if backbone == 'resnet18':
            if initialization == 'imagenet':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            else:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        elif backbone == 'resnet34':
            if initialization == 'imagenet':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
            else:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        elif backbone == 'resnet50':
            if initialization == 'imagenet':
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            else:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Load SimCLR weights if specified
        if initialization == 'simclr':
            self._load_simclr_weights()
        
        # Replace final FC layer for the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def _load_simclr_weights(self):
        """Load SimCLR pre-trained weights if available"""
        simclr_path = Path(f"pretrained/simclr_{self.backbone_name}.pth")
        if simclr_path.exists():
            state_dict = torch.load(simclr_path, map_location='cpu')
            # Remove projection head weights, keep only encoder
            encoder_state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('projection_head')}
            
            # Load weights
            missing, unexpected = self.model.load_state_dict(encoder_state_dict, strict=False)
            logger.info(f"Loaded SimCLR weights for {self.backbone_name}")
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")
        else:
            logger.warning(f"SimCLR weights not found at {simclr_path}. Using random initialization.")
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def get_features(self, x, layer='layer4'):
        """
        Extract features from intermediate layers
        
        Args:
            x: Input tensor
            layer: Layer name to extract features from
        
        Returns:
            Feature maps from the specified layer
        """
        if layer == 'layer4':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            features = self.model.layer4(x)
            return features
        else:
            # Other intermediate layers could be implemented here
            raise ValueError(f"Layer {layer} not supported")

def create_classifier(config_path, experiment=None):
    """
    Factory function to create classifier model
    
    Args:
        config_path (str): Path to configuration file
        experiment (str): Experiment name from config (if None, use default settings)
    
    Returns:
        ResNetClassifier: Configured classifier model
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['dataset']['num_classes']
    
    # Get backbone and initialization based on experiment or defaults
    if experiment and experiment in [exp['name'] for exp in config['experiments']['classifier']]:
        # Find the specific experiment
        for exp in config['experiments']['classifier']:
            if exp['name'] == experiment:
                backbone = exp['backbone']
                initialization = exp['initialization']
                break
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