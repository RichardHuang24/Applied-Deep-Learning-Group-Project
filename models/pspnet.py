"""
PSPNet (Pyramid Scene Parsing Network) implementation
for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging

logger = logging.getLogger(__name__)

class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing Module
    
    Creates a pyramid of different pooling levels and concatenates them
    to maintain both global and local context information.
    """
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        """
        Initialize PSP module
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels after bottleneck
            sizes: Pooling window sizes
        """
        super(PSPModule, self).__init__()
        
        # Create pyramid levels
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, size) for size in sizes
        ])
        
        # Bottleneck layer to reduce feature dimensions
        self.bottleneck = nn.Conv2d(
            in_channels * (1 + len(sizes)),
            out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_stage(self, in_channels, size):
        """Create a single pyramid stage"""
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_channels)
        relu = nn.ReLU(inplace=True)
        
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, x):
        """Forward pass through PSP module"""
        h, w = x.size(2), x.size(3)
        
        # Pass through pyramid stages
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages]
        
        # Concatenate with original features
        features = [x] + priors
        out = torch.cat(features, dim=1)
        
        # Pass through bottleneck
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class PSPNet(nn.Module):
    """
    PSPNet implementation with configurable backbone
    """
    def __init__(self, num_classes=2, backbone='resne50', pretrained=True):
        """
        Initialize PSPNet
        
        Args:
            num_classes: Number of output classes (including background)
            backbone: Backbone model name ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights for backbone
        """
        super(PSPNet, self).__init__()
        
        # Create backbone
        if backbone == 'resnet18':
            base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            layers = list(base_model.children())
            self.layer0 = nn.Sequential(*layers[:4])  # Initial layers before stride=2
            self.layer1 = layers[4]  # First residual block
            self.layer2 = layers[5]  # Second residual block
            self.layer3 = layers[6]  # Third residual block
            self.layer4 = layers[7]  # Fourth residual block
            feature_channels = 512
        elif backbone == 'resnet34':
            base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
            layers = list(base_model.children())
            self.layer0 = nn.Sequential(*layers[:4])
            self.layer1 = layers[4]
            self.layer2 = layers[5]
            self.layer3 = layers[6]
            self.layer4 = layers[7]
            feature_channels = 512
        elif backbone == 'resnet50':
            base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            layers = list(base_model.children())
            self.layer0 = nn.Sequential(*layers[:4])
            self.layer1 = layers[4]
            self.layer2 = layers[5]
            self.layer3 = layers[6]
            self.layer4 = layers[7]
            feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # PSP Module
        self.psp = PSPModule(
            in_channels=feature_channels,
            out_channels=512,
            sizes=(1, 2, 3, 6)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Auxiliary classification layer (optional, for deeper training supervision)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(feature_channels // 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        """Forward pass through PSPNet"""
        input_size = (x.size(2), x.size(3))
        
        # Extract features with backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux_out = x  # Save for auxiliary loss
        x = self.layer4(x)
        
        # Apply PSP module
        x = self.psp(x)
        
        # Apply classifier
        x = self.classifier(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # Apply auxiliary classifier if training
        if self.training:
            aux = self.aux_classifier(aux_out)
            aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return x, aux
        
        return x


def create_segmentation_model(config_path='config.json', backbone='resnet50'):
    """
    Factory function to create segmentation model
    
    Args:
        config_path: Path to configuration file
        backbone: Backbone model name
        
    Returns:
        PSPNet instance
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # num_classes = config['dataset']['num_classes'] + 1  # Add background class
    
    return PSPNet(
        num_classes=2,
        backbone=backbone,
        pretrained=True
    )