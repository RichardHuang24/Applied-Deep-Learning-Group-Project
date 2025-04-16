"""
PSPNet (Pyramid Scene Parsing Network) implementation
for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)

logger = logging.getLogger(__name__)

class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing Module
    """
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, size) for size in sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels * (1 + len(sizes)), out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, in_channels, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages]
        out = torch.cat([x] + priors, dim=1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class PSPNet(nn.Module):
    """
    PSPNet for semantic segmentation with configurable ResNet backbone
    """
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super(PSPNet, self).__init__()

        # Backbone selection with weights handling
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            feature_channels = 512
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base_model = resnet34(weights=weights)
            feature_channels = 512
        elif backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model = resnet50(weights=weights)
            feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract layers from backbone
        layers = list(base_model.children())
        self.layer0 = nn.Sequential(*layers[:4])  # Conv1 + BN + ReLU + MaxPool
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        # PSP Module
        self.psp = PSPModule(in_channels=feature_channels, out_channels=512)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Auxiliary classifier for deep supervision
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(feature_channels // 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = (x.size(2), x.size(3))

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux = x  # for auxiliary loss
        x = self.layer4(x)

        x = self.psp(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        if self.training:
            aux_out = self.aux_classifier(aux)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
            return x, aux_out

        return x


def create_segmentation_model(backbone='resnet50', pretrained=True, num_classes=2):
    """
    Factory function to create a PSPNet model.
    
    Args:
        backbone (str): ResNet backbone
        pretrained (bool): Load ImageNet weights or not
        num_classes (int): Number of segmentation classes
    
    Returns:
        PSPNet model
    """
    # Optional: load num_classes from config if needed

    return PSPNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
