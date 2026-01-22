"""
Image detector model architecture for submission.
This file defines the model architecture and load_model() function required by gasbench.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


class ImageELAPRNUDetector(nn.Module):
    """CNN model for ELA+PRNU fusion features for image detection.
    
    Input: [B, 4, H, W] where channels are [ELA_R, ELA_G, ELA_B, PRNU]
    Output: [B, num_classes] logits for [real, synthetic, semisynthetic]
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 4):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # Input: [B, C, H, W] where C=4 (ELA+PRNU)
        x = self.backbone(x)
        x = x.flatten(1)  # [B, 128]
        x = self.classifier(x)  # [B, num_classes]
        return x


def load_model(weights_path: str, num_classes: int = 3) -> nn.Module:
    """Required entry point - called by gasbench.
    
    Args:
        weights_path: Path to the .safetensors file
        num_classes: Number of output classes from config (should be 3: real, synthetic, semisynthetic)
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    model = ImageELAPRNUDetector(num_classes=num_classes, input_channels=4)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model
