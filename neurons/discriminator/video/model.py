"""
Video detector model architecture for submission.
This file defines the model architecture and load_model() function required by gasbench.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


class VideoELAPRNUDetector(nn.Module):
    """3D CNN model for ELA+PRNU fusion features for video detection.
    
    Input: [B, T, 4, H, W] where T is number of frames, channels are [ELA_R, ELA_G, ELA_B, PRNU]
    Output: [B, num_classes] logits for [real, synthetic, semisynthetic]
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 4, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames
        
        # 3D CNN backbone for temporal-spatial features
        self.backbone = nn.Sequential(
            # Block 1: 3D conv
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 2: 3D conv
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 3: 3D conv
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # Input: [B, T, C, H, W] -> [B, C, T, H, W] for Conv3d
        if x.dim() == 5 and x.shape[1] != 4:
            # Assume format is [B, T, C, H, W], convert to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        # Input: [B, C, T, H, W] where C=4 (ELA+PRNU)
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
    model = VideoELAPRNUDetector(num_classes=num_classes, input_channels=4, num_frames=8)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    return model
