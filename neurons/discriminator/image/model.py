"""
Image detector model architecture for submission.
This file defines the model architecture and load_model() function required by gasbench.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
import numpy as np
import cv2
import pywt
from scipy.signal import wiener
import warnings


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


def extract_prnu_enhanced(rgb: np.ndarray) -> np.ndarray:
    """Extract PRNU (Photo Response Non-Uniformity) features using wavelet denoising."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    
    # Wavelet decomposition for denoising
    try:
        coeffs = pywt.wavedec2(gray, 'db4', level=2)
        cA = coeffs[0]
        details = []
        for level_coeffs in coeffs[1:]:
            cH, cV, cD = level_coeffs
            # Use wiener filter with error handling for division by zero
            # Suppress division warnings from scipy.signal.wiener
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in divide.*')
                try:
                    cH_denoised = wiener(cH, mysize=5)
                    cV_denoised = wiener(cV, mysize=5)
                    cD_denoised = wiener(cD, mysize=5)
                    # Check for NaN/Inf and replace with median filter result if needed
                    if np.any(np.isnan(cH_denoised)) or np.any(np.isinf(cH_denoised)):
                        cH_denoised = cv2.medianBlur(cH.astype(np.float32), 5)
                    if np.any(np.isnan(cV_denoised)) or np.any(np.isinf(cV_denoised)):
                        cV_denoised = cv2.medianBlur(cV.astype(np.float32), 5)
                    if np.any(np.isnan(cD_denoised)) or np.any(np.isinf(cD_denoised)):
                        cD_denoised = cv2.medianBlur(cD.astype(np.float32), 5)
                except (RuntimeWarning, ValueError):
                    # Fallback to median filter if wiener fails
                    cH_denoised = cv2.medianBlur(cH.astype(np.float32), 5)
                    cV_denoised = cv2.medianBlur(cV.astype(np.float32), 5)
                    cD_denoised = cv2.medianBlur(cD.astype(np.float32), 5)
            
            # Check for NaN/Inf values
            cH_denoised = np.nan_to_num(cH_denoised, nan=0.0, posinf=1.0, neginf=-1.0)
            cV_denoised = np.nan_to_num(cV_denoised, nan=0.0, posinf=1.0, neginf=-1.0)
            cD_denoised = np.nan_to_num(cD_denoised, nan=0.0, posinf=1.0, neginf=-1.0)
            
            details.extend([cH_denoised, cV_denoised, cD_denoised])
        
        reconstructed_coeffs = [cA] + [tuple(details[i:i+3]) for i in range(0, len(details), 3)]
        denoised = pywt.waverec2(reconstructed_coeffs, 'db4')
        
        if denoised.shape != gray.shape:
            denoised = cv2.resize(denoised, (gray.shape[1], gray.shape[0]))
        
        # Check for NaN/Inf in denoised
        denoised = np.nan_to_num(denoised, nan=gray, posinf=1.0, neginf=0.0)
        
    except Exception:
        # Fallback to simple denoising if wavelet fails
        denoised = cv2.medianBlur((gray * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    
    # Calculate noise residual
    residual = gray - denoised
    residual = residual - residual.mean()
    std = residual.std()
    if std > 1e-6:  # Use small epsilon instead of 0
        residual = residual / (3*std)
    else:
        residual = np.zeros_like(residual)
    
    residual = np.clip(residual, -1, 1)*0.5 + 0.5
    
    # Final NaN/Inf check
    residual = np.nan_to_num(residual, nan=0.5, posinf=1.0, neginf=0.0)
    
    return residual.astype(np.float32)


def extract_ela_enhanced(rgb: np.ndarray, quality: int = 95) -> np.ndarray:
    """Extract ELA (Error Level Analysis) features."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not result:
        raise ValueError("JPEG recompression failed in ELA.")
    
    dec = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if dec is None:
        # Fallback if decode fails
        dec = rgb
    else:
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    
    diff = cv2.absdiff(rgb, dec).astype(np.float32)
    
    ela_channels = []
    for c in range(3):
        channel_diff = diff[:, :, c]
        p95 = np.percentile(channel_diff, 95)
        scale = 255.0/p95 if p95 > 1e-6 else 1.0  # Use epsilon
        scaled = channel_diff * scale
        ela_channels.append(np.clip(scaled, 0, 255)/255.0)
    
    ela = np.stack(ela_channels, axis=-1).astype(np.float32)
    
    # Check for NaN/Inf
    ela = np.nan_to_num(ela, nan=0.0, posinf=1.0, neginf=0.0)
    
    return ela


class PreprocessingWrapper(nn.Module):
    """Wrapper that converts RGB images to 4-channel ELA+PRNU features."""
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB input [B, 3, H, W] to 4-channel features [B, 4, H, W].
        
        Args:
            x: Input tensor of shape [B, 3, H, W] with values in [0, 1] or [0, 255]
        
        Returns:
            Model output logits [B, num_classes]
        """
        # Handle different input formats
        if x.dtype == torch.uint8:
            # Convert from [0, 255] to [0, 1]
            x = x.float() / 255.0
        elif x.max() > 1.0:
            # Assume [0, 255] range
            x = x / 255.0
        
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Convert to numpy for feature extraction (must be on CPU for OpenCV)
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]
        x_np = (x_np * 255.0).astype(np.uint8)  # Convert to uint8 for OpenCV
        
        # Extract features for each image in batch
        feature_list = []
        for i in range(batch_size):
            rgb = x_np[i]
            
            # Extract features
            prnu = extract_prnu_enhanced(rgb)  # [H, W]
            ela = extract_ela_enhanced(rgb)     # [H, W, 3]
            
            # Concatenate: [H, W, 4]
            feat = np.concatenate([ela, prnu[..., None]], axis=-1)
            
            # Final NaN/Inf check
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Convert to tensor format: [4, H, W]
            feat_tensor = np.transpose(feat, (2, 0, 1)).astype(np.float32)
            feature_list.append(feat_tensor)
        
        # Stack to [B, 4, H, W] and move to original device
        features = torch.from_numpy(np.stack(feature_list)).to(device)
        
        # Pass through base model
        return self.base_model(features)


def load_model(weights_path: str, num_classes: int = 3) -> nn.Module:
    """Required entry point - called by gasbench.
    
    Args:
        weights_path: Path to the .safetensors file
        num_classes: Number of output classes from config (should be 3: real, synthetic, semisynthetic)
        
    Returns:
        Loaded PyTorch model ready for inference (wrapped with preprocessing)
    """
    model = ImageELAPRNUDetector(num_classes=num_classes, input_channels=4)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    # Wrap with preprocessing to convert RGB to 4-channel features
    wrapped_model = PreprocessingWrapper(model)
    wrapped_model.eval()
    
    return wrapped_model
