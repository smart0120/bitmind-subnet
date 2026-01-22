"""
Preprocessing utilities for image inference.
Extracts ELA+PRNU features from images for model input.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import pywt
from scipy.signal import wiener


def extract_prnu_enhanced(rgb: np.ndarray) -> np.ndarray:
    """Extract PRNU (Photo Response Non-Uniformity) features using wavelet denoising."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    # Wavelet decomposition for denoising
    coeffs = pywt.wavedec2(gray, 'db4', level=2)
    cA = coeffs[0]
    details = []
    for level_coeffs in coeffs[1:]:
        cH, cV, cD = level_coeffs
        details.extend([wiener(cH, mysize=5), wiener(cV, mysize=5), wiener(cD, mysize=5)])
    
    reconstructed_coeffs = [cA] + [tuple(details[i:i+3]) for i in range(0, len(details), 3)]
    denoised = pywt.waverec2(reconstructed_coeffs, 'db4')
    
    if denoised.shape != gray.shape:
        denoised = cv2.resize(denoised, (gray.shape[1], gray.shape[0]))
    
    # Calculate noise residual
    residual = gray - denoised
    residual = residual - residual.mean()
    std = residual.std()
    if std > 0:
        residual = residual / (3*std)
    residual = np.clip(residual, -1, 1)*0.5 + 0.5
    return residual.astype(np.float32)


def extract_ela_enhanced(rgb: np.ndarray, quality: int = 95) -> np.ndarray:
    """Extract ELA (Error Level Analysis) features."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not result:
        raise ValueError("JPEG recompression failed in ELA.")
    
    dec = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    diff = cv2.absdiff(rgb, dec).astype(np.float32)
    
    ela_channels = []
    for c in range(3):
        channel_diff = diff[:, :, c]
        p95 = np.percentile(channel_diff, 95)
        scale = 255.0/p95 if p95>0 else 1.0
        ela_channels.append(np.clip(channel_diff*scale, 0, 255)/255.0)
    
    ela = np.stack(ela_channels, axis=-1).astype(np.float32)
    return ela


def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image (RGB)
        target_size: (width, height) target size
        
    Returns:
        Feature tensor of shape (4, H, W) ready for model input
        Channels: [ELA_R, ELA_G, ELA_B, PRNU]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy
    rgb = np.asarray(image)
    
    # Extract features
    prnu = extract_prnu_enhanced(rgb)  # Single channel
    ela = extract_ela_enhanced(rgb)     # 3 channels (RGB)
    
    # Concatenate: [H, W, 4]
    feat = np.concatenate([ela, prnu[..., None]], axis=-1)
    
    # Convert to tensor format: [4, H, W]
    feat_tensor = np.transpose(feat, (2, 0, 1)).astype(np.float32)
    
    return feat_tensor
