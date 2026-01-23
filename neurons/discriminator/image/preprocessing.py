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
import warnings


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
    
    # Final NaN/Inf check
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Convert to tensor format: [4, H, W]
    feat_tensor = np.transpose(feat, (2, 0, 1)).astype(np.float32)
    
    return feat_tensor
