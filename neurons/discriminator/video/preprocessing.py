"""
Preprocessing utilities for video inference.
Extracts ELA+PRNU features from video frames for model input.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
import pywt
from scipy.signal import wiener
import av
import warnings

# Suppress scipy.signal warnings about division by zero
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.signal')


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
            try:
                cH_denoised = wiener(cH, mysize=5)
                cV_denoised = wiener(cV, mysize=5)
                cD_denoised = wiener(cD, mysize=5)
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


def extract_frames_from_video(video_path: str, num_frames: int = 8, target_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: (width, height) target size for frames
        
    Returns:
        List of RGB frames as numpy arrays
    """
    frames = []
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Calculate frame indices to sample uniformly
        total_frames = stream.frames
        if total_frames == 0:
            # Fallback: count frames manually
            total_frames = sum(1 for _ in container.decode(stream))
            container = av.open(video_path)  # Reopen
        
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Extract frames
        for i, frame in enumerate(container.decode(stream)):
            if i in frame_indices:
                img = frame.to_image().convert("RGB")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(np.asarray(img))
                if len(frames) >= num_frames:
                    break
        
        container.close()
        
        # If we didn't get enough frames, pad with last frame
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Create a black frame if no frames were extracted
                frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        
        return frames[:num_frames]
    
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        # Return black frames on error
        return [np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8) for _ in range(num_frames)]


def preprocess_video(video_path: str, num_frames: int = 8, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess video for model inference.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: (width, height) target size
        
    Returns:
        Feature tensor of shape (num_frames, 4, H, W) ready for model input
        Channels: [ELA_R, ELA_G, ELA_B, PRNU]
    """
    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames, target_size)
    
    # Extract features for each frame
    frame_features = []
    for frame_rgb in frames:
        # Extract features
        prnu = extract_prnu_enhanced(frame_rgb)  # Single channel
        ela = extract_ela_enhanced(frame_rgb)     # 3 channels (RGB)
        
        # Concatenate: [H, W, 4]
        feat = np.concatenate([ela, prnu[..., None]], axis=-1)
        
        # Final NaN/Inf check and cleanup
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Convert to tensor format: [4, H, W]
        feat_tensor = np.transpose(feat, (2, 0, 1)).astype(np.float32)
        frame_features.append(feat_tensor)
    
    # Stack frames: [num_frames, 4, H, W]
    video_features = np.stack(frame_features, axis=0)
    
    # Final check
    video_features = np.nan_to_num(video_features, nan=0.0, posinf=1.0, neginf=0.0)
    
    return video_features
