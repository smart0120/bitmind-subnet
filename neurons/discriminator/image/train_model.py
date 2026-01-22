"""
Image Detector Training Script
Loads datasets from Hugging Face and trains a multiclass classifier (real, synthetic, semisynthetic)
using ELA+PRNU fusion features. Outputs model in Safetensors format for submission.

Uses PyTorch DataLoader with direct parquet file reading (no load_dataset).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any
from datetime import datetime
from io import BytesIO
import base64
import tempfile
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import pyarrow.parquet as pq
from safetensors.torch import save_file
import yaml

# Suppress scipy.signal warnings about division by zero
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.signal')

# Import Hugging Face Hub for direct file access
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("WARNING: huggingface_hub not found. Please install with: pip install huggingface_hub")

# Import model and preprocessing from local modules
try:
    from .model import ImageELAPRNUDetector
    from .preprocessing import extract_prnu_enhanced, extract_ela_enhanced
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from model import ImageELAPRNUDetector
    from preprocessing import extract_prnu_enhanced, extract_ela_enhanced

# ---------------------------
# GPU Configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset Configuration
# ---------------------------
# Real image datasets
REAL_DATASETS = [
    "drawthingsai/megalith-10m",
    "bitmind/open-images-v7",
    "bitmind/bm-eidon-image",
    "bitmind/bm-real",
    "bitmind/open-image-v7-256",
    "bitmind/celeb-a-hq",
    "bitmind/ffhq-256",
    "bitmind/MS-COCO-unique",
    "bitmind/MS-COCO-unique-256",
    "bitmind/AFHQ",
    "bitmind/lfw",
    "bitmind/caltech-256",
    "bitmind/caltech-101",
    "bitmind/dtd",
    "bitmind/ffhq-jpg",
]

# Synthetic image datasets
SYNTHETIC_DATASETS = [
    "bitmind/JourneyDB",
    "bitmind/GenImage_MidJourney",
    "bitmind/bm-aura-imagegen",
    "bitmind/bm-imagine",
    "Yejy53/Echo-4o-Image",
    "bitmind/bm-sdxl",
    "bitmind/bm-mobius",
    "bitmind/bm-realvisxl",
    "bitmind/bm-diffusion",
]

# Semi-synthetic image datasets
SEMISYNTHETIC_DATASETS = [
    "bitmind/face-swap",
    "bitmind/ffhq-256___stable-diffusion-xl-base-1.0",
    "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0",
    "bitmind/celeb-a-hq___FLUX.1-dev",
    "bitmind/ffhq-256___FLUX.1-dev",
    "bitmind/MS-COCO-unique___FLUX.1-dev",
    "bitmind/MS-COCO-unique___stable-diffusion-xl-base-1.0"
]

# ---------------------------
# Utils
# ---------------------------
def pil_load_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    """Load and resize image to RGB."""
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return im

def to_numpy(im: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.asarray(im)

def make_feature(rgb: np.ndarray) -> np.ndarray:
    """Create fusion feature: ELA (3 channels) + PRNU (1 channel) = 4 channels."""
    prnu = extract_prnu_enhanced(rgb)  # Single channel
    ela = extract_ela_enhanced(rgb)     # 3 channels (RGB)
    feat = np.concatenate([ela, prnu[..., None]], axis=-1)
    
    # Final NaN/Inf check and cleanup
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
    
    return feat.astype(np.float32)

def load_image_from_data(data: Any, img_size: Tuple[int, int], max_size_mb: float = 50.0) -> Image.Image:
    """
    Load image from various formats (PIL Image, bytes, dict, path).
    
    Args:
        data: Image data in various formats
        img_size: Target image size
        max_size_mb: Maximum image size in MB before decompression (default: 50MB)
    """
    if isinstance(data, Image.Image):
        im = data.convert("RGB")
    elif isinstance(data, (bytes, bytearray)):
        # Check size before loading
        size_mb = len(data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
        
        try:
            im = Image.open(BytesIO(data))
            # Check decompressed size
            if im.size[0] * im.size[1] > 10000 * 10000:  # 100MP limit
                raise ValueError(f"Decompressed image too large: {im.size[0]}x{im.size[1]}")
            im = im.convert("RGB")
        except Exception as e:
            if "Decompressed Data Too Large" in str(e) or "too large" in str(e).lower():
                raise ValueError(f"Decompressed Data Too Large: {size_mb:.2f}MB")
            raise
    elif isinstance(data, str):
        # Try base64 decode first
        if len(data) > 100 and not (data.startswith('http') or '/' in data or '\\' in data):
            try:
                img_bytes = base64.b64decode(data)
                size_mb = len(img_bytes) / (1024 * 1024)
                if size_mb > max_size_mb:
                    raise ValueError(f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
                im = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                if "Decompressed Data Too Large" in str(e) or "too large" in str(e).lower():
                    raise ValueError(f"Decompressed Data Too Large")
                # Assume it's a file path
                im = pil_load_rgb(data, img_size)
        else:
            im = pil_load_rgb(data, img_size)
    elif isinstance(data, dict):
        # Try common keys
        for key in ['image', 'bytes', 'path', 'file_path', 'file']:
            if key in data:
                return load_image_from_data(data[key], img_size, max_size_mb)
        # Use first value
        first_val = list(data.values())[0]
        return load_image_from_data(first_val, img_size, max_size_mb)
    else:
        raise ValueError(f"Unknown image data type: {type(data)}")
    
    if img_size and (im.size[0] != img_size[0] or im.size[1] != img_size[1]):
        im = im.resize(img_size, Image.Resampling.LANCZOS)
    
    return im

# ---------------------------
# Dataset Classes
# ---------------------------
class ImageDataset(Dataset):
    """PyTorch Dataset for loading images."""
    
    def __init__(self, image_data: List[Any], labels: List[int], img_size: Tuple[int, int]):
        self.image_data = image_data
        self.labels = labels
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        label = self.labels[idx]
        
        try:
            im = load_image_from_data(data, self.img_size, max_size_mb=50.0)
            rgb = to_numpy(im)
            feat = make_feature(rgb)  # Shape: (H, W, 4)
            
            # Check for NaN/Inf before converting to tensor
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                raise ValueError("NaN or Inf values in features")
            
            # Convert to tensor: (C, H, W)
            feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
            
            # Final check for NaN/Inf in tensor
            if torch.any(torch.isnan(feat_tensor)) or torch.any(torch.isinf(feat_tensor)):
                raise ValueError("NaN or Inf values in tensor")
            
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return feat_tensor, label_tensor
        
        except Exception as e:
            # Silently skip problematic images (don't print every error to avoid spam)
            if "Decompressed Data Too Large" in str(e) or "too large" in str(e).lower():
                pass  # Skip silently for size errors
            elif idx % 1000 == 0:  # Only print every 1000th error
                print(f"Error loading image {idx}: {e}")
            
            # Return zero tensor on error
            C = 4  # ELA (3) + PRNU (1)
            H, W = self.img_size[1], self.img_size[0]
            return torch.zeros(C, H, W), torch.tensor(0, dtype=torch.long)

# ---------------------------
# Dataset Loading from Hugging Face (Direct Parquet Reading)
# ---------------------------
def load_hf_dataset_from_parquet(
    dataset_name: str, 
    split: str = "train",
    max_samples: int = None,
    max_parquet_files: int = None
) -> List[Any]:
    """
    Load dataset from Hugging Face by reading parquet files directly.
    Uses huggingface_hub to download and pyarrow to read parquet files.
    
    Args:
        dataset_name: Name of the dataset (e.g., "bitmind/ffhq-256")
        split: Dataset split to load (e.g., "train")
        max_samples: Maximum number of samples to load (None for all)
        max_parquet_files: Maximum number of parquet files to process
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub is required. Please install it with:\n"
            "  pip install huggingface_hub\n"
            "Or install all training requirements:\n"
            "  pip install -r requirements_training.txt"
        )
    
    try:
        print(f"Loading {dataset_name} (split: {split})...")
        
        # List parquet files in the dataset
        try:
            all_files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
            parquet_files = [f for f in all_files if f.endswith('.parquet') and split in f]
            
            # Also check in data/ subdirectory
            if not parquet_files:
                data_files = [f for f in all_files if 'data' in f and f.endswith('.parquet')]
                parquet_files = [f for f in data_files if split in f]
            
            # If still no files, try any parquet file
            if not parquet_files:
                parquet_files = [f for f in all_files if f.endswith('.parquet')]
            
            if not parquet_files:
                print(f"  ⚠️  No parquet files found in {dataset_name}")
                return []
            
            # Limit number of parquet files to process (if max_parquet_files is set)
            all_parquet_files = [f for f in all_files if f.endswith('.parquet')]
            total_parquet_count = len(all_parquet_files)
            if max_parquet_files and max_parquet_files > 0:
                original_count = len(parquet_files)
                parquet_files = parquet_files[:max_parquet_files]
                print(f"  ⚠️  LIMITING: Found {total_parquet_count} total parquet file(s), processing {len(parquet_files)}/{original_count} matching files (limited by --max-parquet-files={max_parquet_files})")
            else:
                print(f"  ✓ Loading ALL {len(parquet_files)} parquet file(s) from {total_parquet_count} total")
            
        except Exception as e:
            print(f"  ⚠️  Error listing files: {e}")
            return []
        
        image_data = []
        count = 0
        
        # Process each parquet file
        for parquet_file in parquet_files:
            if max_samples and count >= max_samples:
                break
            
            try:
                print(f"  Processing {parquet_file}...")
                
                # Download parquet file (returns path to cached file)
                try:
                    tmp_path = hf_hub_download(
                        repo_id=dataset_name,
                        filename=parquet_file,
                        repo_type="dataset"
                    )
                    
                    # Read parquet file
                    table = pq.read_table(tmp_path)
                    df = table.to_pandas()
                    
                    # Find image column
                    image_col = None
                    for col in ['image', 'path', 'file_path', 'file', 'img', 'image_path', 'url']:
                        if col in df.columns:
                            image_col = col
                            break
                    
                    if image_col is None:
                        # Try to find any column with image-like data
                        for col in df.columns:
                            if 'image' in col.lower() and '_id' not in col.lower():
                                image_col = col
                                break
                    
                    if image_col is None:
                        print(f"    ⚠️  No image column found in {parquet_file}")
                        continue
                    
                    # Extract image data
                    for _, row in df.iterrows():
                        if max_samples and count >= max_samples:
                            break
                        
                        try:
                            img_data = row[image_col]
                            
                            if pd.isna(img_data) or img_data is None:
                                continue
                            
                            # Handle different data types
                            if isinstance(img_data, Image.Image):
                                image_data.append(img_data)
                                count += 1
                            elif isinstance(img_data, (bytes, bytearray)):
                                image_data.append(img_data)
                                count += 1
                            elif isinstance(img_data, str):
                                # Could be path or base64
                                image_data.append(img_data)
                                count += 1
                            elif isinstance(img_data, dict):
                                # Extract bytes or image from dict
                                for key in ['bytes', 'image', 'data', 'content']:
                                    if key in img_data:
                                        image_data.append(img_data[key])
                                        count += 1
                                        break
                            
                            if count % 10000 == 0:
                                print(f"    Loaded {count} images so far...")
                        
                        except Exception as e:
                            continue
                    
                    print(f"    Loaded {len(image_data) - (count - df.shape[0])} images from {parquet_file}")
                
                except Exception as e:
                    print(f"    ⚠️  Error reading parquet: {e}")
                    continue
            
            except Exception as e:
                print(f"    ⚠️  Error processing {parquet_file}: {e}")
                continue
        
        print(f"Loaded {len(image_data)} total images from {dataset_name}")
        return image_data
    
    except Exception as e:
        print(f"  ⚠️  Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_all_datasets(
    real_datasets: List[str],
    synthetic_datasets: List[str],
    semisynthetic_datasets: List[str],
    balance_classes: bool = False,
    max_samples_per_dataset: int = None,
    max_parquet_files: int = None
) -> Tuple[List[Any], List[int]]:
    """Load all datasets and create labeled dataset."""
    
    all_data = []
    all_labels = []
    
    # Load real images (label 0)
    print("\n=== Loading Real Images ===")
    for ds_name in real_datasets:
        data = load_hf_dataset_from_parquet(
            ds_name, split="train", 
            max_samples=max_samples_per_dataset,
            max_parquet_files=max_parquet_files
        )
        all_data.extend(data)
        all_labels.extend([0] * len(data))
    
    # Load synthetic images (label 1)
    print("\n=== Loading Synthetic Images ===")
    for ds_name in synthetic_datasets:
        data = load_hf_dataset_from_parquet(
            ds_name, split="train",
            max_samples=max_samples_per_dataset,
            max_parquet_files=max_parquet_files
        )
        all_data.extend(data)
        all_labels.extend([1] * len(data))
    
    # Load semisynthetic images (label 2)
    print("\n=== Loading Semi-synthetic Images ===")
    for ds_name in semisynthetic_datasets:
        data = load_hf_dataset_from_parquet(
            ds_name, split="train",
            max_samples=max_samples_per_dataset,
            max_parquet_files=max_parquet_files
        )
        all_data.extend(data)
        all_labels.extend([2] * len(data))
    
    print(f"\nTotal samples loaded: {len(all_data):,}")
    print(f"  Real: {sum(1 for l in all_labels if l == 0):,}")
    print(f"  Synthetic: {sum(1 for l in all_labels if l == 1):,}")
    print(f"  Semi-synthetic: {sum(1 for l in all_labels if l == 2):,}")
    
    # Balance classes if requested
    if balance_classes:
        counts = [sum(1 for l in all_labels if l == i) for i in range(3)]
        min_count = min(counts)
        discarded = len(all_data) - (min_count * 3)
        print(f"\n⚠️  WARNING: Balancing will reduce dataset from {len(all_data):,} to {min_count * 3:,} samples")
        print(f"   This discards {discarded:,} samples ({100*discarded/len(all_data):.1f}% of data)!")
        print(f"   Class counts: Real={counts[0]:,}, Synthetic={counts[1]:,}, Semi-synthetic={counts[2]:,}")
        print(f"   Will balance to: {min_count:,} samples per class")
        print(f"\nBalancing to {min_count:,} samples per class...")
        
        balanced_data = []
        balanced_labels = []
        for label in range(3):
            label_data = [d for d, l in zip(all_data, all_labels) if l == label]
            balanced_data.extend(label_data[:min_count])
            balanced_labels.extend([label] * min_count)
        
        all_data = balanced_data
        all_labels = balanced_labels
        
        print(f"After balancing: {len(all_data):,} samples")
    else:
        print(f"\n✓ Using ALL {len(all_data):,} samples (no balancing)")
        counts = [sum(1 for l in all_labels if l == i) for i in range(3)]
        print(f"   Class distribution: Real={counts[0]:,}, Synthetic={counts[1]:,}, Semi-synthetic={counts[2]:,}")
    
    return all_data, all_labels

# ---------------------------
# Training Function
# ---------------------------
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    num_epochs: int,
    learning_rate: float,
    output_dir: Path,
    device: torch.device,
    use_amp: bool = True,
    compile_model: bool = True
):
    """Train the model with optimizations for large GPU."""
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed precision training (FP16/BF16) - uses more VRAM and speeds up training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Compile model for faster training (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("Compiling model for faster training...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}, continuing without compilation")
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Non-blocking transfer for faster GPU utilization
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Check for NaN/Inf in input
            if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
                if batch_idx % 1000 == 0:
                    print(f"Warning: NaN/Inf detected in batch {batch_idx}, skipping...")
                continue
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    
                    # Check for NaN/Inf in output
                    if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                        if batch_idx % 1000 == 0:
                            print(f"Warning: NaN/Inf in model output at batch {batch_idx}, skipping...")
                        continue
                    
                    loss = criterion(output, target)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    if batch_idx % 1000 == 0:
                        print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                
                # Check for NaN/Inf in output
                if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                    if batch_idx % 1000 == 0:
                        print(f"Warning: NaN/Inf in model output at batch {batch_idx}, skipping...")
                    continue
                
                loss = criterion(output, target)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    if batch_idx % 1000 == 0:
                        print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 100 == 0:
                # Show GPU memory usage
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"VRAM: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    return model

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, output_dir: Path):
    """Evaluate model and generate reports."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Classification report
    cr = classification_report(
        all_labels, all_preds,
        target_names=["Real", "Synthetic", "Semi-synthetic"],
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save results
    pd.DataFrame(cm, index=["Real", "Synthetic", "Semi-synthetic"], 
                 columns=["Real", "Synthetic", "Semi-synthetic"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(cr)
    
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'prob_real': [p[0] for p in all_probs],
        'prob_synthetic': [p[1] for p in all_probs],
        'prob_semisynthetic': [p[2] for p in all_probs],
    })
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(cr)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nResults saved to: {output_dir.resolve()}")

# ---------------------------
# Main Training Function
# ---------------------------
def main(args):
    """Main training pipeline."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"./outputs/image_detector/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Load datasets
    print("\n" + "="*50)
    print("LOADING DATASETS FROM HUGGING FACE (Direct Parquet Reading)")
    print("="*50)
    
    all_data, all_labels = load_all_datasets(
        real_datasets=REAL_DATASETS if not args.real_datasets else args.real_datasets,
        synthetic_datasets=SYNTHETIC_DATASETS if not args.synthetic_datasets else args.synthetic_datasets,
        semisynthetic_datasets=SEMISYNTHETIC_DATASETS if not args.semisynthetic_datasets else args.semisynthetic_datasets,
        balance_classes=args.balance,
        max_samples_per_dataset=args.max_samples_per_dataset,
        max_parquet_files=args.max_parquet_files
    )
    
    if len(all_data) == 0:
        print("Error: No images loaded!")
        return
    
    # Split train/val
    train_data, val_data, train_labels, val_labels = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets
    img_size = (args.img_size[0], args.img_size[1])
    train_dataset = ImageDataset(train_data, train_labels, img_size)
    val_dataset = ImageDataset(val_data, val_labels, img_size)
    
    # Create data loaders with optimizations
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None,  # Prefetch batches
        drop_last=True  # Drop incomplete batch for consistent training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Create model
    model = ImageELAPRNUDetector(num_classes=3, input_channels=4).to(device)
    print(f"\nModel created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Estimate memory usage and suggest batch size
    if torch.cuda.is_available():
        # Estimate memory per sample (4 channels, 256x256)
        img_size = args.img_size[0] * args.img_size[1]
        bytes_per_sample = img_size * 4 * 4  # 4 channels * 4 bytes (float32)
        estimated_batch_memory = args.batch_size * bytes_per_sample / (1024**3)  # GB
        
        print(f"\nEstimated memory per batch: ~{estimated_batch_memory:.2f} GB")
        print(f"Current batch size: {args.batch_size}")
        
        # Suggest larger batch size for 80GB GPU
        if torch.cuda.get_device_properties(0).total_memory > 70 * 1024**3:  # > 70GB
            suggested_batch = min(512, int(80 / estimated_batch_memory * args.batch_size))
            if suggested_batch > args.batch_size:
                print(f"💡 Suggestion: For 80GB GPU, try --batch-size {suggested_batch} to use more VRAM")
    
    # Train
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    model = train_model(
        train_loader, val_loader, model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=output_dir,
        device=device,
        use_amp=args.use_amp,
        compile_model=args.compile_model
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    evaluate_model(model, val_loader, device, output_dir)
    
    # Save model in Safetensors format
    print("\n" + "="*50)
    print("SAVING MODEL IN SAFETENSORS FORMAT")
    print("="*50)
    
    state_dict = model.state_dict()
    safetensors_path = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"Saved model weights to: {safetensors_path}")
    
    # Save model config for submission
    config = {
        'name': args.model_name,
        'version': args.model_version,
        'modality': 'image',
        'preprocessing': {
            'resize': list(img_size),
            'normalize': {
                'mean': [0.0, 0.0, 0.0, 0.0],  # 4 channels - no normalization (features already [0,1])
                'std': [1.0, 1.0, 1.0, 1.0]
            }
        },
        'model': {
            'num_classes': 3,
            'weights_file': 'model.safetensors'
        }
    }
    
    config_path = output_dir / "model_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved model config to: {config_path}")
    
    print(f"\n✓ Training complete! Model ready for submission.")
    print(f"  Model files: {output_dir}")
    print(f"  Run package_model.py to create submission zip")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Detector with ELA+PRNU Fusion")
    
    # Dataset arguments
    parser.add_argument("--real-datasets", type=str, nargs="+", default=None,
                        help="Custom list of real image datasets (default: use predefined)")
    parser.add_argument("--synthetic-datasets", type=str, nargs="+", default=None,
                        help="Custom list of synthetic image datasets")
    parser.add_argument("--semisynthetic-datasets", type=str, nargs="+", default=None,
                        help="Custom list of semisynthetic image datasets")
    parser.add_argument("--balance", action="store_true", default=False,
                        help="Balance classes by taking min samples per class (WARNING: discards data, default: False)")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Maximum samples to load per dataset (None for all, default: None = ALL)")
    parser.add_argument("--max-parquet-files", type=int, default=None,
                        help="Maximum parquet files to process per dataset (None for all, default: None = ALL)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=256, 
                        help="Batch size (default: 256 for large GPU, increase for 80GB GPU)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                        help="Image size (width height)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8, 
                        help="DataLoader workers (default: 8, increase for faster data loading)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use Automatic Mixed Precision (FP16) for faster training and more VRAM usage")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--compile-model", action="store_true", default=True,
                        help="Compile model with torch.compile for faster training (PyTorch 2.0+)")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false",
                        help="Disable model compilation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="image-detector",
                        help="Model name for submission")
    parser.add_argument("--model-version", type=str, default="1.0.0",
                        help="Model version")
    
    args = parser.parse_args()
    
    print("="*50)
    print("IMAGE DETECTOR - ELA+PRNU FUSION")
    print("="*50)
    print(f"Image Size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"Batch Size: {args.batch_size} (adjust for your GPU VRAM)")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"DataLoader Workers: {args.num_workers}")
    print(f"Mixed Precision (AMP): {args.use_amp} {'✓' if args.use_amp else '✗'}")
    print(f"Model Compilation: {args.compile_model} {'✓' if args.compile_model else '✗'}")
    print(f"Balance Classes: {args.balance} {'⚠️  (WILL DISCARD DATA!)' if args.balance else '✓ (using all data)'}")
    print(f"Max Parquet Files per Dataset: {args.max_parquet_files if args.max_parquet_files else 'ALL ✓'}")
    print(f"Max Samples per Dataset: {args.max_samples_per_dataset if args.max_samples_per_dataset else 'ALL ✓'}")
    print(f"Using PyTorch DataLoader with direct parquet file reading")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Current VRAM usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("="*50 + "\n")
    
    main(args)
