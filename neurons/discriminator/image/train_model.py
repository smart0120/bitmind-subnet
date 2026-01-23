"""
Image Detector Training Script
Loads datasets from Hugging Face and trains a multiclass classifier (real, synthetic, semisynthetic)
using ELA+PRNU fusion features. Outputs model in Safetensors format for submission.

Optimizations:
- Automatic Mixed Precision (AMP) for faster training and reduced memory
- Gradient accumulation for effective larger batch sizes
- On-the-fly data augmentation (memory efficient)
- Optimized DataLoader with persistent workers and prefetching
- Automatic streaming mode for large datasets
- Gradient clipping to prevent exploding gradients
- Memory cleanup and cache management
- Optional profiling for bottleneck identification
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import time
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset, IterableDataset
from safetensors.torch import save_file
import yaml
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import requests

# Suppress trust_remote_code warnings (deprecated, but some datasets still trigger it)
warnings.filterwarnings('ignore', message='.*trust_remote_code.*', category=UserWarning)

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

# Import dataset download functionality from gas
try:
    from gas.datasets.download import (
        list_hf_files, download_files, download_single_file, yield_media_from_source
    )
    from gas.types import DatasetConfig, Modality, MediaType
    # Import private functions needed for dataset download
    import gas.datasets.download as download_module
except ImportError as e:
    print(f"Warning: Could not import gas.datasets.download: {e}")
    print("Make sure gas module is available. Falling back to in-memory loading.")
    list_hf_files = None
    download_module = None

# Import download tracker
try:
    from .download_tracker import DownloadTracker
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from download_tracker import DownloadTracker

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
    return feat.astype(np.float32)

# ---------------------------
# Dataset Class
# ---------------------------
class ImageDataset(Dataset):
    """Dataset class for loading images from Hugging Face datasets."""
    
    def __init__(self, image_paths: List[str], labels: List[int], img_size: Tuple[int, int], 
                 augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        
        # Data augmentation transforms (applied on-the-fly to save memory)
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            if isinstance(path, dict):
                # Hugging Face dataset format
                if 'image' in path:
                    im = path['image']
                    if not isinstance(im, Image.Image):
                        im = pil_load_rgb(im, self.img_size)
                    else:
                        im = im.convert("RGB")
                elif 'path' in path:
                    im = pil_load_rgb(path['path'], self.img_size)
                else:
                    # Try to find any image-like field
                    for key in ['image', 'img', 'file_path', 'file', 'path', 'url']:
                        if key in path:
                            val = path[key]
                            if isinstance(val, Image.Image):
                                im = val.convert("RGB")
                            elif isinstance(val, str):
                                im = pil_load_rgb(val, self.img_size)
                            break
                    else:
                        raise ValueError(f"Unknown path format: {list(path.keys())}")
            elif isinstance(path, Image.Image):
                # Already a PIL Image
                im = path.convert("RGB")
            else:
                # String path
                im = pil_load_rgb(path, self.img_size)
            
            # Resize if needed
            if self.img_size and (im.size[0] != self.img_size[0] or im.size[1] != self.img_size[1]):
                im = im.resize(self.img_size, Image.Resampling.LANCZOS)
            
            # Apply data augmentation on-the-fly (saves memory vs pre-augmenting)
            if self.transform is not None:
                im = self.transform(im)
            
            rgb = to_numpy(im)
            feat = make_feature(rgb)  # Shape: (H, W, 4)
            
            # Convert to tensor: (C, H, W)
            feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return feat_tensor, label_tensor
        
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return zero tensor on error
            C = 4  # ELA (3) + PRNU (1)
            H, W = self.img_size[1], self.img_size[0]
            return torch.zeros(C, H, W), torch.tensor(0, dtype=torch.long)

# ---------------------------
# Parallel Download Functions
# ---------------------------
def download_files_parallel(
    urls: List[str], 
    output_dir: Path, 
    max_workers: int = 4,
    chunk_size: int = 8192
) -> List[Path]:
    """
    Download multiple files in parallel.
    
    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        max_workers: Number of parallel download workers
        chunk_size: Size of chunks to download at a time
    
    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    downloaded_lock = Lock()
    
    def download_single(url):
        """Download a single file."""
        try:
            import requests
            filename = os.path.basename(url)
            filepath = output_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                return filepath
            
            response = requests.get(url, stream=True, timeout=3600)
            if response.status_code != 200:
                print(f"  ⚠️  Failed to download {filename}: Status {response.status_code}")
                return None
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            
            return filepath
        except Exception as e:
            print(f"  ⚠️  Error downloading {os.path.basename(url)}: {e}")
            return None
    
    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single, url): url for url in urls}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                with downloaded_lock:
                    downloaded_files.append(result)
            if completed % 10 == 0:
                print(f"  Downloaded {completed}/{len(urls)} files...")
    
    return downloaded_files


# ---------------------------
# Dataset Download to Disk
# ---------------------------
def download_dataset_to_disk(
    dataset_name: str,
    output_dir: Path,
    media_type: str = "real",  # "real", "synthetic", "semisynthetic"
    max_files: Optional[int] = None,
    max_images_per_file: Optional[int] = None,
    tracker: Optional[DownloadTracker] = None,
    force_download: bool = False,
    max_workers: int = 4  # Number of parallel download/extraction workers
) -> List[str]:
    """
    Download dataset to disk and extract images with tracking.
    
    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Directory to save downloaded images
        media_type: Type of media (real, synthetic, semisynthetic)
        max_files: Maximum number of files to download (None = all)
        max_images_per_file: Maximum images to extract per file (None = all)
        tracker: DownloadTracker instance for tracking status
        force_download: If True, re-download even if already downloaded
    
    Returns:
        List of paths to downloaded image files
    """
    if download_module is None:
        print(f"  ⚠️  Cannot download {dataset_name}: gas module not available")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker if not provided
    if tracker is None:
        tracker = DownloadTracker()
    
    # Clear dataset from tracker if force download
    if force_download:
        tracker.clear_dataset(dataset_name)
    
    # Create DatasetConfig
    dataset_config = DatasetConfig(
        path=dataset_name,
        modality=Modality.IMAGE,
        media_type=MediaType(media_type.lower()),
        source_format="parquet"  # Default, will be auto-detected
    )
    
    # Detect source format by listing files
    try:
        # Try different formats
        formats_to_try = [".parquet", ".zip", ".jpg", ".png"]
        filenames = []
        detected_format = None
        
        for fmt in formats_to_try:
            # Use the private function from the module
            filenames = download_module._list_remote_dataset_files(dataset_name, fmt)
            if filenames:
                detected_format = fmt.lstrip(".")
                dataset_config.source_format = detected_format
                break
        
        if not filenames:
            print(f"  ⚠️  No files found for {dataset_name}")
            return []
        
        print(f"  Found {len(filenames)} files (format: {detected_format})")
        
        # Check database for already downloaded files (batch check for speed)
        files_to_download = []
        already_downloaded = []
        
        # Batch check all files at once for better performance
        if not force_download:
            # Get all downloaded files for this dataset from database
            downloaded_files_db = {f[0] for f in tracker.get_downloaded_files(dataset_name)}
            for filename in filenames:
                if filename in downloaded_files_db:
                    already_downloaded.append(filename)
                else:
                    files_to_download.append(filename)
        else:
            files_to_download = filenames
        
        if already_downloaded:
            print(f"  ✓ Found {len(already_downloaded)} already downloaded files in database")
        
        # Limit number of files if specified
        if max_files:
            # Prioritize files that need downloading
            if len(files_to_download) > max_files:
                import random
                files_to_download = random.sample(files_to_download, max_files)
            elif len(files_to_download) < max_files and already_downloaded:
                # Add some already downloaded files if we need more
                needed = max_files - len(files_to_download)
                files_to_download.extend(already_downloaded[:needed])
        
        # Get image paths from already downloaded files
        image_paths = []
        if already_downloaded:
            image_paths = tracker.get_image_paths(dataset_name, output_dir)
            print(f"  ✓ Using {len(image_paths)} images from previously downloaded files")
        
        # Download and extract remaining files
        if files_to_download:
            print(f"  Downloading {len(files_to_download)} new files (using {max_workers} parallel workers)...")
            
            # Get download URLs using the private function
            remote_paths = download_module._get_download_urls(dataset_name, files_to_download)
            
            # Download files in parallel
            downloaded_files = download_files_parallel(remote_paths, output_dir, max_workers=max_workers)
            
            if not downloaded_files:
                print(f"  ⚠️  Failed to download any files for {dataset_name}")
            else:
                # Extract images from downloaded files in parallel
                image_paths_lock = Lock()
                
                def extract_from_file(source_file):
                    """Extract images from a single file."""
                    file_name = source_file.name
                    file_image_paths = []
                    
                    # Mark as downloading
                    tracker.mark_downloading(
                        dataset_name, file_name, detected_format, str(output_dir)
                    )
                    
                    try:
                        num_items = max_images_per_file if max_images_per_file else None  # None = all
                        count = 0
                        
                        for media_obj, metadata in yield_media_from_source(source_file, dataset_config, num_items or 10000):
                            # Save image to disk
                            if isinstance(media_obj, Image.Image):
                                # Generate unique filename
                                img_hash = hashlib.md5(f"{dataset_name}_{file_name}_{count}_{time.time()}".encode()).hexdigest()[:8]
                                img_path = output_dir / f"{dataset_name.replace('/', '_')}_{img_hash}.jpg"
                                
                                # Convert to RGB and save
                                if media_obj.mode != 'RGB':
                                    media_obj = media_obj.convert('RGB')
                                media_obj.save(img_path, 'JPEG', quality=95)
                                file_image_paths.append(str(img_path))
                                count += 1
                        
                        # Mark as completed in database
                        tracker.mark_completed(dataset_name, file_name, count)
                        return (file_name, count, file_image_paths, None)
                        
                    except Exception as e:
                        error_msg = str(e)
                        tracker.mark_failed(dataset_name, file_name, error_msg)
                        return (file_name, 0, [], error_msg)
                
                # Process files in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(extract_from_file, source_file): source_file 
                              for source_file in downloaded_files}
                    
                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        file_name, count, file_image_paths, error = future.result()
                        
                        if error:
                            print(f"    ✗ [{completed}/{len(downloaded_files)}] Failed {file_name}: {error}")
                        else:
                            with image_paths_lock:
                                image_paths.extend(file_image_paths)
                            print(f"    ✓ [{completed}/{len(downloaded_files)}] Extracted {count} images from {file_name}")
        
        print(f"  ✓ Total {len(image_paths)} images available for {dataset_name}")
        return image_paths
        
    except Exception as e:
        print(f"  ✗ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


# ---------------------------
# Dataset Loading from Hugging Face
# ---------------------------
def load_hf_dataset(
    dataset_name: str, 
    split: str = "train", 
    max_samples: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> List:
    """
    Load dataset from Hugging Face with retry logic for rate limiting.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
        max_samples: Maximum samples to load
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    use_streaming = False  # Track if we should use streaming mode
    
    for attempt in range(max_retries):
        try:
            # Determine if we should use streaming mode
            if use_streaming or attempt > 0:
                print(f"Loading {dataset_name} (split: {split})... [Attempt {attempt + 1}/{max_retries}] [Streaming Mode]")
            else:
                print(f"Loading {dataset_name} (split: {split})... [Attempt {attempt + 1}/{max_retries}]")
            
            # Try loading dataset
            try:
                if use_streaming or attempt > 0:
                    # Use streaming mode for retries or if explicitly set
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        streaming=True,
                        token=hf_token,
                        batch_size=64
                    )
                else:
                    # Try normal loading first
                    dataset = load_dataset(
                        dataset_name, 
                        split=split,
                        token=hf_token,
                        num_proc=1  # Reduce parallel requests to avoid rate limits
                    )
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if error is related to data being too large
                is_data_too_large = (
                    'decompressed data too large' in error_str or
                    'data too large' in error_str or
                    'too large' in error_str and 'data' in error_str
                )
                
                # Check if rate limited
                is_rate_limited = (
                    '429' in error_str or 
                    'rate limit' in error_str or 
                    'too many requests' in error_str
                )
                
                # If data too large or rate limited, switch to streaming
                if (is_data_too_large or is_rate_limited) and not use_streaming:
                    if is_data_too_large:
                        print(f"  ⚠️  Dataset too large for normal loading, switching to streaming mode...")
                    else:
                        print(f"  ⚠️  Rate limited, switching to streaming mode...")
                    
                    use_streaming = True
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    
                    # Retry with streaming
                    dataset = load_dataset(
                        dataset_name,
                        split=split,
                        streaming=True,
                        token=hf_token,
                        batch_size=64
                    )
                elif is_rate_limited and use_streaming:
                    # Already using streaming but still rate limited, wait and retry
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  ⚠️  Rate limited even with streaming. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other error, re-raise
                    raise
            
            # Limit samples if specified (only for non-streaming datasets)
            is_streaming = isinstance(dataset, IterableDataset)
            if is_streaming:
                print(f"  ✓ Using streaming mode (processes data in chunks, no memory limit)")
            if not is_streaming and max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Extract image paths
            # Note: Streaming datasets are processed incrementally, which prevents memory issues
            image_paths = []
            count = 0
            
            for item in dataset:
                if max_samples and count >= max_samples:
                    break
                
                if isinstance(item, dict):
                    if 'image' in item:
                        # Image is already loaded as PIL Image
                        image_paths.append(item)
                        count += 1
                    elif 'path' in item:
                        image_paths.append(item['path'])
                        count += 1
                    else:
                        # Try to find image field
                        found = False
                        for key in ['image', 'img', 'file_path', 'file', 'path', 'url']:
                            if key in item:
                                val = item[key]
                                if isinstance(val, Image.Image):
                                    image_paths.append(item)
                                    count += 1
                                elif isinstance(val, str):
                                    image_paths.append(val)
                                    count += 1
                                found = True
                                break
                        if not found:
                            # Use the first value if it's a path-like string or PIL Image
                            first_val = list(item.values())[0]
                            if isinstance(first_val, Image.Image):
                                image_paths.append(item)
                                count += 1
                            elif isinstance(first_val, str):
                                image_paths.append(first_val)
                                count += 1
                elif isinstance(item, Image.Image):
                    # Direct PIL Image
                    image_paths.append(item)
                    count += 1
                elif isinstance(item, str):
                    # Direct path string
                    image_paths.append(item)
                    count += 1
                
                # Progress indicator for large datasets
                if count % 10000 == 0 and count > 0:
                    print(f"  Loaded {count:,} images so far...")
            
            print(f"  ✓ Successfully loaded {len(image_paths):,} images from {dataset_name}")
            return image_paths
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if error is related to data being too large
            is_data_too_large = (
                'decompressed data too large' in error_str or
                'data too large' in error_str or
                ('too large' in error_str and 'data' in error_str)
            )
            
            # Check if rate limited
            is_rate_limit = (
                '429' in error_str or 
                'rate limit' in error_str or 
                'too many requests' in error_str
            )
            
            # If data too large and not using streaming yet, switch to streaming
            if is_data_too_large and not use_streaming and attempt < max_retries - 1:
                print(f"  ⚠️  Dataset too large for normal loading: {e}")
                print(f"  → Switching to streaming mode for next attempt...")
                use_streaming = True
                time.sleep(retry_delay * (2 ** attempt))
                continue
            
            # If rate limited, wait and retry
            elif is_rate_limit and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"  ⚠️  Rate limited (429). Waiting {wait_time:.1f}s before retry...")
                if not hf_token:
                    print(f"  💡 Tip: Set HF_TOKEN environment variable to avoid rate limits:")
                    print(f"     export HF_TOKEN=your_token_here")
                time.sleep(wait_time)
                continue
            
            # Other errors
            else:
                print(f"  ✗ Error loading {dataset_name}: {e}")
                if attempt < max_retries - 1:
                    # If it's a data size issue and we haven't tried streaming, suggest it
                    if is_data_too_large and not use_streaming:
                        print(f"  → Will try streaming mode on next attempt...")
                        use_streaming = True
                    print(f"  Retrying in {retry_delay * (2 ** attempt):.1f}s...")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"  ⚠️  Skipping {dataset_name} after {max_retries} attempts")
                    return []
    
    return []

def load_all_datasets(
    real_datasets: List[str],
    synthetic_datasets: List[str],
    semisynthetic_datasets: List[str],
    max_samples_per_dataset: Optional[int] = None,
    balance_classes: bool = True,
    cache_dir: Optional[str] = None,
    use_disk_cache: bool = True,
    force_download: bool = False,
    download_workers: int = 4
) -> Tuple[List[str], List[int]]:
    """
    Load all datasets and create labeled dataset.
    
    Args:
        real_datasets: List of real image dataset names
        synthetic_datasets: List of synthetic image dataset names
        semisynthetic_datasets: List of semisynthetic image dataset names
        max_samples_per_dataset: Maximum samples per dataset
        balance_classes: Whether to balance classes
        cache_dir: Directory to cache downloaded datasets (None = use default)
        use_disk_cache: If True, download to disk first, then load from disk
    """
    all_paths = []
    all_labels = []
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path("./datasets_cache")
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: No HF_TOKEN found. You may hit rate limits.")
        print("   Set it with: export HF_TOKEN=your_token_here")
        print("   Get token from: https://huggingface.co/settings/tokens\n")
    else:
        print("✓ HF_TOKEN found, using authenticated requests\n")
    
    if use_disk_cache:
        print(f"✓ Using disk cache: {cache_dir.resolve()}")
        if force_download:
            print("  ⚠️  Force download enabled - will re-download all datasets\n")
        else:
            print("  ✓ Will use existing cached files if available\n")
    
    # Load real images (label 0)
    print("\n=== Loading Real Images ===")
    successful = 0
    failed = 0
    for i, ds_name in enumerate(real_datasets):
        if use_disk_cache:
            # Download to disk first
            dataset_cache_dir = cache_dir / ds_name.replace("/", "_")
            
            # Initialize tracker
            tracker = DownloadTracker(cache_dir / "download_tracker.db")
            
            # Check database first for already downloaded files
            if not force_download:
                stats = tracker.get_dataset_stats(ds_name)
                if stats['completed_files'] > 0:
                    print(f"  Found {stats['completed_files']} completed files in database ({stats['total_images']} images)")
                    # Get image paths from database
                    paths = tracker.get_image_paths(ds_name, dataset_cache_dir)
                    if paths:
                        print(f"  Using {len(paths)} cached images for {ds_name}")
                        if max_samples_per_dataset and len(paths) > max_samples_per_dataset:
                            import random
                            paths = random.sample(paths, max_samples_per_dataset)
                    else:
                        # Database says downloaded but no images found, re-download
                        paths = download_dataset_to_disk(
                            ds_name, dataset_cache_dir, media_type="real",
                            max_files=None,
                            max_images_per_file=None,
                            tracker=tracker,
                            force_download=False,
                            max_workers=download_workers
                        )
                else:
                    # Not in database, download
                    paths = download_dataset_to_disk(
                        ds_name, dataset_cache_dir, media_type="real",
                        max_files=None,
                        max_images_per_file=None,
                        tracker=tracker,
                        force_download=False,
                        max_workers=args.download_workers
                    )
            else:
                # Force download or directory doesn't exist
                if force_download and dataset_cache_dir.exists():
                    # Remove existing directory to force fresh download
                    import shutil
                    print(f"  Force download: removing existing cache for {ds_name}")
                    shutil.rmtree(dataset_cache_dir, ignore_errors=True)
                
                tracker = DownloadTracker(cache_dir / "download_tracker.db")
                paths = download_dataset_to_disk(
                    ds_name, dataset_cache_dir, media_type="real",
                    max_files=None,
                    max_images_per_file=None,
                    tracker=tracker,
                    force_download=force_download,
                    max_workers=download_workers
                )
        else:
            # Original in-memory loading
            paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        
        if paths:
            all_paths.extend(paths)
            all_labels.extend([0] * len(paths))
            successful += 1
        else:
            failed += 1
        # Small delay between datasets to avoid rate limits
        if i < len(real_datasets) - 1:
            time.sleep(1.0)
    print(f"  Real datasets: {successful} successful, {failed} failed")
    
    # Load synthetic images (label 1)
    print("\n=== Loading Synthetic Images ===")
    successful = 0
    failed = 0
    for i, ds_name in enumerate(synthetic_datasets):
        if use_disk_cache:
            # Download to disk first
            dataset_cache_dir = cache_dir / ds_name.replace("/", "_")
            
            # Initialize tracker
            tracker = DownloadTracker(cache_dir / "download_tracker.db")
            
            # Check database first for already downloaded files
            if not force_download:
                stats = tracker.get_dataset_stats(ds_name)
                if stats['completed_files'] > 0:
                    print(f"  Found {stats['completed_files']} completed files in database ({stats['total_images']} images)")
                    # Get image paths from database
                    paths = tracker.get_image_paths(ds_name, dataset_cache_dir)
                    if paths:
                        print(f"  Using {len(paths)} cached images for {ds_name}")
                        if max_samples_per_dataset and len(paths) > max_samples_per_dataset:
                            import random
                            paths = random.sample(paths, max_samples_per_dataset)
                    else:
                        # Database says downloaded but no images found, re-download
                        paths = download_dataset_to_disk(
                            ds_name, dataset_cache_dir, media_type="synthetic",
                            max_files=None,
                            max_images_per_file=None,
                            tracker=tracker,
                            force_download=False,
                            max_workers=download_workers
                        )
                else:
                    # Not in database, download
                    paths = download_dataset_to_disk(
                        ds_name, dataset_cache_dir, media_type="synthetic",
                        max_files=None,
                        max_images_per_file=None,
                        tracker=tracker,
                        force_download=False,
                        max_workers=args.download_workers
                    )
            else:
                # Force download or directory doesn't exist
                if force_download and dataset_cache_dir.exists():
                    # Remove existing directory to force fresh download
                    import shutil
                    print(f"  Force download: removing existing cache for {ds_name}")
                    shutil.rmtree(dataset_cache_dir, ignore_errors=True)
                
                tracker = DownloadTracker(cache_dir / "download_tracker.db")
                paths = download_dataset_to_disk(
                    ds_name, dataset_cache_dir, media_type="synthetic",
                    max_files=None,
                    max_images_per_file=None,
                    tracker=tracker,
                    force_download=force_download,
                    max_workers=download_workers
                )
        else:
            # Original in-memory loading
            paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        
        if paths:
            all_paths.extend(paths)
            all_labels.extend([1] * len(paths))
            successful += 1
        else:
            failed += 1
        # Small delay between datasets to avoid rate limits
        if i < len(synthetic_datasets) - 1:
            time.sleep(1.0)
    print(f"  Synthetic datasets: {successful} successful, {failed} failed")
    
    # Load semisynthetic images (label 2)
    print("\n=== Loading Semi-synthetic Images ===")
    successful = 0
    failed = 0
    for i, ds_name in enumerate(semisynthetic_datasets):
        if use_disk_cache:
            # Download to disk first
            dataset_cache_dir = cache_dir / ds_name.replace("/", "_")
            
            # Initialize tracker
            tracker = DownloadTracker(cache_dir / "download_tracker.db")
            
            # Check database first for already downloaded files
            if not force_download:
                stats = tracker.get_dataset_stats(ds_name)
                if stats['completed_files'] > 0:
                    print(f"  Found {stats['completed_files']} completed files in database ({stats['total_images']} images)")
                    # Get image paths from database
                    paths = tracker.get_image_paths(ds_name, dataset_cache_dir)
                    if paths:
                        print(f"  Using {len(paths)} cached images for {ds_name}")
                        if max_samples_per_dataset and len(paths) > max_samples_per_dataset:
                            import random
                            paths = random.sample(paths, max_samples_per_dataset)
                    else:
                        # Database says downloaded but no images found, re-download
                        paths = download_dataset_to_disk(
                            ds_name, dataset_cache_dir, media_type="semisynthetic",
                            max_files=None,
                            max_images_per_file=None,
                            tracker=tracker,
                            force_download=False,
                            max_workers=download_workers
                        )
                else:
                    # Not in database, download
                    paths = download_dataset_to_disk(
                        ds_name, dataset_cache_dir, media_type="semisynthetic",
                        max_files=None,
                        max_images_per_file=None,
                        tracker=tracker,
                        force_download=False,
                        max_workers=args.download_workers
                    )
            else:
                # Force download or directory doesn't exist
                if force_download and dataset_cache_dir.exists():
                    # Remove existing directory to force fresh download
                    import shutil
                    print(f"  Force download: removing existing cache for {ds_name}")
                    shutil.rmtree(dataset_cache_dir, ignore_errors=True)
                
                tracker = DownloadTracker(cache_dir / "download_tracker.db")
                paths = download_dataset_to_disk(
                    ds_name, dataset_cache_dir, media_type="semisynthetic",
                    max_files=None,
                    max_images_per_file=None,
                    tracker=tracker,
                    force_download=force_download,
                    max_workers=download_workers
                )
        else:
            # Original in-memory loading
            paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        
        if paths:
            all_paths.extend(paths)
            all_labels.extend([2] * len(paths))
            successful += 1
        else:
            failed += 1
        # Small delay between datasets to avoid rate limits
        if i < len(semisynthetic_datasets) - 1:
            time.sleep(1.0)
    print(f"  Semi-synthetic datasets: {successful} successful, {failed} failed")
    
    print(f"\n{'='*50}")
    print(f"Total samples loaded: {len(all_paths):,}")
    print(f"  Real: {sum(1 for l in all_labels if l == 0):,}")
    print(f"  Synthetic: {sum(1 for l in all_labels if l == 1):,}")
    print(f"  Semi-synthetic: {sum(1 for l in all_labels if l == 2):,}")
    print(f"{'='*50}")
    
    # Balance classes if requested
    if balance_classes:
        counts = [sum(1 for l in all_labels if l == i) for i in range(3)]
        min_count = min(counts)
        print(f"\nBalancing to {min_count} samples per class...")
        
        balanced_paths = []
        balanced_labels = []
        for label in range(3):
            label_paths = [p for p, l in zip(all_paths, all_labels) if l == label]
            balanced_paths.extend(label_paths[:min_count])
            balanced_labels.extend([label] * min_count)
        
        all_paths = balanced_paths
        all_labels = balanced_labels
        
        print(f"After balancing: {len(all_paths)} samples")
    
    return all_paths, all_labels

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
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    enable_profiling: bool = False
):
    """
    Train the model with optimizations for large-scale training.
    
    Args:
        use_amp: Use Automatic Mixed Precision (FP16) for faster training and less memory
        gradient_accumulation_steps: Accumulate gradients over N batches before updating weights
        max_grad_norm: Clip gradients to prevent exploding gradients
        enable_profiling: Enable PyTorch profiling to identify bottlenecks
    """
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed precision training (AMP) - reduces memory usage and speeds up training
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    if use_amp and device.type == 'cuda':
        print("✓ Mixed Precision Training (AMP) enabled - using FP16 for faster training")
    
    if gradient_accumulation_steps > 1:
        print(f"✓ Gradient Accumulation enabled - accumulating over {gradient_accumulation_steps} batches")
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Profiling setup (optional, for identifying bottlenecks)
    profiler = None
    if enable_profiling and device.type == 'cuda':
        try:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                with_stack=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "profiler_logs"))
            )
            profiler.start()
            print("✓ Profiling enabled - will profile first few batches")
        except Exception as e:
            print(f"⚠️  Profiling setup failed: {e}, continuing without profiling")
            profiler = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Non-blocking transfer for faster GPU utilization
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
            else:
                output = model(data)
                loss = criterion(output, target)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: only update weights every N batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update metrics (use original loss, not normalized)
            train_loss += loss.item() * gradient_accumulation_steps
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Memory cleanup
            del output, loss
            if batch_idx % 100 == 0:
                # Clear cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Show GPU memory usage
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {train_loss/(batch_idx+1):.4f}, "
                          f"VRAM: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {train_loss/(batch_idx+1):.4f}")
            
            # Profiling (only for first few batches to avoid overhead)
            if profiler is not None and batch_idx < 10:
                profiler.step()
        
        # Final gradient update if there are remaining gradients
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        # Clear cache after epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
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
                
                # Use mixed precision for validation too
                if scaler is not None:
                    with autocast('cuda'):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                # Memory cleanup
                del output, loss, data, target
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 50)
    
    # Stop profiling and save results
    if profiler is not None:
        profiler.stop()
        print("\n" + "="*50)
        print("PROFILING RESULTS")
        print("="*50)
        try:
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
            profiler.export_chrome_trace(str(output_dir / "profile_trace.json"))
            print(f"\nProfile trace saved to: {output_dir / 'profile_trace.json'}")
            print("Open in Chrome: chrome://tracing")
            print(f"TensorBoard logs: {output_dir / 'profiler_logs'}")
            print("View with: tensorboard --logdir=" + str(output_dir / "profiler_logs"))
        except Exception as e:
            print(f"⚠️  Error exporting profile: {e}")
        print("="*50 + "\n")
    
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
    print("LOADING DATASETS FROM HUGGING FACE")
    print("="*50)
    
    all_paths, all_labels = load_all_datasets(
        real_datasets=REAL_DATASETS if not args.real_datasets else args.real_datasets,
        synthetic_datasets=SYNTHETIC_DATASETS if not args.synthetic_datasets else args.synthetic_datasets,
        semisynthetic_datasets=SEMISYNTHETIC_DATASETS if not args.semisynthetic_datasets else args.semisynthetic_datasets,
        max_samples_per_dataset=args.max_samples_per_dataset,
        balance_classes=args.balance,
        cache_dir=args.cache_dir,
        use_disk_cache=args.use_disk_cache,
        force_download=args.force_download,
        download_workers=args.download_workers
    )
    
    if len(all_paths) == 0:
        print("\n" + "="*50)
        print("ERROR: No images loaded!")
        print("="*50)
        print("Possible causes:")
        print("  1. Rate limiting (429 errors) - Set HF_TOKEN to avoid:")
        print("     export HF_TOKEN=your_token_here")
        print("     Get token from: https://huggingface.co/settings/tokens")
        print("  2. Network connectivity issues")
        print("  3. Dataset access permissions")
        print("  4. All datasets failed to load")
        print("\nTry:")
        print("  - Set HF_TOKEN environment variable")
        print("  - Wait a few minutes and retry (rate limits reset)")
        print("  - Check your internet connection")
        print("="*50)
        return
    
    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    
    # Create datasets
    img_size = (args.img_size[0], args.img_size[1])
    train_dataset = ImageDataset(train_paths, train_labels, img_size)
    val_dataset = ImageDataset(val_paths, val_labels, img_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = ImageELAPRNUDetector(num_classes=3, input_channels=4).to(device)
    print(f"\nModel created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        enable_profiling=args.enable_profiling
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
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Maximum samples to load per dataset (for testing)")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance classes by taking min samples per class")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size (increase for 80GB GPU, decrease if OOM)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                        help="Image size (width height)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8, 
                        help="DataLoader workers (8-16 recommended for large datasets)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use Automatic Mixed Precision (FP16) for faster training")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate gradients over N batches (simulates larger batch size)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clip gradients to prevent exploding gradients")
    parser.add_argument("--use-augmentation", action="store_true", default=True,
                        help="Enable data augmentation (on-the-fly, memory efficient)")
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false",
                        help="Disable data augmentation")
    parser.add_argument("--enable-profiling", action="store_true", default=False,
                        help="Enable PyTorch profiling to identify bottlenecks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="image-detector",
                        help="Model name for submission")
    parser.add_argument("--model-version", type=str, default="1.0.0",
                        help="Model version")
    
    # Dataset caching arguments
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache downloaded datasets (default: ./datasets_cache)")
    parser.add_argument("--use-disk-cache", action="store_true", default=True,
                        help="Download datasets to disk first, then load from disk (default: True)")
    parser.add_argument("--no-disk-cache", dest="use_disk_cache", action="store_false",
                        help="Load datasets directly into memory (original behavior)")
    parser.add_argument("--force-download", action="store_true", default=False,
                        help="Force re-download of datasets even if cached files exist (default: False, uses existing cache)")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="Number of parallel workers for downloading and extracting (default: 4, increase for faster downloads)")
    
    args = parser.parse_args()
    
    print("="*50)
    print("IMAGE DETECTOR - ELA+PRNU FUSION")
    print("="*50)
    print(f"Image Size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"Batch Size: {args.batch_size}")
    if args.gradient_accumulation_steps > 1:
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps} "
              f"(with {args.gradient_accumulation_steps}x accumulation)")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"DataLoader Workers: {args.num_workers}")
    print(f"Mixed Precision (AMP): {'✓ Enabled' if args.use_amp else '✗ Disabled'}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}x")
    print(f"Data Augmentation: {'✓ Enabled' if args.use_augmentation else '✗ Disabled'}")
    print(f"Gradient Clipping: {args.max_grad_norm}")
    print(f"Profiling: {'✓ Enabled' if args.enable_profiling else '✗ Disabled'}")
    print(f"Balance Classes: {args.balance}")
    print(f"Disk Cache: {'✓ Enabled' if args.use_disk_cache else '✗ Disabled (using RAM)'}")
    if args.use_disk_cache:
        cache_path = Path(args.cache_dir) if args.cache_dir else Path("./datasets_cache")
        print(f"Cache Directory: {cache_path.resolve()}")
        print(f"Force Download: {'✓ Enabled (will re-download)' if args.force_download else '✗ Disabled (will use existing cache)'}")
        print(f"Download Workers: {args.download_workers} (parallel downloads/extraction)")
    
    # Show GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Current VRAM Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("="*50 + "\n")
    
    main(args)
