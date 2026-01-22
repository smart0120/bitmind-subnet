"""
Image Detector Training Script
Loads datasets from Hugging Face and trains a multiclass classifier (real, synthetic, semisynthetic)
using ELA+PRNU fusion features. Outputs model in Safetensors format for submission.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from safetensors.torch import save_file
import yaml

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
    "bitmind/bm-eidon-image",
    "bitmind/bm-real",
    "bitmind/open-image-v7-256",
    "bitmind/celeb-a-hq",
    "bitmind/ffhq-256",
    "bitmind/MS-COCO-unique-256",
    "bitmind/AFHQ",
    "bitmind/lfw",
    "bitmind/caltech-256",
    "bitmind/caltech-101",
    "bitmind/dtd",
]

# Synthetic image datasets
SYNTHETIC_DATASETS = [
    "bitmind/JourneyDB",
    "bitmind/GenImage_MidJourney",
    "bitmind/bm-aura-imagegen",
    "bitmind/bm-imagine",
    "bitmind/ffhq-256___stable-diffusion-xl-base-1.0",
    "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0",
    "bitmind/celeb-a-hq___FLUX.1-dev",
    "bitmind/ffhq-256___FLUX.1-dev",
    "bitmind/bm-sdxl",
    "bitmind/bm-mobius",
    "bitmind/bm-realvisxl",
    "bitmind/bm-diffusion",
]

# Semi-synthetic image datasets
SEMISYNTHETIC_DATASETS = [
    "bitmind/face-swap",
    "bitmind/ffhq-256_training_faces",
    "bitmind/celeb-a-hq_training_faces",
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
    
    def __init__(self, image_paths: List[str], labels: List[int], img_size: Tuple[int, int]):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
    
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
                    for key in ['image', 'img', 'file_path', 'file', 'path']:
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
# Dataset Loading from Hugging Face
# ---------------------------
def load_hf_dataset(dataset_name: str, split: str = "train", max_samples: int = None):
    """Load dataset from Hugging Face."""
    try:
        print(f"Loading {dataset_name} (split: {split})...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Extract image paths
        image_paths = []
        for item in dataset:
            if isinstance(item, dict):
                if 'image' in item:
                    # Image is already loaded as PIL Image
                    image_paths.append(item)
                elif 'path' in item:
                    image_paths.append(item['path'])
                else:
                    # Try to find image field
                    found = False
                    for key in ['image', 'img', 'file_path', 'file', 'path']:
                        if key in item:
                            val = item[key]
                            if isinstance(val, Image.Image):
                                image_paths.append(item)
                            elif isinstance(val, str):
                                image_paths.append(val)
                            found = True
                            break
                    if not found:
                        # Use the first value if it's a path-like string or PIL Image
                        first_val = list(item.values())[0]
                        if isinstance(first_val, Image.Image):
                            image_paths.append(item)
                        elif isinstance(first_val, str):
                            image_paths.append(first_val)
            elif isinstance(item, Image.Image):
                # Direct PIL Image
                image_paths.append(item)
            elif isinstance(item, str):
                # Direct path string
                image_paths.append(item)
        
        print(f"Loaded {len(image_paths)} images from {dataset_name}")
        return image_paths
    
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return []

def load_all_datasets(
    real_datasets: List[str],
    synthetic_datasets: List[str],
    semisynthetic_datasets: List[str],
    max_samples_per_dataset: int = None,
    balance_classes: bool = True
) -> Tuple[List[str], List[int]]:
    """Load all datasets and create labeled dataset."""
    
    all_paths = []
    all_labels = []
    
    # Load real images (label 0)
    print("\n=== Loading Real Images ===")
    for ds_name in real_datasets:
        paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        all_paths.extend(paths)
        all_labels.extend([0] * len(paths))
    
    # Load synthetic images (label 1)
    print("\n=== Loading Synthetic Images ===")
    for ds_name in synthetic_datasets:
        paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        all_paths.extend(paths)
        all_labels.extend([1] * len(paths))
    
    # Load semisynthetic images (label 2)
    print("\n=== Loading Semi-synthetic Images ===")
    for ds_name in semisynthetic_datasets:
        paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        all_paths.extend(paths)
        all_labels.extend([2] * len(paths))
    
    print(f"\nTotal samples: {len(all_paths)}")
    print(f"  Real: {sum(1 for l in all_labels if l == 0)}")
    print(f"  Synthetic: {sum(1 for l in all_labels if l == 1)}")
    print(f"  Semi-synthetic: {sum(1 for l in all_labels if l == 2)}")
    
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
    device: torch.device
):
    """Train the model."""
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
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
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 100 == 0:
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
                data, target = data.to(device), target.to(device)
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
    print("LOADING DATASETS FROM HUGGING FACE")
    print("="*50)
    
    all_paths, all_labels = load_all_datasets(
        real_datasets=REAL_DATASETS if not args.real_datasets else args.real_datasets,
        synthetic_datasets=SYNTHETIC_DATASETS if not args.synthetic_datasets else args.synthetic_datasets,
        semisynthetic_datasets=SEMISYNTHETIC_DATASETS if not args.semisynthetic_datasets else args.semisynthetic_datasets,
        max_samples_per_dataset=args.max_samples_per_dataset,
        balance_classes=args.balance
    )
    
    if len(all_paths) == 0:
        print("Error: No images loaded!")
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
        device=device
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
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                        help="Image size (width height)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
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
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Balance Classes: {args.balance}")
    print("="*50 + "\n")
    
    main(args)
