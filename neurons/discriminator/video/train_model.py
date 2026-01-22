"""
Video Detector Training Script
Loads video datasets from Hugging Face and trains a multiclass classifier (real, synthetic, semisynthetic)
using ELA+PRNU fusion features on video frames. Outputs model in Safetensors format for submission.
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
    from .model import VideoELAPRNUDetector
    from .preprocessing import extract_prnu_enhanced, extract_ela_enhanced, extract_frames_from_video
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from model import VideoELAPRNUDetector
    from preprocessing import extract_prnu_enhanced, extract_ela_enhanced, extract_frames_from_video

# ---------------------------
# GPU Configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset Configuration
# ---------------------------
# Real video datasets
REAL_DATASETS = [
    "bitmind/bm-eidon-video",
    "shangxd/imagenet-vidvrd",
    "nkp37/OpenVid-1M",
    "facebook/PE-Video",
]

# Synthetic video datasets
SYNTHETIC_DATASETS = [
    "Rapidata/text-2-video-human-preferences-veo3",
    "Rapidata/text-2-video-human-preferences-veo2",
    "bitmind/aura-video",
    "bitmind/aislop-videos",
]

# Semi-synthetic video datasets
SEMISYNTHETIC_DATASETS = [
    "bitmind/semisynthetic-video",
]

# ---------------------------
# Utils
# ---------------------------
def make_feature(rgb: np.ndarray) -> np.ndarray:
    """Create fusion feature: ELA (3 channels) + PRNU (1 channel) = 4 channels."""
    prnu = extract_prnu_enhanced(rgb)  # Single channel
    ela = extract_ela_enhanced(rgb)     # 3 channels (RGB)
    feat = np.concatenate([ela, prnu[..., None]], axis=-1)
    return feat.astype(np.float32)

# ---------------------------
# Dataset Class
# ---------------------------
class VideoDataset(Dataset):
    """Dataset class for loading videos from Hugging Face datasets."""
    
    def __init__(self, video_paths: List[str], labels: List[int], img_size: Tuple[int, int], num_frames: int = 8):
        self.video_paths = video_paths
        self.labels = labels
        self.img_size = img_size
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Handle different path formats
            if isinstance(path, dict):
                # Hugging Face dataset format
                video_path = path.get('video', path.get('path', path.get('file', None)))
                if video_path is None:
                    video_path = list(path.values())[0]
            else:
                video_path = path
            
            # Extract frames
            frames = extract_frames_from_video(video_path, self.num_frames, self.img_size)
            
            # Extract features for each frame
            frame_features = []
            for frame_rgb in frames:
                feat = make_feature(frame_rgb)  # Shape: (H, W, 4)
                # Convert to tensor format: (4, H, W)
                feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
                frame_features.append(feat_tensor)
            
            # Stack frames: [num_frames, 4, H, W]
            video_features = torch.stack(frame_features, dim=0)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return video_features, label_tensor
        
        except Exception as e:
            print(f"Error loading video {path}: {e}")
            # Return zero tensor on error
            C = 4  # ELA (3) + PRNU (1)
            H, W = self.img_size[1], self.img_size[0]
            return torch.zeros(self.num_frames, C, H, W), torch.tensor(0, dtype=torch.long)

# ---------------------------
# Dataset Loading from Hugging Face
# ---------------------------
def load_hf_dataset(dataset_name: str, split: str = "train", max_samples: int = None):
    """Load video dataset from Hugging Face."""
    try:
        print(f"Loading {dataset_name} (split: {split})...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Extract video paths
        video_paths = []
        for item in dataset:
            if isinstance(item, dict):
                # Try to find video field
                for key in ['video', 'path', 'file', 'file_path']:
                    if key in item:
                        video_paths.append(item)
                        break
                else:
                    # Use the first value
                    first_val = list(item.values())[0]
                    if isinstance(first_val, str):
                        video_paths.append(first_val)
            elif isinstance(item, str):
                video_paths.append(item)
        
        print(f"Loaded {len(video_paths)} videos from {dataset_name}")
        return video_paths
    
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
    """Load all video datasets and create labeled dataset."""
    
    all_paths = []
    all_labels = []
    
    # Load real videos (label 0)
    print("\n=== Loading Real Videos ===")
    for ds_name in real_datasets:
        paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        all_paths.extend(paths)
        all_labels.extend([0] * len(paths))
    
    # Load synthetic videos (label 1)
    print("\n=== Loading Synthetic Videos ===")
    for ds_name in synthetic_datasets:
        paths = load_hf_dataset(ds_name, split="train", max_samples=max_samples_per_dataset)
        all_paths.extend(paths)
        all_labels.extend([1] * len(paths))
    
    # Load semisynthetic videos (label 2)
    print("\n=== Loading Semi-synthetic Videos ===")
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
            # data shape: [B, T, C, H, W] -> convert to [B, C, T, H, W] for Conv3d
            data = data.permute(0, 2, 1, 3, 4).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
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
                data = data.permute(0, 2, 1, 3, 4).to(device)
                target = target.to(device)
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
            data = data.permute(0, 2, 1, 3, 4).to(device)
            target = target.to(device)
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
    output_dir = Path(f"./outputs/video_detector/{timestamp}")
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
        print("Error: No videos loaded!")
        return
    
    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    
    # Create datasets
    img_size = (args.img_size[0], args.img_size[1])
    train_dataset = VideoDataset(train_paths, train_labels, img_size, args.num_frames)
    val_dataset = VideoDataset(val_paths, val_labels, img_size, args.num_frames)
    
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
    model = VideoELAPRNUDetector(num_classes=3, input_channels=4, num_frames=args.num_frames).to(device)
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
        'modality': 'video',
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
    parser = argparse.ArgumentParser(description="Train Video Detector with ELA+PRNU Fusion")
    
    # Dataset arguments
    parser.add_argument("--real-datasets", type=str, nargs="+", default=None,
                        help="Custom list of real video datasets (default: use predefined)")
    parser.add_argument("--synthetic-datasets", type=str, nargs="+", default=None,
                        help="Custom list of synthetic video datasets")
    parser.add_argument("--semisynthetic-datasets", type=str, nargs="+", default=None,
                        help="Custom list of semisynthetic video datasets")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Maximum samples to load per dataset (for testing)")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance classes by taking min samples per class")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (smaller for videos)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                        help="Frame size (width height)")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per video")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (fewer for videos)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="video-detector",
                        help="Model name for submission")
    parser.add_argument("--model-version", type=str, default="1.0.0",
                        help="Model version")
    
    args = parser.parse_args()
    
    print("="*50)
    print("VIDEO DETECTOR - ELA+PRNU FUSION")
    print("="*50)
    print(f"Frame Size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"Number of Frames: {args.num_frames}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Balance Classes: {args.balance}")
    print("="*50 + "\n")
    
    main(args)
