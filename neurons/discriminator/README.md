# Discriminative Mining - Model Training

This directory contains training scripts and models for discriminative mining (image and video detection).

## Structure

```
discriminator/
├── image/
│   ├── model.py              # Image model architecture (for submission)
│   ├── train_model.py        # Image training script
│   ├── preprocessing.py      # Image feature extraction (ELA+PRNU)
│   └── package_model.py      # Package image model for submission
├── video/
│   ├── model.py              # Video model architecture (for submission)
│   ├── train_model.py        # Video training script
│   ├── preprocessing.py      # Video frame extraction and feature extraction
│   └── package_model.py      # Package video model for submission
└── README.md                 # This file
```

## Image Detector

### Training

Train an image detector model:

```bash
# From project root
python -m neurons.discriminator.image.train_model \
    --batch-size 16 \
    --img-size 256 256 \
    --epochs 20 \
    --lr 1e-3 \
    --model-name "my-image-detector" \
    --model-version "1.0.0"
```

Or run directly:

```bash
cd neurons/discriminator/image
python train_model.py --batch-size 16 --epochs 20
```

### Packaging

Package the trained model for submission:

```bash
python -m neurons.discriminator.image.package_model \
    outputs/image_detector/YYYY-MM-DD_HH-MM-SS \
    --output image_detector.zip
```

## Video Detector

### Training

Train a video detector model:

```bash
# From project root
python -m neurons.discriminator.video.train_model \
    --batch-size 4 \
    --img-size 256 256 \
    --num-frames 8 \
    --epochs 20 \
    --lr 1e-3 \
    --model-name "my-video-detector" \
    --model-version "1.0.0"
```

Or run directly:

```bash
cd neurons/discriminator/video
python train_model.py --batch-size 4 --num-frames 8 --epochs 20
```

### Packaging

Package the trained model for submission:

```bash
python -m neurons.discriminator.video.package_model \
    outputs/video_detector/YYYY-MM-DD_HH-MM-SS \
    --output video_detector.zip
```

## Submitting Models

After packaging, submit your models:

```bash
gascli d push \
    --image-model image_detector.zip \
    --video-model video_detector.zip \
    --wallet-name your_wallet_name \
    --wallet-hotkey your_hotkey_name
```

## Model Architecture

### Image Model
- **Input**: 4-channel features (ELA: 3 channels, PRNU: 1 channel)
- **Architecture**: 3-layer 2D CNN with BatchNorm and Dropout
- **Output**: 3 classes (real, synthetic, semisynthetic)

### Video Model
- **Input**: 8 frames × 4-channel features per frame
- **Architecture**: 3D CNN for temporal-spatial features
- **Output**: 3 classes (real, synthetic, semisynthetic)

## Feature Extraction

Both models use ELA (Error Level Analysis) + PRNU (Photo Response Non-Uniformity) fusion:

- **ELA**: Recompresses image at quality 95 and computes difference to highlight compression artifacts
- **PRNU**: Uses wavelet decomposition for denoising and extracts noise residual pattern

## Datasets

Models automatically load datasets from Hugging Face (bitmind organization):

### Image Datasets
- Real: `bitmind/ffhq-256`, `bitmind/celeb-a-hq`, etc.
- Synthetic: `bitmind/bm-sdxl`, `bitmind/bm-mobius`, etc.
- Semi-synthetic: `bitmind/face-swap`, etc.

### Video Datasets
- Real: `bitmind/bm-eidon-video`, `facebook/PE-Video`, etc.
- Synthetic: `bitmind/aura-video`, `bitmind/aislop-videos`, etc.
- Semi-synthetic: `bitmind/semisynthetic-video`

See training scripts for full dataset lists.

## Output Structure

After training, models are saved to:

```
outputs/
├── image_detector/
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── model.safetensors
│       ├── model_config.yaml
│       ├── best_model.pt
│       ├── training_history.csv
│       └── ...
└── video_detector/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── model.safetensors
        ├── model_config.yaml
        ├── best_model.pt
        ├── training_history.csv
        └── ...
```

## Requirements

Install training dependencies:

```bash
pip install -r requirements_training.txt
```

Main dependencies:
- PyWavelets (for PRNU extraction)
- scipy (for signal processing)
- safetensors (for model saving)
- pyyaml (for config files)
- av (for video processing)
- dask[dataframe] (for loading Hugging Face datasets via hf:// protocol)
- fsspec (for filesystem abstraction)
- hf-file-system (for Hugging Face filesystem support)

Most base dependencies are already in `pyproject.toml`.

**Note:** You need to authenticate with Hugging Face first:
```bash
huggingface-cli login
# or
hf auth login
```

## References

- [Discriminative Mining Guide](../../docs/Discriminative-Mining.md)
- [Incentive Mechanism](../../docs/Incentive.md)
- [BitMind Datasets](https://huggingface.co/datasets/bitmind)
