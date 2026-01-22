"""
Script to package trained image model for submission.
Creates a zip file with model_config.yaml, model.py, and model.safetensors.
"""

import argparse
import zipfile
from pathlib import Path


def package_image_model(
    model_dir: str,
    output_zip: str = None,
    model_name: str = "image_detector"
):
    """
    Package image model files into a zip for submission.
    
    Args:
        model_dir: Directory containing model files (model.safetensors, model_config.yaml)
        output_zip: Output zip file path (default: {model_name}.zip in model_dir)
        model_name: Name for the output zip file
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Required files
    safetensors_file = model_dir / "model.safetensors"
    config_file = model_dir / "model_config.yaml"
    model_py_file = Path(__file__).parent / "model.py"
    
    # Check if files exist
    missing_files = []
    if not safetensors_file.exists():
        missing_files.append("model.safetensors")
    if not config_file.exists():
        missing_files.append("model_config.yaml")
    if not model_py_file.exists():
        missing_files.append("model.py")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing_files)}\n"
            f"  Expected in: {model_dir}\n"
            f"  model.py should be in: {model_py_file.parent}"
        )
    
    # Create output zip path
    if output_zip is None:
        output_zip = model_dir / f"{model_name}.zip"
    else:
        output_zip = Path(output_zip)
    
    # Create zip file
    print(f"Packaging image model from: {model_dir}")
    print(f"Output zip: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model.safetensors
        print(f"  Adding: model.safetensors")
        zipf.write(safetensors_file, "model.safetensors")
        
        # Add model_config.yaml
        print(f"  Adding: model_config.yaml")
        zipf.write(config_file, "model_config.yaml")
        
        # Add model.py
        print(f"  Adding: model.py")
        zipf.write(model_py_file, "model.py")
    
    print(f"\n✓ Image model packaged successfully: {output_zip.resolve()}")
    print(f"  File size: {output_zip.stat().st_size / (1024*1024):.2f} MB")
    print(f"\nReady to submit with:")
    print(f"  gascli d push --image-model {output_zip.name} --wallet-name <your_wallet> --wallet-hotkey <your_hotkey>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package image model for submission")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing model files (model.safetensors, model_config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zip file path (default: {model_name}.zip in model_dir)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="image_detector",
        help="Model name for zip file (default: image_detector)"
    )
    
    args = parser.parse_args()
    
    try:
        package_image_model(args.model_dir, args.output, args.name)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
