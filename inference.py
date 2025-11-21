"""
inference.py

Run colorization inference using a trained U-Net saved to `best_unet_colorization.pt`.

Usage examples:
  python inference.py --image path/to/grayscale.png
  python inference.py --image path/to/grayscale.png --model-path best_unet_colorization.pt --out out.png --upscale

Behavior:
- Loads the U-Net architecture from `colorization_unet.UNet` and the saved weights.
- Preprocesses the input image: converts to grayscale, resizes to 32x32, converts to tensor in [0,1].
- Runs the model on the preprocessed tensor and produces an RGB prediction in [0,1].
- Saves a side-by-side image showing: input grayscale, predicted color, and (optionally) upscaled predicted color.

Notes:
- The script automatically selects CUDA if available and prints the device used.
- Requires `colorization_unet.py` (UNet implementation) and the saved weights file.
"""

import os
import argparse
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# Import the UNet definition from the training script so the architecture matches.
# This file (`colorization_unet.py`) must be in the same folder or on PYTHONPATH.
from colorization_unet import UNet


def get_device():
    """Return the device (cuda if available else cpu) and print it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate UNet and load state dict from `model_path`."""
    model = UNet(in_channels=1, out_channels=3)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: str, size=(32, 32)) -> (torch.Tensor, tuple, Image.Image):
    """Load image from disk and return (tensor, original_size, pil_image).

    - tensor is shaped [1,1,H,W] with values in [0,1].
    - original_size is (width, height) of the input image.
    - pil_image is the opened PIL image (used for optional upscaling reference).
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size),
        transforms.ToTensor(),  # maps to [0,1]
    ])

    tensor = transform(img).unsqueeze(0)  # [1,1,H,W]
    return tensor, (orig_w, orig_h), img


def run_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run model on input_tensor and return predicted RGB tensor [1,3,H,W]."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    # clamp to [0,1]
    pred = pred.clamp(0.0, 1.0)
    return pred.cpu()


def build_and_save_grid(gray_tensor: torch.Tensor, pred_tensor: torch.Tensor, out_path: str, orig_size: tuple, upscale: bool = False):
    """Create a side-by-side image and save it.

    - gray_tensor: [1,1,H,W]
    - pred_tensor: [1,3,H,W]
    - orig_size: (orig_w, orig_h)
    - upscale: if True, both gray and predicted images are resized to original image size
      so the saved grid shows the upscaled prediction as the third column.
    """
    # Bring to CPU and remove batch dim
    gray = gray_tensor.squeeze(0)  # 1xH xW
    pred = pred_tensor.squeeze(0)  # 3xH xW

    if upscale:
        # Resize to original image size for better visualization
        orig_w, orig_h = orig_size
        # torchvision.transforms.functional.resize expects size as (H, W)
        target_size = (orig_h, orig_w)
        gray_vis = torchvision.transforms.functional.resize(gray, size=target_size)
        # replicate gray to 3 channels for display
        gray_vis = gray_vis.repeat(3, 1, 1)
        pred_vis_small = torchvision.transforms.functional.resize(pred, size=target_size)
        # Also include a smaller (model resolution) prediction for comparison
        pred_small = pred
        images = [gray_vis, pred_small, pred_vis_small]
        # Resize all to same spatial size for make_grid: we will pad/rescale pred_small
        # to the same target_size so shapes match for the grid
        pred_small_resized = torchvision.transforms.functional.resize(pred_small, size=target_size)
        images = [gray_vis, pred_small_resized, pred_vis_small]
    else:
        # Keep model resolution (32x32)
        gray_vis = gray.repeat(3, 1, 1)
        pred_vis = pred
        images = [gray_vis, pred_vis]

    grid = make_grid(images, nrow=len(images), pad_value=1)
    save_image(grid, out_path)
    print(f"Saved inference result to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run colorization inference using a trained U-Net.")
    parser.add_argument("--image", type=str, required=True, help="Path to input grayscale image")
    parser.add_argument("--model-path", type=str, default="best_unet_colorization.pt", help="Path to saved model weights")
    parser.add_argument("--out", type=str, default="inference_result.png", help="Output path for side-by-side image")
    parser.add_argument("--upscale", action="store_true", help="Also upscale prediction to original image size and include in the output grid")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Load model
    model = load_model(args.model_path, device)

    # Preprocess image
    input_tensor, orig_size, pil_img = preprocess_image(args.image, size=(32, 32))

    # Run inference
    pred = run_inference(model, input_tensor, device)

    # Save side-by-side result
    build_and_save_grid(input_tensor, pred, args.out, orig_size, upscale=args.upscale)


if __name__ == "__main__":
    main()
