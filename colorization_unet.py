"""
colorization_unet.py

Refactored single-file PyTorch project for automatic image colorization on CIFAR-10.

This file keeps the original script's behavior but is reorganized for clarity:
- `Model` section: U-Net implementation.
- `Dataset` section: CIFAR-10 colorization dataset wrapper.
- `Training/Eval` section: training loop, validation, metrics, and sample saving.

The script automatically selects CUDA if available and prints the chosen device.
Functionality is unchanged: it trains a U-Net to map 1-channel grayscale inputs
to 3-channel color outputs, evaluates SSIM/PSNR, saves sample grids each epoch,
and checkpoints the best model by validation SSIM.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


# Device selection: automatically use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
# Model: U-Net implementation
########################################


class DoubleConv(nn.Module):
    """Two consecutive conv layers with BatchNorm and ReLU.

    Reusable building block for the U-Net encoder/decoder.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """Simple U-Net mapping grayscale (1 channel) -> RGB (3 channels).

    The architecture mirrors the original script: a few encoder blocks, a
    bottleneck, and symmetric decoder blocks with transpose convolutions.
    Output uses a `sigmoid` to constrain values to [0, 1].
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, features=(64, 128, 256)):
        super().__init__()
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.encs.append(DoubleConv(in_channels, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling)
        rev_features = list(reversed(features))
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_ch = features[-1] * 2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2))
            # After concatenation, channel dim = skip_channels + upconv_channels
            self.decs.append(DoubleConv(in_ch, f))
            in_ch = f

        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
            x = upconv(x)
            # If shapes mismatch (unlikely for CIFAR-10 32x32), resize to skip spatial
            if x.shape[2:] != skip.shape[2:]:
                x = torchvision.transforms.functional.resize(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = dec(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x


########################################
# Dataset wrapper
########################################


class CIFAR10Colorization(torch.utils.data.Dataset):
    """Dataset that returns (grayscale_input, color_target) pairs from CIFAR-10.

    - `grayscale_input` is a 1xHxW tensor in [0,1].
    - `color_target` is a 3xHxW tensor in [0,1].
    """

    def __init__(self, root: str, train: bool = True, transform_color=None, transform_gray=None, download: bool = True):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform_color = transform_color or transforms.ToTensor()
        self.transform_gray = transform_gray or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        color = self.transform_color(img)
        gray = self.transform_gray(img)
        return gray, color


########################################
# Metrics, saving, and helpers
########################################


def compute_metrics_batch(preds: torch.Tensor, targets: torch.Tensor):
    """Compute average SSIM and PSNR for a batch.

    Both `preds` and `targets` are expected as tensors in [B,3,H,W] with values in [0,1].
    Returns (avg_ssim, avg_psnr).
    """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    batch_ssim = 0.0
    batch_psnr = 0.0
    b = preds.shape[0]
    for i in range(b):
        p = np.transpose(preds[i], (1, 2, 0))
        t = np.transpose(targets[i], (1, 2, 0))
        p = np.clip(p, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)
        try:
            s = ssim_metric(t, p, multichannel=True, data_range=1.0)
        except TypeError:
            s = ssim_metric(t, p, data_range=1.0)
        pnr = psnr_metric(t, p, data_range=1.0)
        batch_ssim += s
        batch_psnr += pnr
    return batch_ssim / b, batch_psnr / b


def save_sample_grid(gray_batch: torch.Tensor, pred_batch: torch.Tensor, gt_batch: torch.Tensor, epoch: int, out_dir: str, n_samples: int = 6):
    """Save a grid of sample images showing (gray, prediction, ground-truth) triplets.

    The grid layout has 3 columns per sample: grayscale (replicated to 3 ch),
    predicted color, and ground-truth color.
    """
    os.makedirs(out_dir, exist_ok=True)
    gray_batch = gray_batch.detach().cpu()
    pred_batch = pred_batch.detach().cpu()
    gt_batch = gt_batch.detach().cpu()
    n = min(n_samples, gray_batch.shape[0])
    images = []
    for i in range(n):
        g = gray_batch[i]
        g3 = g.repeat(3, 1, 1)
        p = pred_batch[i]
        t = gt_batch[i]
        images.extend([g3, p, t])
    grid = make_grid(images, nrow=3, pad_value=1)
    save_image(grid, os.path.join(out_dir, f"epoch_{epoch:03d}.png"))


########################################
# Training and validation loops
########################################


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    """Run one training epoch and return average loss."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for gray, color in pbar:
        gray = gray.to(device)
        color = color.to(device)
        optimizer.zero_grad()
        pred = model(gray)
        loss = criterion(pred, color)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * gray.size(0)
        pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * loader.batch_size))
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    """Run validation and return (loss, avg_ssim, avg_psnr)."""
    model.eval()
    running_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total = 0
    pbar = tqdm(loader, desc="val", leave=False)
    with torch.no_grad():
        for gray, color in pbar:
            gray = gray.to(device)
            color = color.to(device)
            pred = model(gray)
            loss = criterion(pred, color)
            running_loss += loss.item() * gray.size(0)
            ssim_val, psnr_val = compute_metrics_batch(pred, color)
            batch_size = gray.size(0)
            total_ssim += ssim_val * batch_size
            total_psnr += psnr_val * batch_size
            total += batch_size
            pbar.set_postfix(val_loss=running_loss / total)
    avg_loss = running_loss / total
    avg_ssim = total_ssim / total
    avg_psnr = total_psnr / total
    return avg_loss, avg_ssim, avg_psnr


########################################
# CLI and main
########################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="path to CIFAR-10 data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="samples")
    parser.add_argument("--save-path", type=str, default="best_unet_colorization.pt")
    parser.add_argument("--num-samples", type=int, default=6, help="samples to save per epoch")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Print selected device so it's clear in logs which hardware is being used.
    print(f"Using device: {DEVICE}")

    # Prepare dataset transforms and loaders (behavior preserved)
    transform_color = transforms.ToTensor()
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_ds = CIFAR10Colorization(root=args.data_dir, train=True, transform_color=transform_color, transform_gray=transform_gray, download=True)
    val_ds = CIFAR10Colorization(root=args.data_dir, train=False, transform_color=transform_color, transform_gray=transform_gray, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Initialize model, loss, and optimizer (same defaults as before)
    model = UNet(in_channels=1, out_channels=3).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_ssim = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_ssim, val_psnr = validate_epoch(model, val_loader, criterion, DEVICE)

        # Save sample grid from the first validation batch
        model.eval()
        with torch.no_grad():
            for gray_sample, color_sample in val_loader:
                gray_sample = gray_sample.to(DEVICE)
                color_sample = color_sample.to(DEVICE)
                pred_sample = model(gray_sample)
                save_sample_grid(gray_sample.cpu(), pred_sample.cpu(), color_sample.cpu(), epoch, args.out_dir, n_samples=args.num_samples)
                break

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SSIM: {val_ssim:.4f} | Val PSNR: {val_psnr:.4f}")

        # Checkpoint best model by SSIM (unchanged behavior)
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model (SSIM={best_ssim:.4f}) to {args.save_path}")

    print("Training complete.")
    print(f"Best validation SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()
