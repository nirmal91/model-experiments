"""
Step 1: Convolutional Autoencoder

Goal: learn a compressed latent representation of our shape images by
training a network to reconstruct its own input through a narrow bottleneck.
This is the simplest building block behind almost everything else in the
vision/world-model space (VAEs, VQ-VAE tokenizers, the encoder in Ha &
Schmidhuber's "World Models", etc).

encoder: image (3x64x64) -> latent vector (LATENT_DIM)
decoder: latent vector -> reconstructed image (3x64x64)
loss: mean squared error between input and reconstruction

Run: python3 src/autoencoder.py
Outputs: outputs/autoencoder_reconstructions.png, outputs/autoencoder_loss.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

LATENT_DIM = 16
EPOCHS = 1500
LR = 1e-3
SEED = 0


def load_images():
    paths = sorted(DATA_DIR.glob("*.png"))
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        imgs.append(arr)
    return torch.stack(imgs), paths


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 64x64 -> 32x32 -> 16x16 -> 8x8, channels 3 -> 16 -> 32 -> 64
        # BatchNorm after every conv keeps activations from exploding/saturating
        # during full-batch training -- without it this network collapses to
        # outputting the same constant image for every input (see git history /
        # README for the debugging story).
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),   # 32x32
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),  # 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),  # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8), nn.BatchNorm1d(64 * 8 * 8), nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),  # 16x16
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),  # 32x32
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # 64x64
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(self.decoder_fc(z))
        return recon, z


def main():
    torch.manual_seed(SEED)
    images, paths = load_images()
    print(f"loaded {len(images)} images, shape={tuple(images.shape)}")

    model = ConvAutoencoder(LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for epoch in range(EPOCHS):
        opt.zero_grad()
        recon, z = model(images)
        loss = F.mse_loss(recon, images)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  mse={loss.item():.5f}")

    # loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("reconstruction MSE")
    plt.title("Autoencoder training loss")
    plt.savefig(OUT_DIR / "autoencoder_loss.png", dpi=120)
    plt.close()

    # a grid of original vs reconstruction for a handful of images
    model.eval()
    with torch.no_grad():
        recon, z = model(images)
    n_show = 8
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for i in range(n_show):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_title("original", loc="left")
    axes[1, 0].set_title("reconstructed", loc="left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "autoencoder_reconstructions.png", dpi=120)
    plt.close()

    print(f"latent dim = {LATENT_DIM}, final mse = {losses[-1]:.5f}")
    print(f"wrote {OUT_DIR / 'autoencoder_loss.png'}")
    print(f"wrote {OUT_DIR / 'autoencoder_reconstructions.png'}")


if __name__ == "__main__":
    main()
