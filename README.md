# Model Experiments: Vision & World Models, from Scratch

A hands-on learning project. Each step is a small, self-contained script you
run and inspect — the goal is to build intuition for how vision/video/world
models actually work, not to build anything production-grade.

No external dataset downloads are used (this sandbox has restricted network
access) — instead `src/generate_data.py` procedurally generates:

- `data/images/` — 64 static images of random shapes (circle/square/triangle),
  colors, and positions on a 64x64 canvas
- `data/sequence/` — 30 frames of a single shape moving in a straight line and
  bouncing off the canvas walls (simple deterministic "physics"), used later
  for the world-model / next-frame-prediction step

Generating our own data means we know the exact ground-truth factors (shape,
color, x, y) behind every pixel, so we can sanity-check whether a model's
learned latent space is actually capturing them.

## Steps

### 1. Convolutional Autoencoder — `src/autoencoder.py` (done)

Encode each image down to a 16-number latent vector, then decode back to a
full image, trained purely to minimize reconstruction error. This is the
foundational building block behind VAEs, VQ-VAE tokenizers, and the encoder
used in classic world models (Ha & Schmidhuber, 2018).

Run: `python3 src/autoencoder.py`
Outputs: `outputs/autoencoder_loss.png`, `outputs/autoencoder_reconstructions.png`

**Debugging notes (the actual learning part):** the first two versions of
this network silently collapsed to outputting the same constant image
regardless of input — a classic autoencoder failure mode:

1. First attempt trained fine on paper (loss went down) but reconstructions
   were uniform gray blobs. Root cause: full-batch Adam with `lr=2e-3` pushed
   the decoder's pre-activations permanently negative, killing all ReLU
   gradients ("dead ReLU"), so the loss plateaued at a suspiciously exact
   constant value.
2. Second attempt added a `Tanh` at the latent bottleneck to bound its scale.
   Loss went down further but reconstructions were still identical for every
   image. Diagnosis (`z.std(dim=0)` across the batch was exactly `0.0`): the
   16-d latent vector had saturated to the same `[+1,-1,+1,-1,...]` pattern
   for every single image — representation collapse. The `Linear(4096, 16)`
   layer's pre-activations exploded early in training and saturated `Tanh`
   in the same direction for all samples before it had a chance to
   differentiate them.
3. Fix: added `BatchNorm` after every conv/linear layer in the encoder and
   decoder. This keeps activation scales controlled throughout training
   regardless of the raw weight magnitudes, which is exactly the failure
   mode BatchNorm was designed to fix. Loss dropped smoothly to `~0.0004`
   and reconstructions correctly recovered shape, color, *and* position from
   just 16 numbers.

Lesson: a low/decreasing loss number is not proof a model is learning
correctly — always look at the actual outputs. "The loss is going down" and
"the model is a no-op that outputs the dataset mean" can look identical in a
loss curve.

### 2. VAE (next)

Same architecture, but the encoder outputs a mean + variance and we sample
from it, with a KL-divergence term pulling the latent distribution toward a
standard normal. Lets us *generate* new shape images by sampling random
latents, not just reconstruct existing ones.

### 3. World model: VAE + RNN over `data/sequence/` (next)

Encode each frame of the moving-shape sequence to a latent vector, then train
a small RNN to predict the *next* latent from the current one. This is the
Ha & Schmidhuber "World Models" recipe in miniature: perception (VAE) +
dynamics prediction (RNN) = a model that can "imagine" future frames without
ever seeing raw pixels during rollout.

### 4. Diffusion model (stretch goal)

A minimal DDPM trained to denoise our shape images, to see the technique
behind Stable Diffusion / modern video generation models.

## Setup

```
pip3 install torch torchvision numpy pillow matplotlib
python3 src/generate_data.py   # regenerate the toy dataset
python3 src/autoencoder.py     # step 1
```
