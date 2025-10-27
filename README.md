# Inducing Linearity in Audio Autoencoders via Implicit Regularization

[**Paper**](https://arxiv.org/abs/your-arxiv-link-here) | [**Demo Page**](https://bernardo-torres.github.io/projects/linear-cae/)

This repository contains the official code and pretrained models for the paper **"Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization"**

## About The Project

This work introduces **Linear Consistency Autoencoders (Lin-CAE)**, a straightforward training methodology to induce linearity in high-compression audio autoencoders. By using data augmentation, we enforce **homogeneity** (equivariance to scalar gain) and **additivity** (preservation of addition) in the latent space without altering the model's architecture or loss function.

This creates a structured latent space where simple algebraic operations correspond directly to intuitive audio manipulations like mixing and volume scaling.

<p align="center">
  <img src="https://bernardo-torres.github.io/documents/images/linear-cae/overview.png" width="400"/>
</p>

This repository currently provides inference code for our pretrained models and the code to reproduce the demos on our project page. **Training code will be made available soon.**

## Getting Started

To get started, install the `linear-cae` package using pip:

```bash
pip install linear-cae
```

or if you use poetry:

```bash
poetry add linear-cae
```

## Usage

You can easily load our pretrained models and use them for audio encoding and decoding.

### Loading a Pretrained Model

The `Autoencoder` class provides a `from_pretrained` method to load models from the Hugging Face Hub.

```python
from linear_cae import Autoencoder
import torch

# Load the Lin-CAE model
model_id = "lin-cae"  # or "m2l", "lin-cae-2"
model = Autoencoder.from_pretrained(model_id)

# Move the model to your desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Available `model_id`s are:

- `"m2l"`: The baseline Music2Latent model.
- `"lin-cae"`: Our proposed Linear Consistency Autoencoder.
- `"lin-cae-2"`: A second version of our Lin-CAE trained without gain annealing described in the paper.

### Encoding and Decoding Audio

Once the model is loaded, you can encode audio into a latent representation and decode it back to audio.

```python
# `audio_tensor` is a torch.Tensor of shape [batch_size, num_samples] representing 44.1 kHz audio
audio_tensor = torch.randn(1, 44100 * 2).to(device)

z = model.encode(audio_tensor)
reconstructed_audio = model.decode(z, full_length=audio_tensor.shape[-1])

scaled_z = z * 0.5
scaled_audio = model.decode(scaled_z, full_length=audio_tensor.shape[-1])
...
```

Check out the demo.ipynb notebook used to create the demos in our [project page](https://bernardo-torres.github.io/projects/linear-cae/) with latent space manipulations (requires the musdb dataset package).

"talk about max batch size and max audio length here with overlap add here"

## Algorithm

While the full training code will be released soon for CAEs, here is a general pseudo-algorithm for how to adapt any autoencoder training loop to induce linearity using our proposed method.

The core idea is to use data augmentation to implicitly teach the model the properties of linearity.

```
# Pseudo-algorithm for training a linear autoencoder

for each batch of audio data x:
  original_batch_size = x.shape[0]
  # 1. Create artificial mixtures for additivity
  x_roll = roll(x)
  x_mixed = x + x_roll
  x = torch.cat([x, x_mixed], dim=0) # Both original and mixed data

  # 2. Encode all audio to get latents
  z = encoder(z)

  # Mix the latents corresponding to the mixed audio
  z_roll = roll(z[:original_batch_size])
  z_mixed = z[:original_batch_size] + z_roll
  new_z = torch.cat([z[:original_batch_size], z_mixed], dim=0)

  # 3. Apply random gains for homogeneity
  gains = sample_random_gains()
  z_scaled = new_z * gains
  x_scaled = x * gains

  # 4. Standard autoencoder training step
  # The decoder receives the scaled latent and must reconstruct the scaled audio
  X_reconstructed = decoder(z_scaled)

  loss = reconstruction_loss(X_reconstructed, X_scaled)
```

For diffusion/Consistency Autoencoders, the scaled latents are used as conditioning for the denoising of a currupted version of the scaled audio (see our paper for details).

## Citation

If you use our work in your research, please cite our paper:

```bibtex


```
