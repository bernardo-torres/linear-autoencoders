# Linear Consistency Autoencoders

This repository contains the official code and pretrained models for the paper **"Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization"**.

[**Paper**](https://arxiv.org/abs/2510.23530) | [**Demo Page**](https://bernardo-torres.github.io/projects/linear-cae/)

## About The Project

This work introduces **Linear Consistency Autoencoders (Lin-CAE)**, a training methodology to induce linearity in audio autoencoders. By using data augmentation, we enforce **homogeneity** (equivariance to scalar gain) and **additivity** (preservation of addition) in the latent space without altering the model's architecture or loss function.

This creates a structured latent space where simple algebraic operations correspond directly to intuitive audio manipulations like mixing and volume scaling.

<p align="center">
  <img src="https://bernardo-torres.github.io/documents/images/linear-cae/overview.png" width="400"/>
</p>

This repository currently provides inference code for our pretrained models and the code to reproduce the demos on our project page. Training code will be made available soon.

## Usage

You can install the `linear-cae` package using pip (or whatever package manager you prefer, we recommend using [Poetry](https://python-poetry.org/) for development):

```bash
pip install linear-cae
poetry add linear-cae  # if using Poetry
```

### Loading a Pretrained Model

The `Autoencoder` class provides a `from_pretrained` method to load models from the Hugging Face Hub.

```python
from linear_cae import Autoencoder
import torch

model_id = "lin-cae"  # or "m2l", "lin-cae-2"
model = Autoencoder.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)
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
```

Check out the `demo.ipynb` notebook used to create the demos on our [project page](https://bernardo-torres.github.io/projects/linear-cae/) for more examples of latent space manipulations (requires the `musdb` dataset package).

### Handling Long Audio Files

To handle long audio files without running out of memory, you can specify `max_chunk_size` when loading the model. Audio longer than this size will be automatically processed in overlapping chunks.

```python
# Set max_chunk_size to 10 seconds of audio at 44.1 kHz
model = Autoencoder.from_pretrained("lin-cae", max_chunk_size=44100 * 10, overlap_percentage=0.25)
model.to(device)

# Encode a long audio file (e.g., 30 seconds)
long_audio_tensor = torch.randn(1, 44100 * 30).to(device)
z_chunked = model.encode(long_audio_tensor)

# The output z_chunked will be a 4D tensor: [batch_size, num_chunks, channels, latent_dim]
print(z_chunked.shape)

# Decoding the chunked latent requires the original audio length for proper reconstruction
reconstructed_long_audio = model.decode(z_chunked, full_length=long_audio_tensor.shape[-1])
```

The model uses an overlap-add mechanism with a crossfade to stitch the decoded chunks back together. You can control the amount of overlap with the `overlap_percentage` argument during model initialization. We recommend some overlap to avoid artifacts at chunk boundaries.

## Algorithm

While the full training code for CAEs will be released soon, here is a general pseudo-algorithm for how to adapt any autoencoder training loop to induce linearity using our proposed method. The core idea is to use data augmentation to implicitly teach the model the properties of linearity.

```python
# Pseudo-algorithm for training a linear autoencoder

for each batch of audio data x:
  original_batch_size = x.shape[0]

  # 1. Create artificial mixtures
  x_roll = torch.roll(x, shifts=1, dims=0) # Circularly shift the batch
  x_mixed = x + x_roll
  x_augmented = torch.cat([x, x_mixed], dim=0) # Batch now contains original and mixed audio

  # 2. Encode all audio to get latents
  z = encoder(x_augmented)

  # For the mixed portion of the batch, create the "additive" latent
  # by summing the latents of the original unmixed sources.
  z_original = z[:original_batch_size]
  z_roll = torch.roll(z_original, shifts=1, dims=0)
  z_add = z_original + z_roll

  # Replace the encoded latents of the mixed audio with the sum of source latents
  z_final = torch.cat([z_original, z_add], dim=0)

  # 3. Apply random gains to enforce Homogeneity
  gains = sample_random_gains(x_augmented.shape[0])
  z_scaled = z_final * gains[:, None, None]
  x_scaled = x_augmented * gains[:, None]

  # 4. Standard autoencoder training step
  # The decoder receives the scaled latent and must reconstruct the scaled audio
  x_reconstructed = decoder(z_scaled)

  loss = reconstruction_loss(x_reconstructed, x_scaled)

  # Update model weights
  optimizer.step()
```

For Consistency Autoencoders (CAEs), as used in the paper, `x_scaled` is first corrupted with noise, and the decoder (a denoising model) is conditioned on `z_scaled` to reconstruct the clean `x_scaled`.

## Citation

If you use our work in your research, please cite our paper:

```bibtex
@misc{torres2025learninglinearityaudioconsistency,
      title={Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization},
      author={Bernardo Torres and Manuel Moussallam and Gabriel Meseguer-Brocal},
      year={2025},
      eprint={2510.23530},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.23530},
}
```
