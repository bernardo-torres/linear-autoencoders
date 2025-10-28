import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download

from linear_cae.components.diffusion import Diffusion
from linear_cae.components.frontends import ScaledComplexSTFT
from linear_cae.components.generator import UNet
from linear_cae.utils import _NoOpEMA

HF_REPO = "BernardoTorres/linear_consistency_autoencoders"
id_to_hash = {
    "m2l": "1511f18a",
    "lin-cae": "0a3afbec",
    "lin-cae-2": "2f4c6d21",
}

class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        frontend,
        generator,
        diffusion,
        max_batch_size=1,
        diffusion_steps=1,
        mixed_precision=False,
        enable_grad_denoise=False,
        name="model",
        max_chunk_size: int = None,
        overlap_percentage: float = 0.25,
    ):
        super().__init__()
        self.generator = generator
        self.diffusion = diffusion
        self.frontend = frontend
        if isinstance(self.frontend, dict):
            self.frontend = ScaledComplexSTFT(**self.frontend)
        if isinstance(self.generator, dict):
            self.generator = UNet(**self.generator)
        if isinstance(self.diffusion, dict):
            self.diffusion = Diffusion(**self.diffusion)

        self.diffusion_steps = diffusion_steps
        self.diffusion.enable_grad_denoise = enable_grad_denoise

        self.encoder = None
        self.mixed_precision = mixed_precision
        self.name = name
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size
        self.overlap_percentage = overlap_percentage
        self.ema = (
            _NoOpEMA()
        )  # No-op EMA for compatibility if the model is encoded under the EMA wrapper

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def sample_rate(self):
        if hasattr(self.frontend, "sample_rate"):
            return self.frontend.sample_rate
        else:
            return None

    @classmethod
    def from_pretrained(cls, model_id: str, ckpt_type: str = "last", **kwargs):
        """
        Loads the model and configuration from a Hugging Face Hub repository.
        """
        # Download all necessary files
        model_id = id_to_hash.get(model_id, model_id)
        hf_id = model_id + "_weights"
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{hf_id}/autoencoder_inference_model_{ckpt_type}.pth"
        )
        frontend_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{hf_id}/frontend_kwargs_{ckpt_type}.yaml"
        )
        generator_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{hf_id}/generator_kwargs_{ckpt_type}.yaml"
        )
        diffusion_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{hf_id}/diffusion_kwargs_{ckpt_type}.yaml"
        )

        with open(frontend_args_path) as f:
            frontend_args = yaml.safe_load(f)
        with open(generator_args_path) as f:
            generator_args = yaml.safe_load(f)
        with open(diffusion_args_path) as f:
            diffusion_args = yaml.safe_load(f)

        model = cls(frontend_args, generator_args, diffusion_args, name=model_id, **kwargs)

        # Load the model state dict
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_latent_size(self, audio_length):
        """
        Calculates the temporal size of the latent representation based on the model's architecture.
        """
        frame_length = self.frontend.fac * self.frontend.hop_size
        frame_step = self.frontend.hop_size
        # Calculate the number of frames using the torch.unfold formula
        num_frames = (audio_length - frame_length) // frame_step + 1

        num_temporal_downsamples = self.generator.freq_downsample_list.count(0)
        # Apply the convolution output size formula for each downsampling layer
        temp_size = num_frames
        for _ in range(num_temporal_downsamples):
            temp_size = (temp_size - 1) // 2 + 1

        return temp_size

    def get_decoded_size(
        self,
        latent_length,
    ):
        """
        Calculates the expected audio length from a latent representation.
        """
        # Count the number of temporal upsampling stages, which mirrors the encoder.
        num_temporal_upsamples = self.generator.freq_downsample_list.count(0)
        temp_size = 2**num_temporal_upsamples * latent_length

        # Reverse the framing process using the overlap-add formula
        frame_length = self.frontend.fac * self.frontend.hop_size
        frame_step = self.frontend.hop_size
        return (temp_size - 1) * frame_step + frame_length

    def _set_encoder(self, encoder):
        self.encoder = encoder

    def encode(self, x, extract_features=False, max_batch_size=None):
        """
        Encodes audio x [B, T] into latents z.
        Handles chunking if T > max_chunk_size.
        Returns:
            - [B, C, Z] if not chunked
            - [B, num_chunks, C, Z] if chunked
        """
        if self.encoder is None:
            self._set_encoder(self.generator.encoder)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision):

            # --- CHUNKING LOGIC ---
            if self.max_chunk_size is None or x.shape[-1] <= self.max_chunk_size:
                # No chunking needed
                x_repr = self.frontend.to_representation(x)
                return self._encode_internal(x_repr, extract_features, max_batch_size)
            else:
                batch_size, T = x.shape
                chunk_len = self.max_chunk_size
                overlap_len = int(chunk_len * self.overlap_percentage)
                step = chunk_len - overlap_len

                padding = (step - (T - chunk_len) % step) % step
                padded_audio = torch.nn.functional.pad(x, (0, padding))

                # Create chunks
                chunks = padded_audio.unfold(dimension=-1, size=chunk_len, step=step)
                # chunks shape is [B, num_chunks, chunk_len]
                num_chunks = chunks.shape[1]

                chunks_batched = chunks.reshape(-1, chunk_len) # [B * num_chunks, chunk_len]

                # Encode chunks
                chunks_repr = self.frontend.to_representation(chunks_batched)
                z_batched = self._encode_internal(chunks_repr, extract_features, max_batch_size) # [B * num_chunks, C, Z]

                # Reshape back to [B, num_chunks, C, Z]
                _, C, Z = z_batched.shape
                z = z_batched.view(batch_size, num_chunks, C, Z)
                return z

    def _encode_internal(self, x_repr, extract_features=False, max_batch_size=None):
        """
        Internal encode method that processes spectrograms and handles max_batch_size.
        Assumes x_repr is [B, C, F, T_spec]
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        if x_repr.shape[0] <= max_batch_size:
            # Batch size is within the limit, process as a single batch
            return self.encoder(x_repr, extract_features=extract_features)
        else:
            # Batch size exceeds the limit, split into chunks and process sequentially
            repr_chunks = torch.split(x_repr, max_batch_size, dim=0)
            latent_chunks = []
            for chunk in repr_chunks:
                latent_chunk = self.encoder(chunk, extract_features=extract_features)
                latent_chunks.append(latent_chunk)
            return torch.cat(latent_chunks, dim=0)

    def _decode_internal(self, z, *args, **kwargs):
        """
        Internal decode method that processes latents and calls generate.
        Assumes z is [B, C, Z]
        """
        return self.generate(latents=z, *args, **kwargs)

    def decode(self, z, full_length=None, *args, **kwargs):
        """
        Public decode method.
        Handles OLA reconstruction if latents are 4D.
        Trims audio if full_length is provided.
        """

        # --- NO CHUNKING ---
        if z.ndim == 3:
            # Standard 3D latent [B, C, Z], decode normally
            # User is responsible for passing 'samples' if needed
            return self._decode_internal(z, *args, samples=full_length, **kwargs)

        # --- CHUNKING ---
        elif z.ndim == 4:
            assert full_length is not None, (
            "final audio length must be provided when decoding chunked latents via kwarg full_length\n"
            "Try model.decode(z, full_length=original_audio_length)"
            )

            # 4D latent [B, num_chunks, C, Z], perform OLA
            B, num_chunks, C, Z = z.shape
            z_batched = z.view(-1, C, Z) # [B * num_chunks, C, Z]

            chunk_len = self.max_chunk_size
            if chunk_len is None:
                raise ValueError("Cannot decode 4D chunked audio if max_chunk_size is not set in the model.\n"
                                 "Try setting model.max_chunk_size (in samples) before encoding/decoding.")
            overlap_len = int(chunk_len * self.overlap_percentage)
            step = chunk_len - overlap_len

            # Pass chunk_len to internal decoder via 'samples' kwarg
            kwargs['samples'] = chunk_len
            dec_chunks_batched = self._decode_internal(z_batched, *args, **kwargs)

            return _overlap_add_crossfade(
                dec_chunks_batched,
                batch_size=B,
                num_chunks=num_chunks,
                in_step=step,
                in_chunk_len=chunk_len,
                final_len=full_length
            )

        else:
            raise ValueError(f"Latent tensor z has invalid dimensions: {z.shape}. Expected 3D or 4D.")

    def generate(
        self, diffusion_steps=None, seconds=None, samples=None, latents=None, max_batch_size=None
    ):
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        if diffusion_steps is None:
            diffusion_steps = self.diffusion_steps
        if latents.shape[0] > max_batch_size:
            # Batch size exceeds the limit, split into chunks and process sequentially
            latents_chunks = torch.split(latents, max_batch_size, dim=0)
            generated_chunks = []
            for chunk in latents_chunks:
                generated_chunk = self.generate(
                    diffusion_steps=diffusion_steps,
                    seconds=seconds,
                    samples=samples,
                    latents=chunk,
                    max_batch_size=max_batch_size,
                )

                generated_chunks.append(generated_chunk)
            return torch.cat(generated_chunks, dim=0)

        freq_downsample_list = self.generator.freq_downsample_list
        if seconds is None and samples is None:
            sample_length = 64

        else:
            if seconds is not None and samples is None:
                raise ValueError(
                    "Please provide instead parameter samples as seconds * sample_rate."
                )
            downscaling_factor = 2 ** freq_downsample_list.count(0)
            sample_length = int((samples) // self.generator.hop // downscaling_factor)
        if latents is not None:
            num_samples = latents.shape[0]
            downscaling_factor = 2 ** freq_downsample_list.count(0)
            sample_length = int(latents.shape[-1] * downscaling_factor)
        initial_noise = (
            torch.randn(
                (
                    num_samples,
                    self.generator.data_channels,
                    self.generator.hop * 2,
                    sample_length,
                ),
                device=self.device,
            )
            * self.generator.sigma_max
        )
        generated_images = self.diffusion.reverse_diffusion(
            self.generator, initial_noise, diffusion_steps, latents=latents
        )
        # return to_waveform(generated_images)
        return self.frontend.to_waveform(generated_images)[..., :samples]


def _overlap_add_crossfade(processed_chunks, batch_size, num_chunks, in_step, in_chunk_len, final_len):
    """
    Reconstructs audio from processed chunks using overlap-add.
    Uses a fast vectorized crossfade and F.fold() for overlapping chunks,
    and a simple reshape() for non-overlapping chunks.
    """
    device = processed_chunks.device
    out_chunk_len = processed_chunks.shape[-1]

    # Reshape back to [batch_size, num_chunks, out_chunk_len]
    processed_chunks = processed_chunks.view(batch_size, num_chunks, out_chunk_len)

    # Calculate output step and overlap, maintaining the input ratio
    out_step = out_chunk_len * in_step // in_chunk_len
    out_overlap_len = out_chunk_len - out_step

    # Calculate the full output length
    out_len = out_step * (num_chunks - 1) + out_chunk_len

    if out_overlap_len > 0:
        # Create fade windows
        fade_in = torch.linspace(0., 1., out_overlap_len, device=device)
        fade_out = 1. - fade_in

        # Apply fades in a vectorized way
        # Apply fade-in to all chunks *except the first*
        processed_chunks[:, 1:, :out_overlap_len] *= fade_in[None, None, :]

        # Apply fade-out to all chunks *except the last*
        processed_chunks[:, :-1, -out_overlap_len:] *= fade_out[None, None, :]

        # Reshape for fold: [B, kernel_size, num_chunks]
        processed_chunks_transposed = processed_chunks.transpose(1, 2)

        reconstructed_audio = F.fold(
            processed_chunks_transposed,
            output_size=(out_len, 1),      # (L_out, W_out=1)
            kernel_size=(out_chunk_len, 1), # (kH, kW=1)
            stride=(out_step, 1)            # (dH, dW=1)
        )

        # Squeeze the dummy dimensions
        reconstructed_audio = reconstructed_audio.squeeze(1).squeeze(-1) # [B, out_len]

    else:
        # Sanity check
        if out_len != num_chunks * out_chunk_len:
             warnings.warn(f"OLA: Non-overlapping length mismatch. out_len={out_len}, expected={num_chunks * out_chunk_len}")

        reconstructed_audio = processed_chunks.reshape(batch_size, -1)

    # Trim to the final, original length
    return reconstructed_audio[..., :final_len]


def load_yaml(file_path):
    import yaml

    with open(file_path) as file:
        data = yaml.safe_load(file)
    return data


def find_ckpt_from_hash(log_dir, hash_str, type="last"):
    """
    Recursively search for a checkpoint file in the given directory
    that matches the specified hash.
    The expected filename format is:
    log_dir/<hash>/<hash>_datetime/checkpoints/last.ckpt
    but we can have other variations like:
    log_dir/<hash>/<hash>/checkpoints/last.ckpt
    log_dir/<hash>_datetime/<hash>_datetime/checkpoints/last.ckpt
    so we will match the hash in the first level of the directory structure. then
    if we find multiple matches to last.ckpt, we will return the one with the latest modification time.
    """
    log_dir = Path(log_dir)
    hash_str = str(hash_str)

    # Search for the hash in multiple recursive levels
    matches = list(log_dir.rglob(f"{hash_str}*/checkpoints/*{type}*.ckpt"))
    matches += list(log_dir.rglob(f"{hash_str}*/weights/*{type}*.pth"))
    for path in matches:
        if "latest" in str(path):
            continue
        if path.is_file():
            return str(path)

    return None
