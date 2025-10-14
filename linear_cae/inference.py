import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from linear_cae.components.diffusion import Diffusion
from linear_cae.components.frontends import ScaledComplexSTFT
from linear_cae.components.generator import UNet
from linear_cae.utils import _NoOpEMA

HF_REPO = "BernardoTorres/linear_consistency_autoencoders"


class EncoderInferenceModel(torch.nn.Module):
    """Encoder inference model, it can accept generator arguments if the encoder has a loose **kwargs** interface."""

    def __init__(self, frontend, generator, max_batch_size=1):
        super().__init__()
        self.frontend = frontend
        # they can be either models (when using hydra or just kwargs)
        # if they are kwargs, we need to instantiate them
        if isinstance(self.frontend, dict):
            self.frontend = ScaledComplexSTFT(**self.frontend)
        if isinstance(generator, dict):
            self.encoder = UNet(**generator).encoder
        else:
            self.encoder = generator.encoder
        # if isinstance(self.encoder, dict):
        #     # For legacy we have to replace layers_list with layers_list_encoder keys
        #     self.encoder["layers_list"] = self.encoder.pop("layers_list_encoder")
        #     self.encoder["attention_list"] = self.encoder["attention_list_encoder"]
        #     self.encoder = Encoder(**self.encoder)
        self.max_batch_size = max_batch_size

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, extract_features=False):
        # x = self.frontend.to_representation(x)
        x = self.frontend.to_representation(x)
        # x = self.encoder(x, extract_features=extract_features)
        if x.shape[0] <= self.max_batch_size:
            # Batch size is within the limit, process as a single batch
            return self.encoder(x, extract_features=extract_features)
        else:
            # Batch size exceeds the limit, split into chunks and process sequentially
            repr_chunks = torch.split(x, self.max_batch_size, dim=0)
            latent_chunks = []
            for chunk in repr_chunks:
                latent_chunk = self.encoder(chunk, extract_features=extract_features)
                latent_chunks.append(latent_chunk)
            return torch.cat(latent_chunks, dim=0)

    def encode(self, x, *args, **kwargs):
        return self(x, *args, **kwargs)


class AutoencoderInferenceModel(torch.nn.Module):
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
        model_id = model_id + "_weights"
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{model_id}/autoencoder_inference_model_{ckpt_type}.pth"
        )
        frontend_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{model_id}/frontend_kwargs_{ckpt_type}.yaml"
        )
        generator_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{model_id}/generator_kwargs_{ckpt_type}.yaml"
        )
        diffusion_args_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{model_id}/diffusion_kwargs_{ckpt_type}.yaml"
        )

        with open(frontend_args_path) as f:
            frontend_args = yaml.safe_load(f)
        with open(generator_args_path) as f:
            generator_args = yaml.safe_load(f)
        with open(diffusion_args_path) as f:
            diffusion_args = yaml.safe_load(f)

        model = cls(frontend_args, generator_args, diffusion_args, **kwargs)

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

        if self.encoder is None:
            self._set_encoder(self.generator.encoder)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision):
            x = self.frontend.to_representation(x)

            if max_batch_size is None:
                max_batch_size = self.max_batch_size
            if x.shape[0] <= max_batch_size:
                # Batch size is within the limit, process as a single batch
                return self.encoder(x, extract_features=extract_features)
            else:
                # Batch size exceeds the limit, split into chunks and process sequentially
                repr_chunks = torch.split(x, self.max_batch_size, dim=0)
                latent_chunks = []
                for chunk in repr_chunks:
                    latent_chunk = self.encoder(chunk, extract_features=extract_features)
                    latent_chunks.append(latent_chunk)
                return torch.cat(latent_chunks, dim=0)

    def decode(self, z, *args, **kwargs):
        """
        Decode the latent representation z back to waveform.
        This method is used for inference and can be called directly.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision):
            return self.generate(latents=z, *args, **kwargs)

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


def load_checkpoint(
    ckpt_path=None,
    device="cuda:0",
    log_dir="logs",
    extract_features=False,
    ckpt_type="last",
    model_type="autoencoder",
    mixed_precision=False,
    diffusion_steps=1,
    max_batch_size=1,
):
    # Load the checkpoint from the specified path
    # e g ckpt_path = "/data/nfs/analysis/interns/btorres/Projects/music2latent/logs/9ace1950/9ace1950_2025-05-16_16-46"
    if ckpt_path is None:
        # ckpt_path = "/data/nfs/analysis/interns/btorres/Projects/music2latent/logs/9ace1950/9ace1950_2025-05-16_16-46" # for debugging onlu
        raise ValueError(f"Please provide a valid checkpoint path, got {ckpt_path}")
    if ckpt_path == "marco":
        from linear_cae.baselines import Music2LatentPublic

        model = Music2LatentPublic(
            device=device, extract_features=extract_features, max_batch_size=max_batch_size
        )
        run_name = "marco"
    elif ckpt_path == "stable-audio-vae":
        from linear_cae.baselines import StableAudioVAE

        model = StableAudioVAE(device=device)
        run_name = "stable-audio-vae"
    else:
        # local_path = "/data/nfs/analysis/interns/btorres/Projects/music2latent"
        # if local_path in sys.path:
        #     sys.path.remove(local_path)  # Remove it first to avoid duplicates
        # sys.path.insert(0, local_path)

        if not os.path.exists(ckpt_path):
            # from linear_cae.utils.hydra_utils import find_ckpt_from_hash
            hash_str = ckpt_path
            ckpt_path = find_ckpt_from_hash(log_dir, hash_str, type=ckpt_type)
            # ckpt path will be log_dir/../*ckpt_path*/checkpoints/*type*.ckpt
            # We want the base folder so we need to go up two levels
            ckpt_path = os.path.dirname(os.path.dirname(ckpt_path))
            print(f"Checkpoint path set to {ckpt_path}")

        checkpoint_path = f"{ckpt_path}/weights/autoencoder_inference_model_{ckpt_type}.pth"
        frontend_args_path = f"{ckpt_path}/weights/frontend_kwargs_{ckpt_type}.yaml"
        generator_args_path = f"{ckpt_path}/weights/generator_kwargs_{ckpt_type}.yaml"
        diffusion_args_path = f"{ckpt_path}/weights/diffusion_kwargs_{ckpt_type}.yaml"

        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"{ckpt_path}/weights/autoencoder_inference_model.pth"
            frontend_args_path = f"{ckpt_path}/weights/frontend_kwargs.yaml"
            generator_args_path = f"{ckpt_path}/weights/generator_kwargs.yaml"
            diffusion_args_path = f"{ckpt_path}/weights/diffusion_kwargs.yaml"

        # Let's get the run name
        run_name = os.path.basename(ckpt_path)

        # Load the frontend, generator, and diffusion arguments
        frontend_args = load_yaml(frontend_args_path)
        generator_args = load_yaml(generator_args_path)
        diffusion_args = load_yaml(diffusion_args_path)

        # if model_type == "autoencoder":
        model = AutoencoderInferenceModel(
            frontend=frontend_args,
            generator=generator_args,
            diffusion=diffusion_args,
            diffusion_steps=diffusion_steps,
            max_batch_size=max_batch_size,
            mixed_precision=mixed_precision,
            name=f"{hash_str}/{ckpt_type}",
        )

        # else:
        #     model = EncoderInferenceModel(
        #         frontend=frontend_args,
        #         generator=generator_args,
        #         # diffusion=diffusion_args,
        #     )

        # Load the model state dict
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # if "state_dict" in checkpoint:
        # new_state = remap_state_dict(checkpoint, model.state_dict())

        model.load_state_dict(
            checkpoint,
            strict=True,
        )

        if model_type == "encoder":
            # Let's just delete everything that is not the encoder from model.generator to
            # save memory
            model._set_encoder(model.generator.encoder)
            encoder_bcp = model.encoder
            # let's delete model.generator
            del model.generator
            del model.diffusion
            model.encoder = encoder_bcp
            model.decode = lambda *args, **kwargs: (_ for _ in ()).throw(
                NotImplementedError("Decode is not available in encoder-only mode")
            )
            model.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
                NotImplementedError("Generate is not available in encoder-only mode")
            )

        # TODO - freeze model
        for name, param in model.named_parameters():
            param.requires_grad = False

        model.eval()

    return model, run_name
