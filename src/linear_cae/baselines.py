import torch


class StableAudioVAE(torch.nn.Module):
    """
    Provides a compatible interface for the Stable Audio Autoencoder.
    """

    def __init__(self, device="cuda", max_batch_size=torch.inf, mixed_precision=False):
        super().__init__()
        from stable_audio_tools import get_pretrained_model

        self.max_batch_size = max_batch_size
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.model = model.pretransform
        self.mixed_precision = mixed_precision
        self.name = "stable-audio-vae"
        self.model.to(device)
        self.model.eval()
        if mixed_precision:
            self.model.half()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def sample_rate(self):
        return 44100

    def encode(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        if extract_features:
            raise ValueError("Stable Audio AE does not support feature extraction.")
        # The encoder expects input in the range [-1, 1]
        # We need batch, 2, time
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            # repeat channel
            x = x.repeat(1, 2, 1)
        if x.dim() == 2:  # we assume batch, time
            x = x.unsqueeze(1)
            # repeat channel
            x = x.repeat(1, 2, 1)

        z = self.model.encode(x)
        return z  # B, C, T

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # The decoder outputs in the range [-1, 1]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision):
            if z.shape[0] > self.max_batch_size:
                # Process in chunks
                generated_chunks = []
                latents_chunks = torch.split(z, self.max_batch_size, dim=0)
                for chunk in latents_chunks:
                    generated_chunk = self.decode(chunk)
                    generated_chunks.append(generated_chunk)
                return torch.cat(generated_chunks, dim=0)
            return self.model.decode(z)[:, 0, :]  # B, 2, T
