# Inspired by https://github.com/SonyCSLParis/music2latent/

from typing import Literal

import numpy as np
import torch
import torch.nn as nn


# Preconditioning
def get_c(sigma, sigma_min=0.002, sigma_data=0.5):
    sigma_correct = sigma_min
    c_skip = (sigma_data**2.0) / (((sigma - sigma_correct) ** 2.0) + (sigma_data**2.0))
    c_out = (sigma_data * (sigma - sigma_correct)) / (
        ((sigma_data**2.0) + (sigma**2.0)) ** 0.5
    )
    c_in = 1.0 / (((sigma**2.0) + (sigma_data**2.0)) ** 0.5)
    return (
        c_skip.reshape(-1, 1, 1, 1),
        c_out.reshape(-1, 1, 1, 1),
        c_in.reshape(-1, 1, 1, 1),
    )


class Diffusion(nn.Module):
    def __init__(
        self,
        schedule: Literal["exponential", "constant"] = "exponential",
        start_exp: float = 1.0,
        end_exp: float = 3.0,
        base_step: float = 0.1,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        p_mean: float = -1.1,
        p_std: float = 2.0,
        sigma_data: float = 0.5,
        use_lognormal: bool = True,
        total_iters: int | None = None,
        enable_grad_denoise: bool = False,
    ):
        super().__init__()
        self.schedule = schedule
        self.start_exp = start_exp
        self.end_exp = end_exp
        self.base_step = base_step
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.use_lognormal = use_lognormal
        self.sigma_data = sigma_data
        self.total_iters = total_iters
        self.enable_grad_denoise = enable_grad_denoise
        self.validate()
        self.inds_schedule = None
        self.sigmas_schedule = None

    def flush_noise_schedule(self):
        """Remove any existing cached schedule."""
        if hasattr(self, "inds_schedule"):
            del self.inds_schedule
        if hasattr(self, "sigmas_schedule"):
            del self.sigmas_schedule

    def validate(self):
        """Validate parameters after initialization"""
        if self.schedule not in ["exponential", "constant"]:
            raise ValueError("schedule must be either 'exponential' or 'constant'")

        if self.sigma_min <= 0 or self.sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must be positive")

        if self.rho <= 0:
            raise ValueError("rho must be positive")

    def get_step_schedule(self, iteration, total_iters=None):
        """
        Get step size based on the current iteration.

        Args:
            iteration: Current training iteration
            total_iters: Total number of iterations (needed for some schedules)

        Returns:
            step: Calculated step size
        """
        if total_iters is None:
            total_iters = self.total_iters

        if self.schedule == "exponential":
            # Exponential interpolation between start_exp and end_exp
            if total_iters is not None:
                # Linear interpolation of exponent based on iteration
                exponent = self.start_exp + (self.end_exp - self.start_exp) * (
                    iteration / total_iters
                )
            else:
                exponent = self.end_exp
            step = self.base_step**exponent
        elif self.schedule == "constant":
            step = self.base_step**self.end_exp
        else:
            raise ValueError(
                f"Unknown schedule: {self.schedule}, must be 'exponential' or 'constant'"
            )

        return step

    def get_step_continuous(self, inds, step):
        """
        Get stepped index for continuous sampling - decreases indices by step amount

        Args:
            inds: Continuous indices in [0, 1]
            step: Step size from get_step_schedule

        Returns:
            Indices with given step subtracted, clamped to minimum of 0
        """
        steps = torch.ones_like(inds) * step
        return (inds - steps).clamp(min=0.0)

    def get_sigma(self, i, k):
        """
        Get noise level sigma for discrete index i in [1, k]

        Args:
            i: Discrete index in [1, k]
            k: Total number of discretization steps

        Returns:
            sigma: Corresponding noise level
        """
        return (
            self.sigma_min ** (1.0 / self.rho)
            + ((i - 1) / (k - 1))
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def get_sigma_continuous(self, i):
        """
        Get noise level sigma for a continuous index i in [0, 1]
        Follows parameterization in https://openreview.net/pdf?id=FmqFfMTNnv

        Args:
            i: Continuous index in [0, 1]

        Returns:
            sigma: Corresponding noise level
        """
        return (
            self.sigma_min ** (1.0 / self.rho)
            + i
            * (self.sigma_max ** (1.0 / self.rho) - self.sigma_min ** (1.0 / self.rho))
        ) ** self.rho

    def get_sigma_step_continuous(self, sigma_i, step):
        """
        Get noise level sigma_{i-step} where i is a continuous index in (0, 1]

        Args:
            sigma_i: Current sigma value
            step: Step to be taken towards lower sigma

        Returns:
            sigma_{i-step}: Noise level corresponding to i-step
        """
        return (
            (
                sigma_i ** (1.0 / self.rho)
                - step
                * (
                    self.sigma_max ** (1.0 / self.rho)
                    - self.sigma_min ** (1.0 / self.rho)
                )
            )
            ** self.rho
        ).clamp(min=self.sigma_min)

    def get_loss_weight(self, sigma_plus_one, sigma):
        return 1.0 / (sigma_plus_one - sigma)

    def get_sampling_weights(self, k, device="cuda"):
        """
        Get sampling weights for sigma values
        Samples from lognormal distribution

        Args:
            k: number of discretization steps
            device: Target device for tensors

        Returns:
            weights: Sampling weights
        """
        sigma = self.get_sigma(
            torch.linspace(1, k - 1, k - 1, dtype=torch.int32, device=device), k
        )
        return self.gaussian_pdf(torch.log(sigma))

    def add_noise(self, x, noise, sigma):
        """
        Add Gaussian noise to input x based on given noise and sigma

        Args:
            x: Input tensor
            noise: Tensor containing Gaussian noise
            sigma: Noise level

        Returns:
            x_noisy: x with noise added
        """
        return x + sigma.reshape(-1, 1, 1, 1) * noise

    def reverse_step(self, x, noise, sigma):
        """
        Reverse the probability flow ODE by one step

        Args:
            x: input
            noise: Gaussian noise
            sigma: noise level

        Returns:
            x: x after reversing ODE by one step
        """
        return x + ((sigma**2 - self.sigma_min**2) ** 0.5) * noise

    def gaussian_pdf(self, x):
        """
        Gaussian probability density function, used to sample noise levels with lognormal distribution

        Args:
            x: input

        Returns:
            pdf: probability density at x
        """
        return (1.0 / (self.p_std * (2.0 * np.pi) ** 0.5)) * torch.exp(
            -0.5 * ((x - self.p_mean) / self.p_std) ** 2.0
        )

    def reverse_diffusion(self, denoiser, initial_noise, diffusion_steps, latents=None):
        next_noisy_samples = initial_noise
        # Reverse process step-by-step
        for k in range(diffusion_steps):
            # Get sigma values
            sigma = self.get_sigma(diffusion_steps + 1 - k, diffusion_steps + 1)
            next_sigma = self.get_sigma(diffusion_steps - k, diffusion_steps + 1)

            # Denoise
            noisy_samples = next_noisy_samples
            pred_noises, pred_samples = self.denoise(
                denoiser, noisy_samples, sigma, latents
            )

            # Step to next (lower) noise level
            next_noisy_samples = self.reverse_step(
                pred_samples, pred_noises, next_sigma
            )

        return pred_samples if self.enable_grad_denoise else pred_samples.detach()

    def denoise(self, denoiser, noisy_samples, sigma, latents=None):
        # Denoise samples
        with torch.inference_mode(mode=not self.enable_grad_denoise):
            if latents is not None:
                pred_samples = denoiser.forward_generator(latents, noisy_samples, sigma)
            else:
                raise ValueError("latents must be provided for denoising")

        # Sample noise
        pred_noises = torch.randn_like(pred_samples)
        return pred_noises, pred_samples

    def precompute_noise_schedule(
        self, epoch_steps: int, batch_size: int, device=torch.device("cpu")
    ):
        """
        Build two buffers of shape (epoch_steps, batch_size):
         - inds_schedule: raw uniform/lognormal indices in [0,1)
         - sigmas_schedule: corresponding sigma = get_sigma_continuous(inds)
        """
        total = epoch_steps * batch_size

        # 1) draw raw indices
        if self.use_lognormal:
            N = 10_000
            w = self.get_sampling_weights(N, device=device)  # (N,)
            inds_int = torch.multinomial(w, total, replacement=True)  # (total,)
            jitter = torch.rand(total, device=device)  # (total,)
            raw_inds = (inds_int.float() + jitter) / float(N - 1)  # (total,)
        else:
            raw_inds = torch.rand(total, device=device)  # (total,)

        raw_inds = raw_inds.view(epoch_steps, batch_size)  # (epoch_steps, batch_size)
        raw_sigmas = self.get_sigma_continuous(raw_inds)  # same shape

        # 3) register buffers (non-persistent so not saved in ckpt)
        self.register_buffer("inds_schedule", raw_inds, persistent=False)
        self.register_buffer("sigmas_schedule", raw_sigmas, persistent=False)

    def get_raw_inds_and_sigmas(self, batch_idx: int):
        """
        Retrieve the precomputed row for batch index.
        Returns two tensors of shape (batch_size,).
        """
        if self.inds_schedule is None or self.sigmas_schedule is None:
            raise RuntimeError(
                "Noise schedule not precomputed. Call precompute_noise_schedule first."
            )
        return self.inds_schedule[batch_idx], self.sigmas_schedule[batch_idx]
