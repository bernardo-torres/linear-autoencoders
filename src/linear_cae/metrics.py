import math

import auraloss
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import Metric

from linear_cae.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def svd_method(embeddings: torch.Tensor) -> None:
    z = torch.nn.functional.normalize(embeddings, dim=1)
    # calculate covariance
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, singular_values, _ = np.linalg.svd(c)
    # taking natural log of singular values
    singular_values_log = np.log(singular_values)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(len(singular_values)), singular_values_log, marker="o", linestyle="-")
    # set other plot params like title, labels, figure size, etc.
    plt.title("SVD Log-Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Log-Singular Value")
    plt.grid()
    plt.tight_layout()

    return fig
    # plt.show()


def get_erank(embeddings: torch.Tensor) -> float:
    """https://github.com/tanatosuu/directspec/blob/main/directspec_mf.py"""
    # We assume embeddings is batch, channel or batch, time, channel
    # Let's flatten the embeddings to 2D if they are not already
    if embeddings.dim() > 2:
        embeddings = embeddings.reshape(-1, embeddings.size(-1))  # moving time dimension to batch

    embeddings_np = embeddings.detach().cpu().numpy()
    values = np.linalg.svd(embeddings_np, full_matrices=False, compute_uv=False)
    # normalize singular values
    values_normalized = values / np.sum(values)
    # calculate entropy
    entropy = -(values_normalized * np.nan_to_num(np.log(values_normalized), neginf=0)).sum()
    # calculate erank and convert it to scalar
    erank = np.exp(entropy).item()
    return erank


class ERankMetric(Metric):
    """Metric to compute the effective rank of embeddings"""

    def __init__(self, device="cuda"):
        super().__init__()
        self.add_state("sum_erank", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.to(device)

    def update(self, embeddings):
        """Update the metric with new embeddings"""
        try:
            erank = get_erank(embeddings)

            self.sum_erank += erank
            self.total_samples += (
                1  # Here we add 1 because the metric takes already the average of the batch
            )
        except Exception as e:
            print(f"Error computing effective rank: {e}")
            log.error(f"Error computing effective rank for embeddings of shape {embeddings.shape}")
            # Let's check if there is any NaN or Inf in the embeddings
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                log.error(
                    f"Embeddings contain NaN or Inf values: {embeddings[torch.isnan(embeddings) | torch.isinf(embeddings)]}"
                )
            self.sum_erank += 0.0  # In case of error, we add 0 to the sum
            self.total_samples += 1  # Still count this sample to avoid division by zero
            # return

    def compute(self):
        """Compute the final effective rank value"""
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.sum_erank.device)
        return self.sum_erank / self.total_samples


def relative_l2_error(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculates the scale-invariant relative L2 error: ||preds - targets|| / ||targets||.
    Assumes inputs are batch-first, e.g., (B, C, F) or (B, T).
    """
    # Calculate the L2 norm over all dimensions except the batch dimension
    dims_to_reduce = tuple(range(1, preds.dim()))
    error_norm = torch.linalg.vector_norm(preds - targets, dim=dims_to_reduce)
    target_norm = torch.linalg.vector_norm(targets, dim=dims_to_reduce)
    return error_norm / (target_norm + eps)


class AudioToAudioMetric(Metric):
    """Base class for audio-to-audio metrics using auraloss under the hood"""

    def __init__(
        self,
        metric_name,
        loss_fn_constructor,
        negate_result=False,
        device="cuda",
        reduction="mean",
        **loss_args,
    ):
        super().__init__()
        self.metric_name = metric_name
        self.loss_fn = loss_fn_constructor(reduction="none", **loss_args)
        self.negate_result = negate_result  # For metrics like SDR where we want to negate the loss

        self.reduction = reduction

        if self.reduction == "mean":
            # State for mean calculation
            self.add_state(f"sum_{metric_name}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        elif self.reduction == "median":
            # State for median calculation
            self.add_state("values", default=[], dist_reduce_fx="cat")
        else:
            raise ValueError(f"Reduction '{self.reduction}' not supported. Use 'mean' or 'median'.")

        self.to(device)

    def update(self, preds, target):
        """Common update logic for all audio metrics"""
        batch_size = preds.shape[0]

        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        with torch.no_grad():
            losses = self.loss_fn(preds, target)

            if self.negate_result:
                losses = -losses

            if losses.dim() > 1:
                losses = self.reduce_losses(
                    losses
                )  # Reduce over time/frequency, not batch dimension

            # Check if losses are NaN or Inf
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                log.error(
                    f"Losses for metric {self.metric_name} contain NaN or Inf values: {losses[torch.isnan(losses) | torch.isinf(losses)]}"
                )
                # Check for nans in the preds and target
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    log.error(
                        f"Predictions contain NaN or Inf values: {preds[torch.isnan(preds) | torch.isinf(preds)]}"
                    )
                if torch.isnan(target).any() or torch.isinf(target).any():
                    log.error(
                        f"Targets contain NaN or Inf values: {target[torch.isnan(target) | torch.isinf(target)]}"
                    )
                # losses = torch.zeros_like(losses)
            if self.reduction == "median":
                # For median, we collect all losses in a list
                self.values.append(losses)
            else:
                getattr(self, f"sum_{self.metric_name}").add_(losses.sum())
                self.total_samples += batch_size

    def reduce_losses(self, losses):
        """Default reduction strategy - reduce over all dimensions except batch"""
        return losses.mean(dim=tuple(range(1, losses.dim())))

    def compute(self):
        """Compute the final metric value"""
        if self.reduction == "mean":
            if self.total_samples == 0:
                return torch.tensor(0.0, device=getattr(self, f"sum_{self.metric_name}").device)
            return getattr(self, f"sum_{self.metric_name}") / self.total_samples
        elif self.reduction == "median":
            if not self.values:
                return torch.tensor(0.0, device=self.device)
            all_values = torch.cat(self.values)
            return torch.median(all_values)


class MRSTFTMetric(AudioToAudioMetric):
    def __init__(self, device="cuda"):
        # Set up the monkey patch for backward compatibility
        auraloss.freq.apply_reduction = lambda losses, reduction: losses.mean(dim=(-1, -2))
        super().__init__(
            metric_name="mrstft",
            loss_fn_constructor=lambda **kwargs: auraloss.freq.MultiResolutionSTFTLoss(
                device=device, **kwargs
            ),
            device=device,
        )


class MRLogMelMetric(AudioToAudioMetric):
    def __init__(
        self,
        device="cuda",
        fft_sizes=None,
        hop_sizes=None,
        win_lengths=None,
        sample_rate=16000,
        n_bins=128,
    ):
        # Set up the monkey patch for backward compatibility
        auraloss.freq.apply_reduction = lambda losses, reduction: losses.mean(dim=(-1, -2))
        super().__init__(
            metric_name="mrstft",
            loss_fn_constructor=lambda **kwargs: auraloss.freq.MultiResolutionSTFTLoss(
                device=device, **kwargs
            ),
            device=device,
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            w_sc=0.0,  # No spectral convergence loss
            w_log_mag=1.0,  # Only log magnitude loss
            w_lin_mag=0.0,  # No linear magnitude loss
            sample_rate=sample_rate,
            scale="mel",  # MEL scale
            n_bins=n_bins,  # Number of mel bins
        )


class SDSDRMetric(AudioToAudioMetric):
    def __init__(self, device="cuda"):
        auraloss.time.apply_reduction = lambda losses, reduction: losses.mean(dim=(-1))
        super().__init__(
            metric_name="sdr",
            loss_fn_constructor=auraloss.time.SDSDRLoss,
            negate_result=True,  # Negate to convert loss to actual SDR value
            device=device,
            zero_mean=False,
        )


class SISDRMetric(AudioToAudioMetric):
    def __init__(self, device="cuda"):
        auraloss.time.apply_reduction = lambda losses, reduction: losses.mean(dim=(-1))
        super().__init__(
            metric_name="si_sdr",
            loss_fn_constructor=auraloss.time.SISDRLoss,
            negate_result=True,  # Negate to convert loss to actual SI-SDR value
            device=device,
            zero_mean=False,  # Zero mean is often used for SI-SDR
        )


class SNRMetric(AudioToAudioMetric):
    def __init__(self, device="cuda"):
        auraloss.time.apply_reduction = lambda losses, reduction: losses.mean(dim=(-1))
        super().__init__(
            metric_name="si_sdr",
            loss_fn_constructor=auraloss.time.SNRLoss,
            negate_result=True,  # Negate to convert loss to actual SI-SDR value
            device=device,
            zero_mean=False,
        )


def next_pow_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def ms_to_samples(ms, sample_rate):
    return int(sample_rate * ms / 1000)


def get_mss_loss_args(sample_rate):
    win_lengths = [ms_to_samples(ms, sample_rate) for ms in [10, 25, 50]]
    fft_sizes = [next_pow_2(wl) for wl in win_lengths]
    hop_lengths = [wl // 2 for wl in win_lengths]

    loss_args = {
        "fft_sizes": fft_sizes,
        "hop_sizes": hop_lengths,
        "win_lengths": win_lengths,
        "sample_rate": sample_rate,
        "scale": "mel",
        "n_bins": 80,
    }
    return loss_args
