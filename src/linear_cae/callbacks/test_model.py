import os

import auraloss
import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MeanMetric, Metric

# --- Wavetable Import ---
from wavetable import AudioLogger

# --- Metric Imports ---
from linear_cae.metrics import (
    ERankMetric,
    MRSTFTMetric,
    SISDRMetric,
    SNRMetric,
    relative_l2_error,
    svd_method,
)
from linear_cae.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        if losses.ndim > 1:
            losses = losses.mean(dim=tuple(range(1, losses.ndim)))
        else:
            losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    elif reduction == "none":
        pass
    return losses


class TestModelCallback(Callback):
    def __init__(
        self,
        num_log_examples=4,
        random_examples=True,
        reconstruction_loss=True,
        diffusion_steps=1,
        sdr=True,
        si_sdr=True,
        equivariance_loss=True,
        log_examples=True,
        log_latents=True,
        latent_space_metrics=True,
        composability_metrics=True,
        validation_sets: dict = None,
        test_sets: dict = None,
    ):
        """
        Args:
            num_log_examples: Number of examples to log.
            random_examples: Whether to log random examples from the batch.
            validation_sets: Dictionary of validation sets configuration.
            test_sets: Dictionary of test sets configuration.
        """
        self.num_log_examples = num_log_examples
        self.random_examples = random_examples
        self.diffusion_steps = diffusion_steps

        # Metrics flags
        self.sdr = sdr
        self.si_sdr = si_sdr
        self.reconstruction_loss = reconstruction_loss
        self.equivariance_loss = equivariance_loss
        self.log_examples = log_examples
        self.log_latents = log_latents
        self.latent_space_metrics = latent_space_metrics
        self.composability_metrics = composability_metrics

        # Data sets
        self.validation_sets = validation_sets if validation_sets is not None else {}
        self.test_sets = test_sets if test_sets is not None else {}

        all_dataset_names = list(self.validation_sets.keys()) + list(self.test_sets.keys())
        self.all_dataset_names = sorted(list(set(all_dataset_names)))

        if not self.test_sets:
            log.warning("No test sets provided, using validation sets for testing")
            self.test_sets = self.validation_sets
        if not self.validation_sets:
            log.warning("No validation sets provided")

        # Initialize Metrics
        self.all_metrics = torch.nn.ModuleDict(
            {name: torch.nn.ModuleDict() for name in self.all_dataset_names}
        )

        # Monkey patch auraloss reduction once
        auraloss.time.apply_reduction = apply_reduction
        auraloss.freq.apply_reduction = apply_reduction

        for val_set in self.all_dataset_names:
            if self.sdr:
                self.all_metrics[val_set]["snr"] = SNRMetric()
            if self.si_sdr:
                self.all_metrics[val_set]["si-sdr"] = SISDRMetric()
            if self.reconstruction_loss:
                self.all_metrics[val_set]["MR-STFT"] = MRSTFTMetric(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                if self.diffusion_steps > 1:
                    self.all_metrics[val_set][f"MR-STFT_{self.diffusion_steps}-steps"] = (
                        MRSTFTMetric(
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        )
                    )

            if self.equivariance_loss:
                self.all_metrics[val_set]["enc_homogeneity"] = MeanMetric()
                self.all_metrics[val_set]["dec_homogeneity_snr"] = SNRMetric()

            if self.latent_space_metrics:
                self.all_metrics[val_set]["latent_erank"] = ERankMetric()

            if self.composability_metrics:
                self.all_metrics[val_set]["enc_additivity"] = MeanMetric()
                self.all_metrics[val_set]["dec_additivity"] = MRSTFTMetric(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.all_metrics[val_set]["separability"] = MRSTFTMetric(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )

            self.all_metrics[val_set]["silence_latent_norm"] = MeanMetric()

        self.latents = {val_set: [] for val_set in self.all_dataset_names}
        self._reset()

    def on_fit_start(self, trainer, pl_module):
        log.info("TestModelCallback fit start - resetting metrics and examples.")

    def _reset(self, stage=None):
        """
        Reset the dictionaries used to store metrics and examples.
        """
        if stage is None:
            names = self.all_dataset_names
        else:
            names = self.validation_sets.keys() if stage == "validation" else self.test_sets.keys()

        self.log_examples = {
            val_set: {"original": None, "reconstructed": None, "latents": None} for val_set in names
        }
        # Reset latent storage for SVD calculation
        for val_set in names:
            self.latents[val_set] = []

        self.log_examples_left = dict.fromkeys(names, self.num_log_examples)

        for name in names:
            for metric in self.all_metrics[name].values():
                if isinstance(metric, Metric):
                    metric.reset()

    def _log_metrics(self, pl_module, stage="validation"):
        val_sets_to_log = (
            self.validation_sets.keys() if stage == "validation" else self.test_sets.keys()
        )
        metrics_to_log = {}
        prefix = "test" if stage == "test" else "val"

        for val_set in val_sets_to_log:
            for metric_name, metric in self.all_metrics[val_set].items():
                if isinstance(metric, Metric):
                    full_key = f"{prefix}/{val_set}/{metric_name}"
                    metrics_to_log[full_key] = metric.compute()

        if pl_module.global_rank == 0:
            pl_module.log_dict({**metrics_to_log}, rank_zero_only=True, on_epoch=True)

            if not isinstance(pl_module.logger, WandbLogger) and stage == "validation":
                wandb.log({**metrics_to_log}, step=pl_module.it, commit=False)

            if stage == "test":
                for k, v in metrics_to_log.items():
                    wandb.summary[f"{k}"] = v

    def on_validation_start(self, trainer, pl_module):
        # Move metrics to device
        for name in self.all_dataset_names:
            for metric in self.all_metrics[name].values():
                if isinstance(metric, Metric):
                    metric.to(pl_module.device)

        # pl_module.log_dict({"MR-STFT": 4}, rank_zero_only=True, on_epoch=True)

    def on_test_start(self, trainer, pl_module):
        self.on_validation_start(trainer, pl_module)
        self._reset(stage="test")

    def _shared_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        val_set="",
        stage="validation",
        total_n_batches=1,
    ):
        x = batch  # B, T
        B = x.shape[0]
        T = x.shape[-1]

        with pl_module.ema.average_parameters():
            model = pl_module

            latent = model.encode(x)
            generated_samples = model.generate(diffusion_steps=1, latents=latent)

            # --- Logging Examples ---
            # Check if we still need examples
            current_count = 0
            if self.log_examples[val_set]["original"] is not None:
                current_count = len(self.log_examples[val_set]["original"])

            if current_count < self.num_log_examples and pl_module.global_rank == 0:
                _log = True

                # Check for silence (prevent logging empty audio)
                if torch.norm(x, dim=-1).mean() < 0.01:
                    _log = False

                if self.random_examples and _log:
                    # Adaptive Probability: guarantees we fill the buffer by the end
                    # p = (needed_examples) / (remaining_batches * estimated_batch_size)
                    # We assume batch_size >= 1, so we are conservative.

                    needed = self.num_log_examples - current_count
                    remaining_batches = total_n_batches - batch_idx

                    # If we are running out of batches, probability goes to 1.0
                    if remaining_batches <= 0:
                        p = 1.0
                    else:
                        # We use 'needed' as the numerator to ensure we catch enough distinct batches
                        # if batch_size is small.
                        p = needed / remaining_batches

                        # Boost probability slightly to be safe against random misses
                        p = min(p * 1.5, 1.0)

                    if torch.rand(1).item() > p:
                        _log = False

                if _log:
                    # Helper to concatenate safely
                    def safe_concat(current, new_data):
                        if current is not None:
                            return torch.concat((current, new_data), dim=0)
                        return new_data

                    num_left = self.log_examples_left[val_set]

                    self.log_examples[val_set]["original"] = safe_concat(
                        self.log_examples[val_set]["original"], x[:num_left].cpu()
                    )
                    self.log_examples[val_set]["reconstructed"] = safe_concat(
                        self.log_examples[val_set]["reconstructed"],
                        generated_samples[:num_left, :T].cpu(),
                    )
                    self.log_examples[val_set]["latents"] = safe_concat(
                        self.log_examples[val_set]["latents"], latent[:num_left].cpu()
                    )

                    self.log_examples_left[val_set] = self.num_log_examples - len(
                        self.log_examples[val_set]["original"]
                    )

            # --- Metrics Computation ---

            # Store latents for SVD
            if self.latent_space_metrics:
                self.latents[val_set].append(latent.permute(0, 2, 1).cpu())

            # Equivariance metrics
            if self.equivariance_loss:
                scales = [0.25, 0.5, 0.75, 1.1]
                for scale in scales:
                    scale_tensor = torch.tensor(scale, device=pl_module.device)

                    # 1. Encoder Homogeneity (Matches Eval Script: relative_l2_error)
                    latent_scaled_input = model.encode(x * scale_tensor)
                    target_latent = latent * scale_tensor

                    # We use the relative_l2_error function imported from linear_cae.metrics
                    enc_homogeneity_err = relative_l2_error(latent_scaled_input, target_latent)
                    self.all_metrics[val_set]["enc_homogeneity"].update(enc_homogeneity_err)

                    # 2. Decoder Homogeneity (Matches Eval Script: SNR)
                    generated_scaled_latent = model.generate(
                        diffusion_steps=1, latents=target_latent
                    )[:, :T]
                    target_signal = generated_samples[:, :T] * scale_tensor

                    # Eval script uses negative SNR as a loss/metric
                    # We negate it back to get the positive SNR value for logging
                    self.all_metrics[val_set]["dec_homogeneity_snr"].update(
                        generated_scaled_latent, target_signal
                    )

                del latent_scaled_input, generated_scaled_latent

            # Reconstruction metrics
            if self.sdr:

                # Pure SNR calculation

                self.all_metrics[val_set]["snr"].update(
                    generated_samples[:, :T].unsqueeze(1), x.unsqueeze(1)
                )
            if self.si_sdr:
                self.all_metrics[val_set]["si-sdr"].update(
                    generated_samples[:, :T].unsqueeze(1), x.unsqueeze(1)
                )
            if self.reconstruction_loss:
                self.all_metrics[val_set]["MR-STFT"].update(
                    generated_samples[:, :T].unsqueeze(1), x.unsqueeze(1)
                )
            if self.diffusion_steps > 1:
                del generated_samples
                torch.cuda.empty_cache()
                generated_samples = model.generate(
                    diffusion_steps=self.diffusion_steps, latents=latent
                )
                self.all_metrics[val_set][f"MR-STFT_{self.diffusion_steps}-steps"].update(
                    generated_samples[:, :T].unsqueeze(1), x.unsqueeze(1)
                )

            # Separability and composability
            if not self.composability_metrics:
                return

            if B <= 1:
                log.warning(
                    "Batch size is <= 1, composability and separability metrics cannot be computed."
                )
                return

            if B % 2 != 0:
                B -= 1
                x = x[:B]
                latent = latent[:B]

            half_indices = torch.arange(0, B, 2, device=pl_module.device)
            x1 = x.index_select(0, half_indices)
            x2 = x.index_select(0, half_indices + 1)
            latent_x1 = latent.index_select(0, half_indices)
            latent_x2 = latent.index_select(0, half_indices + 1)

            mix = x1 + x2
            try:
                latent_mix = model.encode(mix)
            except Exception as e:
                log.warning(f"Error encoding mixture: {e}")
                return

            latent_sum = latent_x1 + latent_x2
            dec_sum = model.generate(diffusion_steps=1, latents=latent_sum)
            dec_mix = model.generate(diffusion_steps=1, latents=latent_mix)

            self.all_metrics[val_set]["dec_additivity"].update(
                dec_sum[:, :T].unsqueeze(1),
                dec_mix[:, :T].unsqueeze(1),
            )
            self.all_metrics[val_set]["enc_additivity"].update(
                relative_l2_error(latent_sum, latent_mix)
            )

            dec_sep2 = model.generate(diffusion_steps=1, latents=(latent_mix - latent_x1))
            dec_sep1 = model.generate(diffusion_steps=1, latents=(latent_mix - latent_x2))

            self.all_metrics[val_set]["separability"].update(
                dec_sep2[:, :T].unsqueeze(1), x2.unsqueeze(1)
            )
            self.all_metrics[val_set]["separability"].update(
                dec_sep1[:, :T].unsqueeze(1), x1.unsqueeze(1)
            )

            silence_latent = model.encode(torch.zeros_like(x[0]).unsqueeze(0))
            self.all_metrics[val_set]["silence_latent_norm"].update(
                torch.norm(silence_latent, dim=-1).mean()
            )

    def _shared_epoch_end(self, trainer, pl_module, stage="validation"):
        prefix = "test" if stage == "test" else "figs"
        val_sets = (
            list(self.validation_sets.keys()) if stage == "validation" else self.test_sets.keys()
        )

        for val_set in val_sets:
            # Latent space metrics (SVD)
            if self.latent_space_metrics and len(self.latents[val_set]) > 0:
                latents_local = torch.cat(self.latents[val_set], dim=0)
                latents_local = pl_module.all_gather(latents_local)

                if trainer.is_global_zero:
                    latents_local = latents_local.reshape(-1, latents_local.shape[-1])
                    self.all_metrics[val_set]["latent_erank"].update(latents_local)

                if trainer.is_global_zero and self.latent_space_metrics:
                    try:
                        svd_fig = svd_method(latents_local)
                        wandb.log(
                            {f"{prefix}/{val_set}_svd_log_singular_values": wandb.Image(svd_fig)},
                            step=pl_module.it if stage == "validation" else 1000000,
                            commit=False,
                        )
                        plt.close(svd_fig)
                    except Exception as e:
                        log.error(f"Error computing SVD: {e}")

        # Log all metrics
        self._log_metrics(pl_module, stage=stage)

        # ----- Figures and audio  ------
        if self.log_examples and trainer.is_global_zero:
            prefix = "test" if stage == "test" else "audio"
            table = AudioLogger(
                name=f"{prefix}_step_{pl_module.it}",
                sr=pl_module.sample_rate,
                root_dir=os.path.join(pl_module.output_dir, "html"),
                save_mode="embed",
                plot_config={
                    "spectrogram": {"dimensions": (300, 100)},
                    "waveform": {"dimensions": (300, 60), "ylim": (-1.0, 1.0)},
                },
            )
            for val_set in val_sets:
                original = self.log_examples[val_set]["original"]
                reconstructed = self.log_examples[val_set]["reconstructed"]

                if original is None or reconstructed is None:
                    log.warning(f"No examples to log for {val_set}, skipping HTML logging.")
                    return

                n_samples = min(len(original), len(reconstructed))
                # Let's permute element 3 of the batch to position 1

                # now let's do it, move element 3 to position 1
                original[[0, -1]] = original[[-1, 0]]
                reconstructed[[0, -1]] = reconstructed[[-1, 0]]

                for i in range(n_samples):
                    # Log the reconstructed audio as the main item, and original as Ground Truth
                    table.log(
                        row=f"ex_{i}",
                        col=val_set,
                        audio=reconstructed[i],
                        ground_truth=original[i],
                    )
            html_path = table.save()
            wandb.log(
                {f"{prefix}/audio_samples": wandb.Html(open(html_path))},
                step=pl_module.trainer.global_step,
                commit=False,
            )

        self._reset(stage=stage)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        log.info("Validation epoch end - logging metrics and examples.")
        self._shared_epoch_end(trainer, pl_module, stage="validation")

    def on_test_epoch_end(self, trainer, pl_module):
        self._shared_epoch_end(trainer, pl_module, stage="test")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        val_set = list(self.test_sets.keys())[dataloader_idx]
        self._shared_batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            val_set=val_set,
            stage="test",
            total_n_batches=trainer.num_test_batches[dataloader_idx],
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        val_set = list(self.validation_sets.keys())[dataloader_idx]
        self._shared_batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            val_set=val_set,
            stage="validation",
            total_n_batches=trainer.num_val_batches[dataloader_idx],
        )
