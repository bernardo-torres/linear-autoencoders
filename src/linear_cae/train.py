import numpy as np
import torch
import torch.distributed as dist
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from linear_cae.components.ema import ExponentialMovingAverage
from linear_cae.inference import Autoencoder
from linear_cae.utils import RankedLogger, get_grad_norm

log = RankedLogger(__name__, rank_zero_only=True)


def huber(x, y, w=None, mean=True, c=0.00054):
    diff = torch.flatten((x - y) ** 2, start_dim=1)
    data_dim = diff.shape[-1]
    c = c * torch.sqrt(torch.ones((1,), device=x.device) * data_dim)
    diff = torch.sum(diff, -1)
    diff = torch.sqrt(diff + c**2) - c
    diff = torch.nan_to_num(diff)
    if w is not None:
        diff = diff * w.squeeze()
    if mean:
        return diff.mean()
    else:
        return diff


class GainScaler(torch.nn.Module):
    """
    Manages gain sampling and annealing for Implicit Homogeneity Regularization.
    """

    def __init__(
        self,
        batch_size,
        prob,
        gain_min=0.1,
        gain_max=1.0,
        zero_threshold=0.05,
        device="cuda",
        prob_decay=False,
        gain_decay=False,
        total_iters=300000,
        start_hold_fraction=0.2,
        end_zero_fraction=0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.prob = prob
        self.initial_prob = prob
        self.gain_min = gain_min
        self.initial_gain_min = gain_min
        self.gain_max = gain_max
        self.initial_gain_max = gain_max
        self.zero_threshold = zero_threshold
        self.device = device
        self.prob_decay = prob_decay
        self.gain_decay = gain_decay
        self.total_iters = total_iters
        self.start_hold_fraction = start_hold_fraction
        self.end_zero_fraction = end_zero_fraction
        self.gain = None

    def update_gain_scaler(self, it):
        """
        Update the gain scaler based on the current iteration.
        """
        if not self.prob_decay and not self.gain_decay:
            return

        start_decay_iter = self.total_iters * self.start_hold_fraction
        end_decay_iter = self.total_iters * (1 - self.end_zero_fraction)
        decay_duration = end_decay_iter - start_decay_iter
        progress = (it - start_decay_iter) / decay_duration
        if self.prob_decay:
            if it <= start_decay_iter:
                self.prob = self.initial_prob
            elif it >= end_decay_iter:
                self.prob = 0.0
            else:
                self.prob = self.initial_prob * (0.5 * (1 + np.cos(progress * np.pi)))

        if self.gain_decay:
            # Decay the gain range over time, symmetric around 1
            if it <= start_decay_iter:
                return
            if it >= end_decay_iter:
                self.gain_min = 1.0
                self.gain_max = 1.0
                return

            # Cosine factor goes from 1 down to 0 during the decay phase
            cosine_factor = 0.5 * (1 + np.cos(progress * np.pi))
            # Symmetrically anneal the gain range towards 1.0
            self.gain_min = 1.0 - (1.0 - self.initial_gain_min) * cosine_factor
            self.gain_max = 1.0 + (self.initial_gain_max - 1.0) * cosine_factor

    def sample_gain(self):
        apply_gain_mask = torch.rand((self.batch_size,), device=self.device) < self.prob
        random_gains = (
            torch.rand((self.batch_size,), device=self.device) * (self.gain_max - self.gain_min)
            + self.gain_min
        )
        # Zero out if below threshold
        random_gains = torch.where(random_gains < self.zero_threshold, 0.0, random_gains)

        identity_gains = torch.ones((self.batch_size,), device=self.device)
        self.gain = torch.where(apply_gain_mask, random_gains, identity_gains)

    def cap_gain(self, x):
        """Caps the gain to ensure max absolute value of output <= 1.0"""
        if self.gain is None:
            return
        dims_to_reduce = tuple(range(1, x.ndim))
        max_abs_val = torch.amax(torch.abs(x), dim=dims_to_reduce)
        max_allowed_gain = 1.0 / (max_abs_val + 1e-7)
        capped_magnitude = torch.min(torch.abs(self.gain), max_allowed_gain)
        self.gain = torch.copysign(capped_magnitude, self.gain)

    def forward(self, x):
        if self.gain is None:
            return x
        return x * self.gain.view(-1, *((1,) * (x.ndim - 1)))

    def inverse(self, x):
        if self.gain is None:
            return x
        return x / self.gain.view(-1, *((1,) * (x.ndim - 1)))


class CAETrainer(LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        diffusion: torch.nn.Module,
        frontend: torch.nn.Module,
        lr=1e-4,
        final_lr=1e-5,
        lr_decay="cosine",
        optimizer="adamw",
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
        warmup_steps=1000,
        enable_ema=True,
        ema_update_every=1,
        ema_momentum=0.9999,
        warmup_ema=True,
        sample_rate=44100,
        alpha_rescale: float = 0.65,  # alpha rescale parameter for STFT representation
        beta_rescale: float = 0.34,  # beta rescale parameter for STFT representation
        compile_model=False,
        # Config flags (injected by Hydra)
        gain_equivariance="teacher_student",  # None, 'encoder', 'student', 'teacher', 'teacher_student'
        prob_additivity_switch=1.0,
        gain_scaler=None,
        additivity=True,
        mixit_augmentation=False,
        scale_augmentation=False,
        normalize_gain_augmentation=False,
        mix_weights="random",
        output_dir=None,
    ):
        super().__init__()
        # Save params for checkpointing
        self.save_hyperparameters(ignore=["generator", "diffusion", "frontend"])

        self.generator = generator
        self.diffusion = diffusion
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.gain_scaler = gain_scaler

        self.enable_ema = enable_ema
        self.ema_update_every = ema_update_every
        self.ema_momentum = ema_momentum
        self.warmup_ema = warmup_ema

        self.lr = lr
        self.final_lr = final_lr
        self.lr_decay = lr_decay
        self.optimizer_name = optimizer
        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.warmup_steps = warmup_steps
        self.compile_model = compile_model

        # Methodology Flags
        self.additivity = additivity
        self.mixit_augmentation = mixit_augmentation
        self.scale_augmentation = scale_augmentation
        self.mix_weights = mix_weights
        self.normalize_gain_augmentation = normalize_gain_augmentation

        self.gain_equivariance = gain_equivariance
        self.prob_additivity_switch = prob_additivity_switch
        self.scale_encoder = "encoder" in (self.gain_equivariance or "")
        self.scale_student = "student" in (self.gain_equivariance or "")
        self.scale_teacher = "teacher" in (self.gain_equivariance or "")

        if self.mixit_augmentation:
            assert (
                not self.additivity
            ), "MixIt Augmentation cannot be used with Additivity regularization."

        if self.scale_augmentation:
            assert (
                self.gain_equivariance is None
            ), "Scale Augmentation cannot be used with Gain Equivariance."

        if self.gain_scaler is None:
            log.info("Using legacy GainScaler. Make sure this is intended.")
            self.gain_scaler = GainScaler(
                1,
                prob=(
                    0.8 if self.gain_equivariance is not None or self.scale_augmentation else 0.0
                ),
                gain_min=0.1,
                gain_max=2.0,
                device="cuda",
            )

        self.ema = ExponentialMovingAverage(
            self.generator.parameters(),
            decay=self.ema_momentum,
            use_num_updates=self.warmup_ema,
        )
        self.ema_initialized = False

        # inference model
        self.autoencoder_inference_model = Autoencoder(
            self.frontend, self.generator, self.diffusion
        )
        # Torch Compile
        if self.compile_model:
            log.info("Compiling train_it method...")
            try:
                self.train_it = torch.compile(self.train_it)
            except Exception as e:
                log.warning(f"Failed to compile train_it: {e}")

        self.output_dir = output_dir

    # some properties for legacy
    @property
    def it(self):
        return self.trainer.global_step

    @property
    def gen(self):
        return self.generator

    @property
    def total_iters(self):
        return self.trainer.max_steps

    def configure_optimizers(self):
        # Explicit choice of optimizer
        if self.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(self.generator.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == "radam":
            return torch.optim.RAdam(self.generator.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def linearly_combine(self, batch, weights=0.5):
        if isinstance(weights, torch.Tensor) and weights.ndim == 1:
            br_weights = weights.view(-1, *([1] * (batch.ndim - 1)))
        else:
            br_weights = weights
        return batch * br_weights + torch.roll(batch, shifts=1, dims=0) * (1 - br_weights)

    def train_it(self, wv, current_gain_prob, current_gain_min, current_gain_max):
        # We pass scalar values to avoid graph breaks in torch.compile
        batch_size = wv.shape[0]

        # Explicitly update scaler state
        gain_scaler = self.gain_scaler
        gain_scaler.batch_size = batch_size  # Update batch size in case it changed
        gain_scaler.device = wv.device  # Update device in case it changed
        gain_scaler.prob = current_gain_prob
        gain_scaler.gain_min = current_gain_min
        gain_scaler.gain_max = current_gain_max

        # Mixing (Additivity / MixIt)
        lc_weight = None
        if self.additivity or self.mixit_augmentation:
            if self.mix_weights == "random":
                lc_weight = torch.rand((batch_size,), device=wv.device) * 0.5 + 0.5
            else:
                lc_weight = torch.tensor(0.5, device=wv.device)

            wv = torch.cat([wv, self.linearly_combine(wv, weights=lc_weight)], dim=0)

            batch_size = wv.shape[0]
            gain_scaler.batch_size = batch_size

        # Gain Sampling
        gain_scaler.sample_gain()

        #  Frontend & Scaling
        data_input = wv
        data_rep = self.frontend.to_representation(data_input)

        if self.scale_augmentation or self.gain_equivariance:
            gain_scaler.cap_gain(data_input)
            scaled_data_rep = self.frontend.to_representation(gain_scaler(data_input))
        else:
            scaled_data_rep = data_rep

        # Noise
        step = self.diffusion.get_step_schedule(min(self.it, self.total_iters))
        self.step = step

        if self.diffusion.use_lognormal:  # TODO - move this logic into Diffusion class
            arbitrary_high_number = 10000
            w = self.diffusion.get_sampling_weights(arbitrary_high_number, device=data_rep.device)
            inds = torch.multinomial(w, batch_size, replacement=True).float()
            inds = (inds + torch.rand_like(inds)) / float(arbitrary_high_number - 1)
        else:
            inds = torch.rand((batch_size,)).type_as(data_rep, device=data_rep.device)

        sigmas = self.diffusion.get_sigma_continuous(inds)
        inds_step = self.diffusion.get_step_continuous(inds, step)
        sigmas_step = self.diffusion.get_sigma_continuous(inds_step)

        noises = torch.randn_like(data_rep)  # shared noise direction

        # Add noise
        if self.scale_student:
            noisy_student = self.diffusion.add_noise(scaled_data_rep, noises, sigmas)
        else:
            noisy_student = self.diffusion.add_noise(data_rep, noises, sigmas)
        # t
        if self.scale_teacher:
            noisy_teacher = self.diffusion.add_noise(scaled_data_rep, noises, sigmas_step)
        else:
            noisy_teacher = self.diffusion.add_noise(data_rep, noises, sigmas_step)

        # Encode
        latents = self.generator.encoder(data_rep)
        latents_student = latents
        latents_teacher = latents
        half = latents_student.shape[0] // 2

        # Replace the latents of (x+y) with the latents of x + latents of y
        if self.additivity:
            first_half = latents_student[:half]
            second_half = latents_student[half:]
            new_half = self.linearly_combine(first_half, weights=lc_weight.to(first_half.dtype))

            # new logic (compile-friendly)
            apply_composable = (
                torch.rand((half,), device=latents_student.device) < self.prob_additivity_switch
            )

            lc_mask = apply_composable.float().view(-1, *([1] * (new_half.ndim - 1)))
            mixed_half = lc_mask * new_half + (1 - lc_mask) * second_half
            # old logic
            # latents_student = torch.cat([first_half, new_half], dim=0)
            latents_student = torch.cat([first_half, mixed_half], dim=0)
            latents_teacher = latents_student

        if self.gain_equivariance is not None:
            if self.scale_encoder:
                # ENCODER EQUIVARIANCE:
                # The input to the encoder was scaled by gain `alpha`. We now scale the
                # resulting latents by `1/alpha`. The diffusion model then learns to
                # reconstruct the ORIGINAL unscaled data from these inversely scaled
                # latents. This forces the encoder to learn: Enc(alpha * x) = alpha * Enc(x).
                scaled_latents = gain_scaler.inverse(latents_student)
                latents_student = scaled_latents
                latents_teacher = scaled_latents

            else:
                # DENOISER EQUIVARIANCE:
                # The encoder input was NOT scaled. We scale the data before adding noise
                # for the student and/or teacher branches, so we must also scale the
                # corresponding latents by `alpha`. This forces the denoiser to learn:
                # Dec(alpha * z, sigma) = alpha * Dec(z, sigma).
                if self.scale_student:
                    latents_student = gain_scaler(latents_student)
                if self.scale_teacher:
                    latents_teacher = gain_scaler(latents_teacher)

        # Decode
        same = self.scale_student == self.scale_teacher
        pyramid_latents = self.generator.decoder(latents_student)
        pyramid_latents_teacher = (
            self.generator.decoder(latents_teacher) if not same else pyramid_latents
        )
        fdata = self.generator.forward_generator(
            latents_teacher, noisy_teacher, sigmas_step, pyramid_latents_teacher
        ).detach()
        fdata_plus_one = self.generator.forward_generator(
            latents_student, noisy_student, sigmas, pyramid_latents
        )

        # If we have an unbalanced branch (scaling only one side), we need to scale the target
        if self.scale_student and not self.scale_teacher:
            fdata = gain_scaler(fdata)

        # Loss
        loss_weight = self.diffusion.get_loss_weight(sigmas, sigmas_step)
        loss = huber(fdata, fdata_plus_one, loss_weight)
        loss = {"consistency_loss": loss}

        return loss, None, step, sigmas, sigmas_step

    def training_step(self, batch, batch_idx):
        self.update_learning_rate()
        if self.normalize_gain_augmentation:
            batch = batch / torch.max(torch.abs(batch), dim=-1, keepdim=True)[0]
            # random gain to each sample
            gain_range = [0.1, 1]
            gain = (
                torch.rand((batch.shape[0],), device=batch.device) * (gain_range[1] - gain_range[0])
                + gain_range[0]
            )
            batch = batch * gain[:, None]

        if self.mixit_augmentation:
            # With probability 0.5, mix each sample with a randomly chosen sample from the batch.
            perm = torch.randperm(batch.size(0), device=batch.device)
            mix_mask = (torch.rand(batch.size(0), device=batch.device) < 0.5).to(batch.dtype)
            # Reshape mask to broadcast along the remaining dimensions.
            mix_mask = mix_mask.view(-1, *[1] * (batch.ndim - 1))
            batch = mix_mask * ((batch + batch[perm]) * 0.5) + (1 - mix_mask) * batch

        self.gain_scaler.update_gain_scaler(
            self.it
        )  # Update gain probabilities based on the current iteration
        loss, _, step, sigmas, sigmas_step = self.train_it(
            batch, self.gain_scaler.prob, self.gain_scaler.gain_min, self.gain_scaler.gain_max
        )

        losses = {}
        if isinstance(loss, dict):
            # If loss is a dict, we assume it contains multiple losses
            # and we sum them up
            loss_sum = sum(loss.values())
            losses = {loss_name: loss_value.detach() for loss_name, loss_value in loss.items()}

            self.log_dict(losses, on_step=True, sync_dist=True)
            loss = loss_sum

        if not self.automatic_optimization:
            self.manual_backward(loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            self.update_ema()

        grad_norm = get_grad_norm(self.gen.parameters())

        self.log("loss", loss, on_step=True, sync_dist=True)

        # All‐reduce across GPUs (sum), then average
        if self.it % self.trainer.log_every_n_steps == 0:
            loss_tensor = loss.detach()
            if dist.is_initialized():
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= dist.get_world_size()

                for _loss_name, _loss in losses.items():
                    _loss_tensor = _loss.detach()
                    dist.all_reduce(_loss_tensor, op=dist.ReduceOp.SUM)
                    _loss_tensor /= dist.get_world_size()
                    losses[_loss_name] = _loss_tensor.item()

            loss_value = loss_tensor.item()

            if self.trainer.is_global_zero:
                wandb.log(
                    {
                        "loss": loss_value,
                        "learning rate": self.optimizers().param_groups[0]["lr"],
                        "gradient norm": grad_norm.item(),
                        "consistency step": step,
                        "sigmas": sigmas.mean().item(),
                        "sigmas_step": sigmas_step.mean().item(),
                        "it": self.it,  # for debugging, should be equal to self.global_step
                        "gain_scaler_prob": (self.gain_scaler.prob if self.gain_scaler else 0),
                        "gain_scaler_gain_min": (
                            self.gain_scaler.gain_min if self.gain_scaler else 1
                        ),
                        "gain_scaler_gain_max": (
                            self.gain_scaler.gain_max if self.gain_scaler else 1
                        ),
                        **losses,
                    },
                    step=self.trainer.global_step,
                    commit=False,
                )

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.automatic_optimization:
            self.update_ema()

    # ------ Legacy code, does nothing, all validation was moved to external callbacks ------
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return None

    def test_step(self, batch, batch_idx, *args, **kwargs):
        return None

    @rank_zero_only
    def update_ema(self):
        if self.enable_ema and self.it % self.ema_update_every == 0:
            # TODO - implement this in a nicer way
            if not self.ema_initialized:
                if self.ema.shadow_params[0].device != next(self.gen.parameters()).device:
                    device = next(self.gen.parameters()).device
                    self.ema.shadow_params = [p.to(device) for p in self.ema.shadow_params]
                self.ema_initialized = True
            self.ema.update()

    def update_learning_rate(self):
        if self.it < self.warmup_steps:
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = self.lr * (self.it / self.warmup_steps)
        else:
            if self.lr_decay == "cosine":
                decay_iters = self.total_iters - self.warmup_steps
                current_iter = (self.it - self.warmup_steps) % (
                    self.total_iters - self.warmup_steps
                )
                new_learning_rate = self.final_lr + (
                    0.5
                    * (self.lr - self.final_lr)
                    * (1.0 + np.cos((current_iter / decay_iters) * np.pi))
                )
            elif self.lr_decay == "linear":
                decay_iters = self.total_iters - self.warmup_steps
                current_iter = (self.it - self.warmup_steps) % (
                    self.total_iters - self.warmup_steps
                )
                new_learning_rate = self.lr - (
                    (self.lr - self.final_lr) * (current_iter / decay_iters)
                )
            elif self.lr_decay == "inverse_sqrt":
                new_learning_rate = (
                    self.lr * (self.warmup_steps**0.5) / max(self.it, self.warmup_steps) ** 0.5
                )
            elif self.lr_decay is None:
                new_learning_rate = self.lr
            else:
                raise ValueError('lr_decay must be None, "cosine", "linear", or "inverse_sqrt"')

            for param_group in self.optimizers().param_groups:
                param_group["lr"] = new_learning_rate

    # --------- Inference ---------
    def encode(self, waveform, extract_features=False):
        # return self.generator.encoder(to_representation_encoder(waveform))
        return self.generator.encoder(
            self.frontend.to_representation(waveform), extract_features=extract_features
        )

    def bottleneck_features(self, feats, from_original_shape=False):
        return self.generator.encoder.bottleneck_features(
            feats, from_original_shape=from_original_shape
        )

    def decode(self, z, *args, **kwargs):
        """
        Decode the latent representation z back to waveform.
        This method is used for inference and can be called directly.
        """
        return self.generate(latents=z, *args, **kwargs)

    def generate(self, num_samples=9, diffusion_steps=3, seconds=None, latents=None):
        freq_downsample_list = self.generator.freq_downsample_list
        if seconds is None:
            sample_length = 64  # hparams.data_length
        else:
            downscaling_factor = 2 ** freq_downsample_list.count(0)
            sample_length = int(
                (((seconds * self.sample_rate) // self.generator.hop) // downscaling_factor)
                * downscaling_factor
            )
        if latents is not None:
            num_samples = latents.shape[0]
            downscaling_factor = 2 ** freq_downsample_list.count(0)
            sample_length = int(latents.shape[-1] * downscaling_factor)
        initial_noise = torch.randn(
            (
                num_samples,
                self.generator.data_channels,
                self.generator.hop * 2,
                sample_length,
            ),
            device=self.device,
        )
        initial_noise = initial_noise * self.generator.sigma_max
        generated_images = self.diffusion.reverse_diffusion(
            self.generator, initial_noise, diffusion_steps, latents=latents
        )
        # return to_waveform(generated_images)
        return self.frontend.to_waveform(generated_images)
