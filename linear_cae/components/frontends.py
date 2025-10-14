import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def overlap_and_add(signal, frame_step):

    outer_dimensions = signal.shape[:-2]
    outer_rank = torch.numel(torch.tensor(outer_dimensions))

    def full_shape(inner_shape):
        s = torch.cat([torch.tensor(outer_dimensions), torch.tensor(inner_shape)], 0)
        s = list(s)
        s = [int(el) for el in s]
        return s

    frame_length = signal.shape[-1]
    frames = signal.shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    signal = torch.nn.functional.pad(signal, (0, segments * frame_step - frame_length, 0, segments))

    shape = full_shape([frames + segments, segments, frame_step])
    signal = torch.reshape(signal, shape)

    perm = torch.cat(
        [torch.arange(0, outer_rank), torch.tensor([el + outer_rank for el in [1, 0, 2]])], 0
    )
    perm = list(perm)
    perm = [int(el) for el in perm]
    signal = torch.permute(signal, perm)

    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal[..., : (frames + segments - 1) * segments, :]

    shape = full_shape([segments, (frames + segments - 1), frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal.sum(-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = torch.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal


def inverse_stft_window(frame_length, frame_step, forward_window):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]


def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
            frame_length - frames_overlap
        )
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames


def istft(SP, fac=4, hop_size=256, device="cuda", window=None):
    x = torch.fft.irfft(SP, dim=-2)
    if window is None:
        window = torch.hann_window(fac * hop_size).to(device)
    window = inverse_stft_window(fac * hop_size, hop_size, window)
    x = x * window.unsqueeze(-1)
    return overlap_and_add(x.permute(0, 2, 1), hop_size)


def stft(wv, fac=4, hop_size=256, device="cuda", window=None):
    if window is None:
        window = torch.hann_window(fac * hop_size).to(device)
    framed_signals = frame(wv, fac * hop_size, hop_size)
    framed_signals = framed_signals * window
    return torch.fft.rfft(framed_signals, n=None, dim=-1, norm=None).permute(0, 2, 1)


def normalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    return (
        beta_rescale
        * (x.abs() ** alpha_rescale).to(torch.complex64)
        * torch.exp(1j * torch.angle(x))
    )


def denormalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    x = x / beta_rescale
    return (x.abs() ** (1.0 / alpha_rescale)).to(torch.complex64) * torch.exp(1j * torch.angle(x))


class ScaledComplexSTFT(nn.Module):
    def __init__(
        self, hop_size=256, n_fft_factor=4, alpha_rescale=0.3, beta_rescale=1.5, sample_rate=None
    ):
        super().__init__()
        self.hop_size = hop_size
        self.fac = n_fft_factor
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.sample_rate = sample_rate
        self.register_buffer("window", torch.hann_window(self.fac * self.hop_size))

    def forward(self, x, inverse=False):
        if inverse:
            return self.to_waveform(x)
        else:
            return self.to_representation(x)

    def to_representation(self, wv):
        if wv.dim() == 1:
            wv = wv.unsqueeze(0)  # Add batch dimension if missing
        X = stft(wv, hop_size=self.hop_size, fac=self.fac, device=wv.device, window=self.window)[
            :, : self.hop_size * 2, :
        ]
        X = normalize_complex(X, self.alpha_rescale, self.beta_rescale)
        return torch.stack((X.real, X.imag), dim=-3)

    def to_waveform(self, x):
        x = F.pad(x, (0, 0, 0, 1))  # pad time dim
        real, imag = torch.chunk(x, 2, dim=-3)
        X = torch.complex(real.squeeze(-3), imag.squeeze(-3))
        X = denormalize_complex(X, self.alpha_rescale, self.beta_rescale)
        return istft(X, hop_size=self.hop_size, fac=self.fac, device=X.device, window=self.window)
