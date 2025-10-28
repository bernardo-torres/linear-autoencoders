import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_1d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_1d(x):
    return F.avg_pool1d(x, kernel_size=2, stride=2)


def upsample_2d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_2d(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def zero_init(module, init_as_zero):
    if init_as_zero:
        for p in module.parameters():
            p.detach().zero_()
    return module


class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, freq_dim, 1)))

    def forward(self, input):
        return input * self.scale


class DownsampleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        use_2d=False,
        normalize=False,
        padding_mode="zeros",
    ):
        super(DownsampleConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            )
        else:
            self.c = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
            )

    def forward(self, x):

        if self.normalize:
            x = self.norm(x)
        x = self.c(x)

        return x


class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(DownsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0)
        )

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = self.c(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        use_2d=False,
        normalize=False,
        padding_mode="zeros",
    ):
        super(UpsampleConv, self).__init__()
        self.normalize = normalize

        self.use_2d = use_2d

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.c = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            )

    def forward(self, x):

        if self.normalize:
            x = self.norm(x)

        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)

        return x


class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(UpsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)

        self.c = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), stride=1, padding="same"
        )

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = F.interpolate(x, scale_factor=(4, 1), mode="nearest")
        x = self.c(x)
        return x


# adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
class GaussianFourierProjection(torch.nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=128, scale=0.02):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(embedding_size // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.embedding_size // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, init_as_zero=False, **kwargs):
        super().__init__(**kwargs)
        self.init_as_zero = init_as_zero
        if init_as_zero:
            self.out_proj = zero_init(self.out_proj, True)

    # def _reset_parameters(self):
    #     super()._reset_parameters()
    # self.out_proj = zero_init(self.out_proj, self.init_as_zero)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, normalize=True, use_2d=False, init_as_zero=True):
        super(Attention, self).__init__()

        self.normalize = normalize
        self.use_2d = use_2d

        self.mha = MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
            init_as_zero=init_as_zero,
        )
        if normalize:
            self.norm = nn.GroupNorm(min(dim // 4, 32), dim)

    def forward(self, x):

        inp = x

        if self.normalize:
            x = self.norm(x)

        if self.use_2d:
            x = x.permute(0, 3, 2, 1)  # shape: [bs,len,freq,channels]
            bs, len, freq, channels = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            x = x.reshape(bs * len, freq, channels)  # shape: [bs*len,freq,channels]
        else:
            x = x.permute(0, 2, 1)  # shape: [bs,len,channels]

        x = self.mha(x, x, x, need_weights=False)[0]

        if self.use_2d:
            x = x.reshape(bs, len, freq, channels).permute(0, 3, 2, 1)
        else:
            x = x.permute(0, 2, 1)
        x = x + inp

        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_channels=None,
        kernel_size=3,
        downsample=False,
        upsample=False,
        normalize=True,
        leaky=False,
        attention=False,
        heads=4,
        use_2d=False,
        normalize_residual=False,
        min_res_dropout=16,
        dropout_rate=0.0,
        init_as_zero=True,
        padding_mode="zeros",
    ):
        super(ResBlock, self).__init__()
        self.normalize = normalize
        self.attention = attention
        self.upsample = upsample
        self.downsample = downsample
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.normalize_residual = normalize_residual
        self.use_2d = use_2d
        self.min_res_dropout = min_res_dropout

        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d
        self.conv1 = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            padding_mode=padding_mode,
        )
        self.conv2 = zero_init(
            Conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            init_as_zero,
        )
        if in_channels != out_channels:
            self.res_conv = Conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.res_conv = nn.Identity()
        if normalize:
            self.norm1 = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
            self.norm2 = nn.GroupNorm(min(out_channels // 4, 32), out_channels)
        if leaky:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.SiLU()
        if cond_channels is not None:
            self.proj_emb = zero_init(
                nn.Linear(cond_channels, out_channels), init_as_zero
            )
        self.dropout = nn.Dropout(dropout_rate)
        if attention:
            self.att = Attention(
                out_channels, heads, use_2d=use_2d, init_as_zero=init_as_zero
            )

    def forward(self, x, time_emb=None):
        if not self.normalize_residual:
            y = x.clone()
        if self.normalize:
            x = self.norm1(x)
        if self.normalize_residual:
            y = x.clone()
        x = self.activation(x)
        if self.downsample:
            if self.use_2d:
                x = downsample_2d(x)
                y = downsample_2d(y)
            else:
                x = downsample_1d(x)
                y = downsample_1d(y)
        if self.upsample:
            if self.use_2d:
                x = upsample_2d(x)
                y = upsample_2d(y)
            else:
                x = upsample_1d(x)
                y = upsample_1d(y)
        x = self.conv1(x)
        if time_emb is not None:
            if self.use_2d:
                x = x + self.proj_emb(time_emb)[:, :, None, None]
            else:
                x = x + self.proj_emb(time_emb)[:, :, None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1] <= self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x + y
        if self.attention:
            x = self.att(x)
        return x
