import numpy as np
import torch
import torch.nn as nn

from linear_cae.components.blocks import (
    DownsampleConv,
    DownsampleFreqConv,
    FreqGain,
    GaussianFourierProjection,
    PositionalEmbedding,
    ResBlock,
    UpsampleConv,
    UpsampleFreqConv,
    zero_init,
)
from linear_cae.components.diffusion import get_c


# "+model.eqvae=true"
# Let's refactor the bottleneck to be a separate class
class Bottleneck(nn.Module):
    def __init__(
        self,
        input_channels,
        bottleneck_base_channels,
        bottleneck_channels=64,
        activation_bottleneck=nn.Tanh(),
        num_layers=4,
        normalization=True,
        dropout_rate=0.0,
        min_res_dropout=16,
        init_as_zero=True,
        freq_dim=16,
        type="encoder",
    ):
        super().__init__()
        self._type = type

        self.input_channels = input_channels
        self.freq_dim = freq_dim
        conv_in_channels_1 = input_channels
        conv_in_channels_2 = bottleneck_base_channels
        conv_out_channels_1 = bottleneck_base_channels
        conv_out_channels_2 = bottleneck_channels
        if self._type == "decoder":
            # We invert input_channels and bottleneck_base_channels
            conv_in_channels_1, conv_in_channels_2 = (
                bottleneck_channels,
                bottleneck_base_channels,
            )
            conv_out_channels_1, conv_out_channels_2 = (
                bottleneck_base_channels,
                input_channels,
            )

        self.conv_inp = nn.Conv1d(
            conv_in_channels_1,
            conv_in_channels_2,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        layers = []
        for _ in range(num_layers):
            layers.append(
                ResBlock(
                    bottleneck_base_channels,
                    bottleneck_base_channels,
                    normalize=normalization,
                    use_2d=False,
                    dropout_rate=dropout_rate,
                    min_res_dropout=min_res_dropout,
                    init_as_zero=init_as_zero,
                )
            )
        self.layers = nn.Sequential(*layers)
        if self._type == "encoder":
            self.norm_out = nn.GroupNorm(
                min(bottleneck_base_channels // 4, 32), bottleneck_base_channels
            )
            self.activation_out = nn.SiLU()
            self.activation_bottleneck = activation_bottleneck

        self.conv_out = nn.Conv1d(
            conv_out_channels_1,
            conv_out_channels_2,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, x):
        x = self.conv_inp(x)
        x = self.layers(x)
        if self._type == "encoder":
            x = self.norm_out(x)
            x = self.activation_out(x)
        x = self.conv_out(x)
        if self._type == "encoder":
            x = self.activation_bottleneck(x)
        if self._type == "decoder":
            # We need to chunk the channel dimension back to the original input channels
            B, C, T = x.shape
            # x_ls = torch.chunk(x.unsqueeze(-2), self.conv_in.in_channels, dim=1)  # chunk by input channels
            x = x.reshape(B, -1, self.freq_dim, T)
        return x



class Encoder(nn.Module):
    def __init__(
        self,
        layers_list=[1, 1, 1, 1, 1],
        attention_list=[0, 0, 1, 1, 1],
        multipliers_list=[1, 2, 4, 4, 4],
        base_channels=64,
        data_channels=1,
        bottleneck_base_channels=512,
        num_bottleneck_layers=4,
        bottleneck_channels=64,
        hop=256,
        freq_downsample_list=[1, 0, 0, 0],
        normalization=True,
        pre_normalize_downsampling_encoder=True,
        pre_normalize_2d_to_1d=True,
        frequency_scaling=True,
        dropout_rate=0.0,
        min_res_dropout=16,
        init_as_zero=True,
        bottleneck_layers=None,
        activation_bottleneck=nn.Tanh(),
        padding_mode="zeros",
        **kwargs,
    ):
        super().__init__()

        self.layers_list = list(layers_list)
        self.multipliers_list = list(multipliers_list)
        self.frequency_scaling = frequency_scaling

        input_channels = base_channels * multipliers_list[0]
        Conv = nn.Conv2d
        self.gain = FreqGain(freq_dim=hop * 2)

        self.conv_inp = Conv(
            data_channels,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )

        self.freq_dim = (hop * 2) // (4 ** freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** freq_downsample_list.count(0))

        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(layers_list, multipliers_list)
        ):
            output_channels = base_channels * multiplier
            for _ in range(num_layers):
                down_layers.append(
                    ResBlock(
                        input_channels,
                        output_channels,
                        normalize=normalization,
                        attention=attention_list[i] == 1,
                        heads=4,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=init_as_zero,
                        padding_mode=padding_mode,
                    )
                )
                input_channels = output_channels
            if i != len(layers_list) - 1:
                if freq_downsample_list[i] == 1:
                    down_layers.append(
                        DownsampleFreqConv(
                            input_channels, normalize=pre_normalize_downsampling_encoder
                        )
                    )
                else:
                    down_layers.append(
                        DownsampleConv(
                            input_channels,
                            use_2d=True,
                            normalize=pre_normalize_downsampling_encoder,
                            padding_mode=padding_mode,
                        )
                    )

        self.down_layers = nn.ModuleList(down_layers)

        if pre_normalize_2d_to_1d:
            self.prenorm_1d_to_2d = nn.GroupNorm(
                min(input_channels // 4, 32), input_channels
            )

        output_channels = bottleneck_base_channels

        if bottleneck_layers is not None:
            self.legacy_bottleneck = False
            # self.bottleneck_layers = bottleneck_layers
            raise NotImplementedError("DynamicBottleneck not implemented yet")
        else:
            self.legacy_bottleneck = True
            # self.bottleneck_layers = Bottleneck(
            #     input_channels * self.freq_dim,
            #     bottleneck_base_channels,
            #     bottleneck_channels=bottleneck_channels,
            #     activation_bottleneck=activation_bottleneck,
            #     num_layers=num_bottleneck_layers,
            #     normalization=normalization,
            #     dropout_rate=dropout_rate,
            #     min_res_dropout=min_res_dropout,
            #     init_as_zero=init_as_zero,
            # )
            bottleneck_layers = [
                nn.Conv1d(
                    input_channels * self.freq_dim,
                    output_channels,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                    padding_mode=padding_mode,
                )
            ]
            for _ in range(num_bottleneck_layers):
                bottleneck_layers.append(
                    ResBlock(
                        output_channels,
                        output_channels,
                        normalize=normalization,
                        use_2d=False,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=init_as_zero,
                        padding_mode=padding_mode,
                    )
                )
            # self.bottleneck_layers = nn.ModuleList(bottleneck_layers)
            self.bottleneck_layers = nn.Sequential(
                *bottleneck_layers
            )  # may be more efficient

            self.norm_out = nn.GroupNorm(min(output_channels // 4, 32), output_channels)
            self.activation_out = nn.SiLU()
            self.conv_out = nn.Conv1d(
                output_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            )
            self.activation_bottleneck = activation_bottleneck

    def forward(self, x, extract_features=False, original_shape=False):
        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = self.gain(x)

        k = 0
        for i, num_layers in enumerate(self.layers_list):
            for _ in range(num_layers):
                x = self.down_layers[k](x)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.down_layers[k](x)
                k += 1

        if hasattr(self, "prenorm_1d_to_2d"):
            x = self.prenorm_1d_to_2d(x)

        if not original_shape:
            x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))

        if extract_features:
            return x

        if self.legacy_bottleneck:
            for layer in self.bottleneck_layers:
                x = layer(x)

            x = self.norm_out(x)
            x = self.activation_out(x)
            x = self.conv_out(x)
            x = self.activation_bottleneck(x)
        else:
            x = self.bottleneck_layers(x)

        return x

    def bottleneck_features(self, x, from_original_shape=False):
        """
        Convert features to latents.
        """
        # x = self.bottleneck_layers(x)
        # x = self.norm_out(x)
        # x = self.activation_out(x)
        # x = self.conv_out(x)
        # x = self.activation_bottleneck(x)
        if from_original_shape:
            x = x.reshape(
                x.size(0), self.bottleneck_layers.input_channels, -1, x.size(-1)
            )
        if self.legacy_bottleneck:
            for layer in self.bottleneck_layers:
                x = layer(x)
            x = self.norm_out(x)
            x = self.activation_out(x)
            x = self.conv_out(x)
            x = self.activation_bottleneck(x)
        else:
            x = self.bottleneck_layers(x)
        return x



class Decoder(nn.Module):
    def __init__(
        self,
        layers_list=[1, 1, 1, 1, 1],
        attention_list=[0, 0, 1, 1, 1],
        multipliers_list=[1, 2, 4, 4, 4],
        base_channels=64,
        bottleneck_base_channels=512,
        bottleneck_channels=64,
        cond_channels=256,
        hop=256,
        freq_downsample_list=[1, 0, 0, 0],
        num_bottleneck_layers=4,
        normalization=True,
        dropout_rate=0.0,
        min_res_dropout=16,
        init_as_zero=True,
        bottleneck_layers=None,
        padding_mode="zeros",
    ):
        super().__init__()

        self.layers_list = list(layers_list)
        self.multipliers_list = list(multipliers_list)
        input_channels = base_channels * multipliers_list[-1]

        self.freq_dim = (hop * 2) // (4 ** freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim // (2 ** freq_downsample_list.count(0))

        if bottleneck_layers is not None:
            self.legacy_bottleneck = False
            raise NotImplementedError("DynamicBottleneck not implemented yet")
        else:
            # Literally same thing as the encoder but with type="decoder"
            self.legacy_bottleneck = True
            # self.bottleneck_layers = Bottleneck(
            #     input_channels * self.freq_dim,
            #     bottleneck_base_channels,
            #     bottleneck_channels=bottleneck_channels,
            #     activation_bottleneck=None,
            #     num_layers=num_bottleneck_layers,
            #     normalization=normalization,
            #     dropout_rate=dropout_rate,
            #     min_res_dropout=min_res_dropout,
            #     init_as_zero=init_as_zero,
            #     type="decoder",
            # )
            # for legacy
            bottleneck_layers = []
            self.conv_inp = nn.Conv1d(
                bottleneck_channels,
                bottleneck_base_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )
            for _ in range(num_bottleneck_layers):
                bottleneck_layers.append(
                    ResBlock(
                        bottleneck_base_channels,
                        bottleneck_base_channels,
                        cond_channels,
                        normalize=normalization,
                        use_2d=False,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=init_as_zero,
                        padding_mode=padding_mode,
                    )
                )
            # self.bottleneck_layers = nn.ModuleList(bottleneck_layers)
            self.bottleneck_layers = nn.Sequential(
                *bottleneck_layers
            )  # may be more efficient

            self.conv_out_bottleneck = nn.Conv1d(
                bottleneck_base_channels,
                input_channels * self.freq_dim,
                kernel_size=1,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            )

        multipliers_list_upsampling = (
            list(reversed(multipliers_list))[1:] + list(reversed(multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(freq_downsample_list))

        up_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(layers_list), multipliers_list_upsampling)
        ):
            for _ in range(num_layers):
                up_layers.append(
                    ResBlock(
                        input_channels,
                        input_channels,
                        normalize=normalization,
                        attention=attention_list[::-1][i] == 1,
                        heads=4,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=init_as_zero,
                        padding_mode=padding_mode,
                    )
                )
            if i != len(layers_list) - 1:
                output_channels = base_channels * multiplier
                if freq_upsample_list[i] == 1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(
                        UpsampleConv(
                            input_channels,
                            output_channels,
                            use_2d=True,
                            # normalize=normalization, # todo check this
                            padding_mode=padding_mode,
                        )
                    )
                input_channels = output_channels

        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, x):
        if self.legacy_bottleneck:
            x = self.conv_inp(x)
            for layer in self.bottleneck_layers:
                x = layer(x)
            x = self.conv_out_bottleneck(x)
            x_ls = torch.chunk(x.unsqueeze(-2), self.freq_dim, -3)
            x = torch.cat(x_ls, -2)
        else:
            x = self.bottleneck_layers(x)

        #

        k = 0
        pyramid_list = []
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for _ in range(num_layers):
                x = self.up_layers[k](x)
                k += 1
            pyramid_list.append(x)
            if i != len(self.layers_list) - 1:
                x = self.up_layers[k](x)
                k += 1

        return pyramid_list[::-1]


class UNet(nn.Module):
    def __init__(
        self,
        base_channels=64,
        layers_list=[2, 2, 2, 2, 2],
        multipliers_list=[1, 2, 4, 4, 4],
        attention_list=[0, 0, 1, 1, 1],
        freq_downsample_list=[1, 0, 0, 0],
        layers_list_encoder=[1, 1, 1, 1, 1],
        attention_list_encoder=[0, 0, 1, 1, 1],
        bottleneck_base_channels=512,
        bottleneck_channels=64,
        num_bottleneck_layers=4,
        bottleneck_layers=None,
        hop=256,
        data_channels=2,
        cond_channels=256,
        heads=4,
        use_fourier=False,
        fourier_scale=0.2,
        normalization=True,
        dropout_rate=0.0,
        min_res_dropout=16,
        pre_normalize_downsampling_encoder=True,
        pre_normalize_2d_to_1d=True,
        frequency_scaling=True,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        init_as_zero=True,
        activation_bottleneck=nn.Tanh(),
        padding_mode="zeros",
    ):
        super().__init__()

        self.layers_list = list(layers_list)
        self.multipliers_list = list(multipliers_list)
        self.attention_list = list(attention_list)
        self.freq_downsample_list = list(freq_downsample_list)
        self.cond_channels = cond_channels
        self.hop = hop
        self.frequency_scaling = frequency_scaling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        input_channels = base_channels * multipliers_list[0]
        self.data_channels = data_channels
        self.use_fourier = use_fourier
        self.fourier_scale = fourier_scale
        self.pre_normalize_downsampling_encoder = pre_normalize_downsampling_encoder
        self.pre_normalize_2d_to_1d = pre_normalize_2d_to_1d
        self.init_as_zero = init_as_zero
        Conv = nn.Conv2d

        # Encoder and Decoder
        self.encoder = Encoder(
            layers_list=layers_list_encoder,
            attention_list=attention_list_encoder,
            multipliers_list=multipliers_list,
            base_channels=base_channels,
            data_channels=data_channels,
            bottleneck_base_channels=bottleneck_base_channels,
            num_bottleneck_layers=num_bottleneck_layers,
            bottleneck_layers=bottleneck_layers,
            bottleneck_channels=bottleneck_channels,
            hop=hop,
            freq_downsample_list=freq_downsample_list,
            normalization=normalization,
            pre_normalize_downsampling_encoder=pre_normalize_downsampling_encoder,
            pre_normalize_2d_to_1d=pre_normalize_2d_to_1d,
            frequency_scaling=frequency_scaling,
            dropout_rate=dropout_rate,
            min_res_dropout=min_res_dropout,
            init_as_zero=init_as_zero,
            activation_bottleneck=activation_bottleneck,
            padding_mode=padding_mode,
        )

        self.decoder = Decoder(
            layers_list=layers_list_encoder,
            attention_list=attention_list_encoder,
            multipliers_list=multipliers_list,
            base_channels=base_channels,
            bottleneck_base_channels=bottleneck_base_channels,
            bottleneck_channels=bottleneck_channels,
            cond_channels=cond_channels,
            hop=hop,
            freq_downsample_list=freq_downsample_list,
            num_bottleneck_layers=num_bottleneck_layers,
            bottleneck_layers=bottleneck_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            min_res_dropout=min_res_dropout,
            init_as_zero=init_as_zero,
            padding_mode=padding_mode,
        )

        # Embeddings for noise conditioning
        if use_fourier:
            self.emb = GaussianFourierProjection(
                embedding_size=cond_channels, scale=fourier_scale
            )
        else:
            self.emb = PositionalEmbedding(embedding_size=cond_channels)

        self.emb_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
        )

        self.scale_inp = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(cond_channels, hop * 2), init_as_zero),
        )

        self.scale_out = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            zero_init(nn.Linear(cond_channels, hop * 2), init_as_zero),
        )

        # Input convolution
        self.conv_inp = Conv(
            data_channels,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )

        # Downsampling path
        down_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(layers_list, multipliers_list)
        ):
            output_channels = base_channels * multiplier
            for _ in range(num_layers):
                down_layers.append(
                    Conv(
                        output_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                down_layers.append(
                    ResBlock(
                        output_channels,
                        output_channels,
                        cond_channels,
                        normalize=normalization,
                        attention=attention_list[i] == 1,
                        heads=heads,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=True,
                        padding_mode=padding_mode,
                    )
                )
            input_channels = output_channels
            if i != len(layers_list) - 1:
                output_channels = base_channels * multipliers_list[i + 1]
                if freq_downsample_list[i] == 1:
                    down_layers.append(
                        DownsampleFreqConv(input_channels, output_channels)
                    )
                else:
                    down_layers.append(
                        DownsampleConv(
                            input_channels,
                            output_channels,
                            use_2d=True,
                            # normalize=normalization,  todo check this
                            padding_mode=padding_mode,
                        )
                    )

        self.down_layers = nn.ModuleList(down_layers)

        # Upsampling path
        multipliers_list_upsampling = (
            list(reversed(multipliers_list))[1:] + list(reversed(multipliers_list))[:1]
        )
        freq_upsample_list = list(reversed(freq_downsample_list))

        up_layers = []
        for i, (num_layers, multiplier) in enumerate(
            zip(reversed(layers_list), multipliers_list_upsampling)
        ):
            for _ in range(num_layers):
                up_layers.append(
                    Conv(
                        input_channels,
                        input_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                up_layers.append(
                    ResBlock(
                        input_channels,
                        input_channels,
                        cond_channels,
                        normalize=normalization,
                        attention=attention_list[::-1][i] == 1,
                        heads=heads,
                        use_2d=True,
                        dropout_rate=dropout_rate,
                        min_res_dropout=min_res_dropout,
                        init_as_zero=True,
                        padding_mode=padding_mode,
                    )
                )
            if i != len(layers_list) - 1:
                output_channels = base_channels * multiplier
                if freq_upsample_list[i] == 1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(
                        UpsampleConv(
                            input_channels,
                            output_channels,
                            use_2d=True,
                            padding_mode=padding_mode,
                        )
                    )
                input_channels = output_channels

        self.up_layers = nn.ModuleList(up_layers)

        self.conv_decoded = Conv(
            input_channels, input_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm_out = nn.GroupNorm(min(input_channels // 4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = zero_init(
            Conv(
                input_channels,
                data_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
            ),
            init_as_zero,
        )

    def forward_generator(self, latents, x, sigma=None, pyramid_latents=None):
        if sigma is None:
            sigma = self.sigma_max

        inp = x
        sigma = torch.ones((x.shape[0],), dtype=torch.float32, device=x.device) * sigma
        sigma_log = torch.log(sigma) / 4.0
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self.emb_proj(emb_sigma_log)

        scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)
        scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0], 1, -1, 1)

        c_skip, c_out, c_in = get_c(sigma, self.sigma_min, self.sigma_data)

        x = c_in * x

        if latents.shape == x.shape:
            latents = self.encoder(latents)

        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)

        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_inp) * x

        skip_list = []
        k = 0
        for i, num_layers in enumerate(self.layers_list):
            for _ in range(num_layers):
                d = self.down_layers[k](pyramid_latents[i])
                k += 1
                x = (x + d) / np.sqrt(2.0)
                x = self.down_layers[k](x, time_emb)
                skip_list.append(x)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.down_layers[k](x)
                k += 1

        k = 0
        for i, num_layers in enumerate(reversed(self.layers_list)):
            for _ in range(num_layers):
                d = self.up_layers[k](pyramid_latents[-i - 1])
                k += 1
                x = (x + skip_list.pop() + d) / np.sqrt(3.0)
                x = self.up_layers[k](x, time_emb)
                k += 1
            if i != len(self.layers_list) - 1:
                x = self.up_layers[k](x)
                k += 1

        d = self.conv_decoded(pyramid_latents[0])
        x = (x + d) / np.sqrt(2.0)

        x = self.norm_out(x)
        x = self.activation_out(x)
        if self.frequency_scaling:
            x = (1.0 + scale_w_out) * x
        x = self.conv_out(x)

        out = c_skip * inp + c_out * x

        return out

    def forward(
        self, data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas
    ):
        latents = self.encoder(data_encoder)
        pyramid_latents = self.decoder(latents)
        fdata = self.forward_generator(
            latents, noisy_samples, sigmas_step, pyramid_latents
        ).detach()
        fdata_plus_one = self.forward_generator(
            latents, noisy_samples_plus_one, sigmas, pyramid_latents
        )
        return fdata, fdata_plus_one
