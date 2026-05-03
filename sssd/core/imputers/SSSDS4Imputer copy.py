import math

import torch
import torch.nn as nn

from sssd.core.layers.s4.s4_layer import S4Layer
from sssd.core.utils import calc_diffusion_step_embedding
from sssd.utils.logger import setup_logger  # New
LOGGER = setup_logger()  # New

from autoFRK import AutoFRK  # New


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        residual_channels,
        skip_channels,
        diffusion_step_embed_dim_output,
        input_channels,
        s4_max_sequence_length,
        s4_state_dim,
        s4_dropout,
        s4_bidirectional,
        s4_use_layer_norm,
    ):
        super().__init__()
        self.residual_channels = residual_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_output, self.residual_channels)

        self.S41 = S4Layer(
            features=2 * self.residual_channels,
            lmax=s4_max_sequence_length,
            N=s4_state_dim,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_use_layer_norm,
        )

        self.conv_layer = Conv(
            self.residual_channels, 2 * self.residual_channels, kernel_size=3
        )

        self.S42 = S4Layer(
            features=2 * self.residual_channels,
            lmax=s4_max_sequence_length,
            N=s4_state_dim,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_use_layer_norm,
        )

        self.cond_conv = Conv(
            2 * input_channels, 2 * self.residual_channels, kernel_size=1
        )
        self.cond_bn = nn.BatchNorm1d(2 * self.residual_channels)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.res_conv = nn.utils.parametrizations.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.residual_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.residual_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)
        h = self.S41(h.permute(2, 0, 1)).permute(1, 2, 0)

        assert cond is not None
        cond = self.cond_conv(cond)
        cond = self.cond_bn(cond)
        h += cond

        h = self.S42(h.permute(2, 0, 1)).permute(1, 2, 0)

        out = torch.tanh(h[:, : self.residual_channels, :]) * torch.sigmoid(
            h[:, self.residual_channels :, :]
        )

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class ResidualGroup(nn.Module):
    def __init__(
        self,
        residual_channels,
        skip_channels,
        residual_layers,
        diffusion_step_embed_dim_input,
        diffusion_step_embed_dim_hidden,
        diffusion_step_embed_dim_output,
        input_channels,
        s4_max_sequence_length,
        s4_state_dim,
        s4_dropout,
        s4_bidirectional,
        s4_use_layer_norm,
        device="cuda",
    ):
        super(ResidualGroup, self).__init__()
        self.residual_layers = residual_layers
        self.diffusion_step_embed_dim_input = diffusion_step_embed_dim_input

        self.fc_t1 = nn.Linear(
            diffusion_step_embed_dim_input, diffusion_step_embed_dim_hidden
        )
        self.fc_t2 = nn.Linear(
            diffusion_step_embed_dim_hidden, diffusion_step_embed_dim_output
        )

        self.residual_blocks = nn.ModuleList()
        for n in range(self.residual_layers):
            self.residual_blocks.append(
                ResidualBlock(
                    residual_channels,
                    skip_channels,
                    diffusion_step_embed_dim_output=diffusion_step_embed_dim_output,
                    input_channels=input_channels,
                    s4_max_sequence_length=s4_max_sequence_length,
                    s4_state_dim=s4_state_dim,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_use_layer_norm=s4_use_layer_norm,
                )
            )

        self.device = device

    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_input, device=self.device
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.residual_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.residual_layers)


class SSSDS4Imputer(nn.Module):
    def __init__(
        self,
        input_channels,
        residual_channels,
        skip_channels,
        output_channels,
        residual_layers,
        diffusion_step_embed_dim_input,
        diffusion_step_embed_dim_hidden,
        diffusion_step_embed_dim_output,
        s4_max_sequence_length,
        s4_state_dim,
        s4_dropout,
        s4_bidirectional,
        s4_use_layer_norm,
        frk_channels=1,
        use_afrk=True,
        device="cuda",
    ):
        super().__init__()
        self.use_afrk = use_afrk

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
        )

        # 把 AFRK spatial prior 投影到 residual_channels，然後加到 latent 上
        if self.use_afrk:
            self.frk_proj = nn.Sequential(
                nn.Conv1d(frk_channels, residual_channels, kernel_size=1),
                nn.ReLU(),
            )

        self.residual_layer = ResidualGroup(
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            residual_layers=residual_layers,
            diffusion_step_embed_dim_input=diffusion_step_embed_dim_input,
            diffusion_step_embed_dim_hidden=diffusion_step_embed_dim_hidden,
            diffusion_step_embed_dim_output=diffusion_step_embed_dim_output,
            input_channels=input_channels,
            s4_max_sequence_length=s4_max_sequence_length,
            s4_state_dim=s4_state_dim,
            s4_dropout=s4_dropout,
            s4_bidirectional=s4_bidirectional,
            s4_use_layer_norm=s4_use_layer_norm,
            device=device,
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, output_channels),
        )

    def forward(self, input_data):
        noise, conditional, mask, diffusion_steps, frk_pred = input_data

        if frk_pred is None and self.use_afrk:
            LOGGER.warning("AFRK is enabled but frk_pred is None. Set use_afrk to False.")
            self.use_afrk = False

        # Handle mask and concatenate it to the conditional input
        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        # # AFRK prior
        # if self.use_afrk and frk_pred is not None:
        #     # 如果 AFRK 是空間先驗，通常不希望 missing 區域變成亂值
        #     conditional = frk_pred * mask.float()

        #     # 加到條件分支
        #     conditional = torch.cat([conditional, frk_pred], dim=1)

        # Forward pass through the network
        x = noise  # Ensure x is 3D (B, C, L)
        x = self.init_conv(x)

        # 再把 AFRK 作為 latent prior 加進去
        if self.use_afrk and frk_pred is not None:
            x = x + self.frk_proj(frk_pred)

        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y
