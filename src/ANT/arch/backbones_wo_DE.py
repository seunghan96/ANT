# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import nn

from .s4 import S4


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        device='cuda:2'
    ):
        super().__init__()
        self.layer = S4(
            d_model=d_model,
            d_state=128,
            bidirectional=True,
            dropout=dropout,
            transposed=True,
            postact=None,
            device=device
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = (
            nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        )
        self.device = device

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)
        # Dropout on the output of the layer
        z = self.dropout(z)
        # Residual connection
        x = z + x
        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


class S4Block(nn.Module):
    def __init__(self, d_model, dropout=0.0, expand=2, num_features=0, device='cuda:2'):
        super().__init__()
        self.s4block = S4Layer(d_model, dropout=dropout, device=device)
        self.time_linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.out_linear2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1
        )
        self.feature_encoder = nn.Conv1d(num_features, d_model, kernel_size=1)

    def forward(self, x, features=None):
        out, _ = self.s4block(x)
        if features is not None:
            out = out + self.feature_encoder(features)
        out = self.tanh(out) * self.sigm(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


def Conv1dKaiming(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        step_emb,
        num_residual_blocks,
        num_features,
        residual_block="s4",
        dropout=0.0,
        init_skip=True,
        device='cuda:2',
    ):
        super().__init__()
        if residual_block == "s4":
            residual_block = S4Block
        else:
            raise ValueError(f"Unknown residual block {residual_block}")
        self.input_init = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        residual_blocks = []
        for i in range(num_residual_blocks):
            residual_blocks.append(
                residual_block(
                    hidden_dim, num_features=num_features, dropout=dropout,device=device
                )
            )
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.init_skip = init_skip
        self.device = device

    def forward(self, input, t, features=None):
        x = self.input_init(input)  # B, L ,C
        x = x.transpose(-1, -2)
        if features is not None:
            features = features.transpose(-1, -2)
        skips = []
        for layer in self.residual_blocks:
            x, skip = layer(x, features)
            skips.append(skip)

        skip = torch.stack(skips).sum(0)
        skip = skip.transpose(-1, -2)
        out = self.out_linear(skip)
        if self.init_skip:
            out = out + input
        return out
