# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    super().__init__()
    self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = nn.Linear(512, residual_channels)

    if not uncond:
      self.conditioner_projection = nn.Conv1d(n_mels, 2 * residual_channels, 1)
      self.target_projection = nn.Conv1d(n_mels, 2 * residual_channels, 1)  # 🔹 target condition 추가
    else:
      self.conditioner_projection = None
      self.target_projection = None  # 🔹 target도 없음

    self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, target=None):
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step

    if self.conditioner_projection is not None and target is not None:
      target = self.target_projection(target)  # 🔹 target을 projection
      y = self.dilated_conv(y) + target
    else:
      y = self.dilated_conv(y)

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

    # 🔹 target을 condition으로 사용하기 위해 target upsampler 추가
    self.target_upsampler = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    self.residual_layers = nn.ModuleList([
      ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                    uncond=params.unconditional)
      for i in range(params.residual_layers)
    ])

    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, target=None):
    # 🔹 target이 None이 아니고 다중 채널이면 1채널로 변환
    if target is not None:
        if target.ndim == 3 and target.shape[1] > 1:  # 다중 채널인 경우 평균 변환
            target = target.mean(dim=1, keepdim=True)  # (B, 1, T)

        target = target.unsqueeze(1)  # (B, 1, T) → (B, 1, 1, T) 변환
        target = self.target_upsampler(target)  # Upsample target
        target = F.leaky_relu(target, 0.4)
        target = torch.squeeze(target, 1)  # (B, 1, T)로 변환

    x = audio.unsqueeze(1)  # (B, 1, T)
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)

    skip = None
    for layer in self.residual_layers:
        x, skip_connection = layer(x, diffusion_step, target)
        skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x



