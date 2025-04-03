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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import librosa
import torchaudio

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from conditionalDiffwave.dataset import from_path
from conditionalDiffwave.model import DiffWave
from conditionalDiffwave.params import AttrDict


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class DiffWaveLearner:
    def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.amp.autocast('cuda', enabled=kwargs.get('fp16', False))
        self.scaler = torch.amp.GradScaler('cuda', enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None

        # 불확실성 기반 weighting을 위한 학습 가능한 파라미터 (초기값 0으로 설정)
        self.log_sigma_original = nn.Parameter(torch.zeros(1, device=next(model.parameters()).device))
        self.log_sigma_custom = nn.Parameter(torch.zeros(1, device=next(model.parameters()).device))
        # 기존 모델 파라미터와 함께 optimizer에 추가합니다.
        self.optimizer.add_param_group({'params': [self.log_sigma_original, self.log_sigma_custom]})

    def compute_custom_loss(self, predicted, target, n_mels, sr, low_weight=0.9, mid_weight=0.1):
        # n_mels와 sr를 정수형으로 변환
        n_mels = int(n_mels) if not isinstance(n_mels, torch.Tensor) else int(n_mels.item())
        sr = int(sr) if not isinstance(sr, torch.Tensor) else int(sr.item())

        # 1️⃣ 파워 스펙트로그램이 0이 되지 않도록 작은 값으로 클리핑
        predicted = predicted.clamp(min=1e-10)
        target = target.clamp(min=1e-10)

        # 2️⃣ Log-Mel 변환 (power-to-dB, ref=1.0로 고정)
        predicted_log = librosa.power_to_db(predicted.detach().cpu().numpy(), ref=1.0)
        target_log = librosa.power_to_db(target.detach().cpu().numpy(), ref=1.0)

        # 3️⃣ 결과를 Tensor로 변환
        predicted_log = torch.tensor(predicted_log, device=predicted.device)
        target_log = torch.tensor(target_log, device=target.device)

        # 배치 차원이 없으면 추가 (2D 텐서라면)
        if predicted_log.dim() == 2:
            predicted_log = predicted_log.unsqueeze(0)
        if target_log.dim() == 2:
            target_log = target_log.unsqueeze(0)

        # 4️⃣ 멜 주파수 계산 및 인덱스 선택
        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)
        low_bins = np.where((mel_frequencies >= 20) & (mel_frequencies < 100))[0]
        mid_bins = np.where((mel_frequencies >= 250) & (mel_frequencies <= 2000))[0]

        # 안전장치: 텐서의 두 번째 차원 범위 내에 있는 인덱스만 사용
        max_index = predicted_log.size(1)
        low_bins = low_bins[low_bins < max_index]
        mid_bins = mid_bins[mid_bins < max_index]

        # low, mid 주파수 대역 선택
        predicted_low = predicted_log[:, low_bins, :]
        target_low = target_log[:, low_bins, :]
        predicted_mid = predicted_log[:, mid_bins, :]
        target_mid = target_log[:, mid_bins, :]

        # 5️⃣ 각 대역에 대해 L1 손실 계산
        low_loss = F.l1_loss(predicted_low, target_low)
        mid_loss = F.l1_loss(predicted_mid, target_mid)

        # 6️⃣ 너무 작은 range_db를 방지하여 정규화
        range_db = max(80.0, torch.max(low_loss, mid_loss).item() * 2)  # 적절한 범위 조절
        scaled_low_loss = low_loss / range_db
        scaled_mid_loss = mid_loss / range_db

        # 7️⃣ 가중치 적용하여 최종 손실 계산
        weighted_loss = low_weight * scaled_low_loss + mid_weight * scaled_mid_loss

        return weighted_loss

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt', weights_only=True)
            print(f"{self.model_dir}/{filename}.pt Loaded...")
            self.load_state_dict(checkpoint)
            print(f"[DEBUG] Restored checkpoint step: {self.step}")
            return True
        except FileNotFoundError:
            print(f"[WARNING] Checkpoint {filename}.pt not found.")
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for features in tqdm(self.dataset,
                                 desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
                if max_steps is not None and self.step >= max_steps:
                    return

                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss, original_loss, custom_loss, weight_original, weight_custom = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                if self.is_master:
                    if self.step % 50 == 0:
                        # 모든 인자를 전달하도록 수정
                        self._write_summary(self.step, features, loss, original_loss, custom_loss, weight_original,
                                            weight_custom)

                    if self.step % len(self.dataset) == 0:
                        self.save_to_checkpoint()

                self.step += 1

    def train_step(self, features):
        # 모델 파라미터의 기울기 초기화
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']  # 입력 waveform: (B, T)
        target = features['target']  # 타겟 waveform: (B, T)

        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            # Diffusion step t 샘플링
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio)

            # Noisy Audio 생성
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            # target을 conditioner로 사용 (upsampler 제거)
            conditioner = target.unsqueeze(1)  # (B, 1, T)

            # 모델 예측 수행 -> waveform: (B, 1, T)
            predicted = self.model(noisy_audio, t, conditioner)

            # 기존 L1 Loss 계산 (waveform 비교)
            original_loss = self.loss_fn(target, predicted.squeeze(1))

            # 멜 스펙트로그램 변환: predicted와 target 모두 (B, n_mels, time_mel) 형태로 변환
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.params.sample_rate,
                n_mels=self.params.n_mels
            ).to(device)

            log_mel_transform = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)

            # predicted는 (B, 1, T) -> squeeze해서 (B, T)
            predicted_mel = log_mel_transform(predicted.squeeze(1))
            target_mel = log_mel_transform(target)

            # custom loss 계산 (멜 스펙트로그램 비교)
            custom_loss = self.compute_custom_loss(predicted_mel, target_mel, self.params.n_mels,
                                                   self.params.sample_rate)

            # 불확실성 기반 가중치 적용 (두 가중치 합이 1이 되도록 softmax 사용)
            log_sigmas = torch.stack([-torch.exp(self.log_sigma_original), -torch.exp(self.log_sigma_custom)])
            weights = torch.softmax(log_sigmas, dim=0)

            weight_original, weight_custom = weights[0], weights[1]

            loss = (weight_original * original_loss + torch.abs(self.log_sigma_original) +
                    weight_custom * custom_loss + torch.abs(self.log_sigma_custom))


        # 역전파 및 옵티마이저 업데이트
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 추가적으로 custom loss 및 가중치 값 반환
        return loss, original_loss.item(), custom_loss.item(), weight_original.item(), weight_custom.item()

    def _write_summary(self, step, features, loss, original_loss, custom_loss, weight_original, weight_custom):
        try:
            if self.summary_writer is None:
                self.summary_writer = SummaryWriter(
                    "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/event_logs_plus/", purge_step=step)

            writer = self.summary_writer

            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/original_loss', original_loss, step)
            writer.add_scalar('train/custom_loss', custom_loss, step)
            writer.add_scalar('train/grad_norm', self.grad_norm, step)
            writer.add_scalar('train/weight_original', weight_original, step)
            writer.add_scalar('train/weight_custom', weight_custom, step)

            if 'audio' in features:
                writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)

            if not self.params.unconditional and 'spectrogram' in features:
                writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)

            writer.flush()

            loss_log_file = os.path.join("/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/event_logs_plus/",
                                         'losslog.csv')
            if not os.path.exists(loss_log_file):
                with open(loss_log_file, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['step', 'loss', 'original_loss', 'custom_loss', 'grad_norm', 'weight_original',
                                         'weight_custom'])

            with open(loss_log_file, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(
                    [step, loss, original_loss, custom_loss, self.grad_norm, weight_original, weight_custom])

        except Exception as e:
            print(f"[ERROR] Failed to write summary at step {step}: {e}")


def _train_impl(replica_id, model, dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)


def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    print(f"[DEBUG] Starting training on device: {device}")
    while True:
        for i, features in enumerate(tqdm(self.dataset,
                                          desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset):
            print(f"[DEBUG] Processing batch {i} at global step {self.step}")
            if max_steps is not None and self.step >= max_steps:
                return

            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            loss = self.train_step(features)
            print(f"[DEBUG] Loss at step {self.step}: {loss.item()}")

            if torch.isnan(loss).any():
                raise RuntimeError(f'Detected NaN loss at step {self.step}.')

            if self.is_master:
                if self.step % 50 == 0:
                    self._write_summary(self.step, features, loss)

                if self.step % len(self.dataset) == 0:
                    print(f"[DEBUG] Saving checkpoint at step {self.step}")
                    self.save_to_checkpoint()

            self.step += 1



def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    dataset = from_path(args.input_dirs, args.target_dirs, params, is_distributed=True)

    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, args, params)
