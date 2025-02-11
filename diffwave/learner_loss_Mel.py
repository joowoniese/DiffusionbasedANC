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

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffwave.dataset import from_path, from_gtzan
from diffwave.model import DiffWave
from diffwave.params import AttrDict
import torchaudio.transforms as T


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def weighted_melspectrogram_loss(predicted, target, n_mels, sr, low_weight=0.9, mid_weight=0.1):
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)
    low_bins = np.where((mel_frequencies >= 20) & (mel_frequencies < 100))[0]
    mid_bins = np.where((mel_frequencies >= 250) & (mel_frequencies <= 2000))[0]

    # print(f"[DEBUG] Predicted shape: {predicted.shape}, Target shape: {target.shape}")

    # 예상 크기 조정
    if target.size(0) != predicted.size(0):
        target = target.expand_as(predicted)

    if predicted.dim() != 3 or target.dim() != 3:
        raise ValueError(f"[ERROR] Unexpected tensor dimensions: Predicted {predicted.shape}, Target {target.shape}")

    try:
        predicted_low = predicted[:, low_bins, :]
        target_low = target[:, low_bins, :]
        predicted_mid = predicted[:, mid_bins, :]
        target_mid = target[:, mid_bins, :]
    except IndexError as e:
        raise IndexError(f"[ERROR] Indexing failed. Predicted: {predicted.shape}, Target: {target.shape}. Error: {e}")

    low_loss = F.mse_loss(predicted_low, target_low)
    mid_loss = F.mse_loss(predicted_mid, target_mid)

    weighted_loss = low_weight * low_loss + mid_weight * mid_loss

    # print(f"[DEBUG] Low loss: {low_loss.item()}, Mid loss: {mid_loss.item()}, Weighted loss: {weighted_loss.item()}")
    return weighted_loss



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
        # Loss 재정의
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None

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
            # print(f"[DEBUG] Restored checkpoint step: {self.step}")
            return True
        except FileNotFoundError:
            # print(f"[WARNING] Checkpoint {filename}.pt not found.")
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for features in tqdm(self.dataset,
                                 desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
                if max_steps is not None and self.step >= max_steps:
                    return
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:
                    if self.step % 50 == 0:
                        self._write_summary(self.step, features, loss)
                    if self.step % len(self.dataset) == 0:
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']
        spectrogram = features['spectrogram']
        file_name = features['file_name']

        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", file_name, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        # 빈 파일 이름 처리
        if not file_name:
            print("[WARNING] Empty file_name encountered in train_step.")
            return torch.tensor(0.0)  # 또는 다른 기본값 반환

        # print(f"[INFO] Processing files in train_step: {file_name}")

        # target_spec_dir = "/hdd_ext/hdd3/joowoniese/diffwave2/audio_data/clean_audio"
        # target_spec_path = os.path.join(target_spec_dir, file_name[0] if isinstance(file_name, list) else file_name)

        target_spec_dir = "/hdd_ext/hdd3/joowoniese/diffwave2/audio_data/clean_audio"

        # .spec.npy 확장자 붙이기
        if isinstance(file_name, list):
            file_name = [name + '.spec.npy' if not name.endswith('.spec.npy') else name for name in file_name]
            if not file_name:  # 리스트가 비어 있는 경우 처리
                # print("[WARNING] No valid file names after filtering.")
                return torch.tensor(0.0)  # 기본값 반환
        else:
            if not file_name.endswith('.spec.npy'):
                file_name += '.spec.npy'

        # 스펙트로그램 경로 생성
        target_spec_path = os.path.join(target_spec_dir, file_name[0] if isinstance(file_name, list) else file_name)

        # 경로 디버깅 출력
        # print(f"[DEBUG] Target spectrogram path: {target_spec_path}")

        # 파일 존재 여부 확인
        if not os.path.exists(target_spec_path):
            raise FileNotFoundError(f"Target Melspectrogram file not found: {target_spec_path}")

        target_spectrogram = torch.tensor(np.load(target_spec_path, allow_pickle=True), dtype=torch.float32).to(
            audio.device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [audio.shape[0]], device=audio.device)

            self.noise_level = self.noise_level.to(audio.device)

            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            predicted = self.model(noisy_audio, t, spectrogram)

            # Predicted를 MelSpectrogram으로 변환
            mel_transform = T.MelSpectrogram(
                sample_rate=self.params.sample_rate,
                n_mels=self.params.n_mels,
                hop_length=self.params.hop_samples
            ).to(predicted.device)

            predicted_spec = mel_transform(predicted.squeeze(1))

            # Target과 크기 맞추기
            target_frames = target_spectrogram.shape[-1]
            predicted_spec = predicted_spec[..., :target_frames]
            if predicted_spec.shape[-1] < target_frames:
                padding = target_frames - predicted_spec.shape[-1]
                predicted_spec = F.pad(predicted_spec, (0, padding))

            # Loss 계산
            loss = weighted_melspectrogram_loss(
                predicted_spec,
                target_spectrogram.unsqueeze(0),
                n_mels=self.params.n_mels,
                sr=self.params.sample_rate,
                low_weight=0.9,
                mid_weight=0.1
            )

            # print(f"[DEBUG] Loss after noise prediction: {loss.item()}")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _write_summary(self, step, features, loss):
        try:
            # SummaryWriter ??? ??
            if self.summary_writer is None:
                # print(f"[DEBUG] Initializing SummaryWriter at step {step}")
                self.summary_writer = SummaryWriter("/hdd_ext/hdd3/joowoniese/diffwave4/event_logs_mel/", purge_step=step)

            writer = self.summary_writer

            # ?? ? ?? ? ?? ??
            writer.add_scalar('train/loss', loss, step)
            # print(f"[DEBUG] train/loss recorded at step {step}: {loss:.6f}")  # Loss ??

            # ????? ?? ?? ? ??
            writer.add_scalar('train/grad_norm', self.grad_norm, step)
            # print(f"[DEBUG] train/grad_norm recorded at step {step}: {self.grad_norm:.6f}")  # Grad Norm ??

            # ??? ??
            if 'audio' in features:
                writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
                # print(f"[DEBUG] feature/audio recorded at step {step}")

            # ?????? ??
            if not self.params.unconditional and 'spectrogram' in features:
                writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
                # print(f"[DEBUG] feature/spectrogram recorded at step {step}")

            writer.flush()
            # print(f"[DEBUG] SummaryWriter flushed at step {step}")

            loss_log_file = os.path.join("/hdd_ext/hdd3/joowoniese/diffwave4/event_logs_mel/", 'loss_log_mel.csv')
            if not os.path.exists(loss_log_file):
                with open(loss_log_file, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['step', 'loss', 'grad_norm'])  # ?? ??

            with open(loss_log_file, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([step, loss.item(), self.grad_norm])
                # print(f"[DEBUG] Loss and grad_norm saved to CSV at step {step}")

        except Exception as e:
            print(f"[ERROR] Failed to write summary at step {step}: {e}")


def _train_impl(replica_id, model, dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)


def train(args, params):
    if args.data_dirs[0] == 'gtzan':
        dataset = from_gtzan(params)
    else:
        dataset = from_path(args.data_dirs, params)
    model = DiffWave(params).cuda()
    _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    if args.data_dirs[0] == 'gtzan':
        dataset = from_gtzan(params, is_distributed=True)
    else:
        dataset = from_path(args.data_dirs, params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, args, params)