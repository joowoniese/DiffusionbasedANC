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
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import Resample


class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            # .npy 확장자만 가져오기
            self.filenames += glob(f'{path}/**/*.spec.npy', recursive=True)
        # print(f"[DEBUG] Found {len(self.filenames)} .npy files for ConditionalDataset.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        spec_filename = self.filenames[idx]
        try:
            # .npy 파일 불러오기
            spectrogram = np.load(spec_filename)
            # print(f"[DEBUG] Loaded spectrogram: {spec_filename}, shape: {spectrogram.shape}")

            # .wav 파일에서 오디오 데이터 로드
            audio_filename = spec_filename.replace('.spec.npy', '')  # 확장자 명확히 지정
            if not os.path.exists(audio_filename):
                raise FileNotFoundError(f"Audio file not found: {audio_filename}")

            # 오디오 로드
            audio, sr = torchaudio.load(audio_filename)
            target_sr = 22050
            if sr != target_sr:
                resample_transform = Resample(orig_freq=sr, new_freq=target_sr)
                audio = resample_transform(audio)
                sr = target_sr

            # 다중 채널 처리
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)  # 모노 변환

            # 길이 보정
            target_length = 110250
            if audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]), mode='constant', value=0)
            else:
                audio = audio[:target_length]

            return {
                'spectrogram': spectrogram.T,
                'audio': audio.numpy(),
                'file_name': os.path.basename(audio_filename)
            }

        except Exception as e:
            print(f"[ERROR] Failed to process file {spec_filename}: {e}")
            return None


class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)
    # print(f"[DEBUG] Found {len(self.filenames)} audio files for UnconditionalDataset.")

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
      audio_filename = self.filenames[idx]
      try:
          signal, _ = torchaudio.load(audio_filename)
          return {
              'audio': signal[0],
              'spectrogram': None,
              'file_name': os.path.basename(audio_filename)  # file_name 추가
          }
      except Exception as e:
          # print(f"[ERROR] Failed to load audio for {audio_filename}: {e}")
          raise


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        filtered_minibatch = []
        for record in minibatch:
            try:
                # 오디오 길이 확인 및 디버깅
                # print(f"[DEBUG] Checking audio length: {len(record['audio'])}, Expected: {self.params.audio_len}")
                if len(record['audio']) < self.params.audio_len:
                    # print(f"[WARNING] Skipping record due to insufficient audio length: {len(record['audio'])}")
                    continue
                if len(record['spectrogram']) < self.params.crop_mel_frames:
                    # print(
                        # f"[DEBUG] Skipping record due to insufficient spectrogram frames: {len(record['spectrogram'])}")
                    continue

                # 오디오 및 스펙트로그램 크기 조정
                start = random.randint(0, len(record['spectrogram']) - self.params.crop_mel_frames)
                end = start + self.params.crop_mel_frames
                record['spectrogram'] = record['spectrogram'][start:end].T

                start *= samples_per_frame
                end *= samples_per_frame
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(record['audio'], (0, self.params.audio_len - len(record['audio'])),
                                         mode='constant')

                filtered_minibatch.append(record)
            except Exception as e:
                print(f"[ERROR] Error during collation: {e}")

        # print(f"[INFO] Total records before filtering: {len(minibatch)}")
        # print(f"[INFO] Valid records after filtering: {len(filtered_minibatch)}")

        if not filtered_minibatch:
            # print("[WARNING] No valid records found. Returning zero-padded data.")
            audio = np.zeros((self.params.batch_size, self.params.audio_len))
            spectrogram = np.zeros((self.params.batch_size, self.params.n_mels, self.params.crop_mel_frames))
            return {
                'audio': torch.from_numpy(audio),
                'spectrogram': torch.from_numpy(spectrogram),
                'file_name': [],
            }

        audio = np.stack([record['audio'] for record in filtered_minibatch])
        spectrogram = np.stack([record['spectrogram'] for record in filtered_minibatch])
        file_names = [record['file_name'] for record in filtered_minibatch]

        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': torch.from_numpy(spectrogram),
            'file_name': file_names,
        }

    def collate_gtzan(self, minibatch):
        ldata = []
        mean_audio_len = self.params.audio_len
        for data in minibatch:
          try:
              if data[0].shape[-1] < mean_audio_len:
                  data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
              elif data[0].shape[-1] > mean_audio_len:
                  start = random.randint(0, data[0].shape[-1] - mean_audio_len)
                  end = start + mean_audio_len
                  data_audio = data[0][:, start:end]
              else:
                  data_audio = data[0]
              ldata.append(data_audio)
          except Exception as e:
              print(f"[ERROR] Error during GTZAN collation: {e}")

        if not ldata:
            raise ValueError("[ERROR] No valid audio data in minibatch for GTZAN.")

        audio = torch.cat(ldata, dim=0)
        return {
              'audio': audio,
              'spectrogram': None,
        }

def from_path(data_dirs, params, is_distributed=False):
    if params.unconditional:
        dataset = UnconditionalDataset(data_dirs)
    else:
        dataset = ConditionalDataset(data_dirs)

    # 데이터셋 샘플 디버깅
    # print("[DEBUG] Checking dataset samples:")
    for idx in range(min(len(dataset), 5)):  # 첫 5개의 샘플 확인
        sample = dataset[idx]
        # print(f"[DEBUG] Sample {idx}: audio length = {len(sample['audio'])}, file_name = {sample['file_name']}")

    # print(f"[DEBUG] Creating DataLoader with batch size {params.batch_size}.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True)


def from_gtzan(params, is_distributed=False):
  # print("[DEBUG] Loading GTZAN dataset.")
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      # num_workers=os.cpu_count(),
      num_workers=2,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
