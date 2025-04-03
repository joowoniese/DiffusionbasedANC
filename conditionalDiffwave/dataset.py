import numpy as np
import os
import torch
import torchaudio
import torch.nn.functional as F

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import Resample


class PairedAudioDataset(Dataset):
    """
    기존 ConditionalDataset을 수정하여 input과 target 오디오 쌍을 로드하는 데이터셋.
    """

    def __init__(self, input_dirs, target_dirs, sample_rate=22050):
        super().__init__()
        self.input_files = []
        self.target_files = []

        # 입력 오디오와 타겟 오디오 파일 리스트 가져오기
        for input_dir, target_dir in zip(input_dirs, target_dirs):
            input_wavs = sorted(glob(f'{input_dir}/**/*.wav', recursive=True))
            target_wavs = sorted(glob(f'{target_dir}/**/*.wav', recursive=True))

            if len(input_wavs) != len(target_wavs):
                raise ValueError(f"[ERROR] Input({len(input_wavs)})과 Target({len(target_wavs)}) 파일 개수가 다릅니다!")

            self.input_files.extend(input_wavs)
            self.target_files.extend(target_wavs)

        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]

        # 오디오 로드
        input_audio, sr = torchaudio.load(input_path)
        target_audio, sr_target = torchaudio.load(target_path)

        # 샘플링 레이트 변환
        if sr != self.sample_rate:
            resample = Resample(orig_freq=sr, new_freq=self.sample_rate)
            input_audio = resample(input_audio)
        if sr_target != self.sample_rate:
            resample = Resample(orig_freq=sr_target, new_freq=self.sample_rate)
            target_audio = resample(target_audio)

        # 다중 채널 -> 모노 변환
        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0)
        if target_audio.shape[0] > 1:
            target_audio = target_audio.mean(dim=0)

        # 길이 맞추기 (110250 samples = 5초 @ 22050Hz 기준)
        target_length = 110250
        if input_audio.shape[-1] < target_length:
            input_audio = F.pad(input_audio, (0, target_length - input_audio.shape[-1]), mode='constant', value=0)
        else:
            input_audio = input_audio[:target_length]

        if target_audio.shape[-1] < target_length:
            target_audio = F.pad(target_audio, (0, target_length - target_audio.shape[-1]), mode='constant', value=0)
        else:
            target_audio = target_audio[:target_length]

        return {
            'audio': input_audio.numpy(),  # Noisy input
            'target': target_audio.numpy(),  # Clean target
            'file_name': os.path.basename(input_path)
        }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        """
        기존 spectrogram을 사용하는 로직을 제거하고 target audio를 condition으로 사용하도록 변경.
        """
        filtered_minibatch = []
        for record in minibatch:
            if len(record['audio']) < self.params.audio_len:
                continue
            if len(record['target']) < self.params.audio_len:
                continue

            filtered_minibatch.append(record)

        if not filtered_minibatch:
            # 빈 데이터를 방지하기 위해 0으로 채운 텐서 반환
            audio = np.zeros((self.params.batch_size, self.params.audio_len))
            target = np.zeros((self.params.batch_size, self.params.audio_len))
            return {
                'audio': torch.from_numpy(audio),
                'target': torch.from_numpy(target),
                'file_name': [],
            }

        audio = np.stack([record['audio'] for record in filtered_minibatch])
        target = np.stack([record['target'] for record in filtered_minibatch])
        file_names = [record['file_name'] for record in filtered_minibatch]

        # print(f"[DEBUG] Batch file paths: {file_names}")

        return {
            'audio': torch.from_numpy(audio),
            'target': torch.from_numpy(target),
            'file_name': file_names,
        }


def from_path(input_dirs, target_dirs, params, is_distributed=False):
    dataset = PairedAudioDataset(input_dirs, target_dirs, sample_rate=params.sample_rate)
    # print(f"[DEBUG] Loading input: {input_dirs}, target: {target_dirs}")

    print("[DEBUG] Checking DataLoader sample batches...")
    # for batch_idx, batch in enumerate(dataset):
    #     print(f"[DEBUG] Batch {batch_idx}: {batch['file_name']}")
    #     if batch_idx >= 5:
    #         break  # 5개 배치만 확인

    # 🚨 데이터셋 크기 확인
    print(f"[DEBUG] Loaded dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("[ERROR] Dataset is empty! Check if the directories contain valid .wav files.")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=True if not is_distributed else False,  # ✅ 수정
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True
    )

