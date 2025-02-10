import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from diffwave.params import params


def transform(filename):
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)

    # 샘플 레이트 변환
    if sr != params.sample_rate:
        # print(f"[INFO] Resampling {filename} from {sr} Hz to {params.sample_rate} Hz")
        resampler = TT.Resample(orig_freq=sr, new_freq=params.sample_rate)
        audio = resampler(audio)
        sr = params.sample_rate

    # MelSpectrogram 매개변수 설정
    mel_args = {
        'sample_rate': sr,
        'win_length': params.hop_samples * 4,
        'hop_length': params.hop_samples,
        'n_fft': params.n_fft,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': params.n_mels,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    # MelSpectrogram 계산 및 저장
    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        output_path = f'{filename}.spec.npy'
        np.save(output_path, spectrogram.cpu().numpy())
        # print(f"[INFO] Processed and saved spectrogram to {output_path}")


def main(args):
    filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
    # print(f"[INFO] Found {len(filenames)} audio files in {args.dir}")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('dir',
        help='directory containing .wav files for training')
    main(parser.parse_args())
