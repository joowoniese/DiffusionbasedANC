import os
import torch
import torchaudio
import numpy as np
import librosa
import pandas as pd
import random
import matplotlib.pyplot as plt

# 실제 프로젝트 모듈 경로에 맞게 임포트하세요.
from conditionalDiffwave.model import DiffWave
from conditionalDiffwave.params import AttrDict, params as default_params
from conditionalDiffwave.learner_condition_withloss_norm import DiffWaveLearner

# === 1. 파일 목록 및 매핑 ===

# test와 target 오디오가 위치한 디렉터리 경로
test_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/anc_output_mixed/"
target_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/ideal_output_mixed/"

# 두 디렉터리의 파일 목록 불러오기
test_files = os.listdir(test_dir)
target_files = os.listdir(target_dir)


# 파일명의 공통 접두어 추출 함수
def get_common_base(filename, marker):
    if filename.endswith(marker):
        return filename.replace(marker, '')
    return None


# test 파일 매핑 (예: '_anc_mixed.wav')
test_dict = {}
for f in test_files:
    base = get_common_base(f, '_anc_mixed.wav')
    if base is not None:
        test_dict[base] = os.path.join(test_dir, f)

# target 파일 매핑 (예: '_anti_mixed.wav')
target_dict = {}
for f in target_files:
    base = get_common_base(f, '_anti_mixed.wav')
    if base is not None:
        target_dict[base] = os.path.join(target_dir, f)

# 공통 접두어를 가진 파일 쌍 선택
common_bases = set(test_dict.keys()) & set(target_dict.keys())
print(f"총 {len(common_bases)} 쌍의 파일이 발견되었습니다.")

# === 2. 모델 및 파라미터 초기화 ===

# 제공된 파라미터 설정 (예: audio_len = 22050*5)
params = AttrDict(
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    audio_len=22050 * 5
)

# 모델 및 옵티마이저 초기화
model = DiffWave(params)
model.eval()  # 평가 모드
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

# 체크포인트 저장 디렉터리
model_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/model_logs_norm/"
os.makedirs(model_dir, exist_ok=True)

# DiffWaveLearner 인스턴스 생성 (테스트 목적으로 weight 불러오기)
learner = DiffWaveLearner(model_dir, model, None, optimizer, params, fp16=False)

# 저장된 checkpoint 불러오기
if learner.restore_from_checkpoint('weights'):
    print("Checkpoint 불러오기 성공!")
else:
    print("Checkpoint 불러오기 실패: 파일 경로와 이름을 확인하세요.")

# 모델이 위치한 device 확인
device = next(model.parameters()).device

# 멜 스펙트로그램 transform 생성 (power spectrogram 생성)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=params.sample_rate,
    n_mels=params.n_mels,
    n_fft=params.n_fft,
    hop_length=params.hop_samples
).to(device)

desired_length = params.audio_len  # 예: 110250 샘플 (5초)

# === 3. 랜덤으로 5개 쌍 선택 후 시각화 (Input, Target, Output) ===

# 쌍 데이터 중 랜덤으로 5개 선택
visualization_bases = random.sample(sorted(common_bases), min(5, len(common_bases)))

for base in visualization_bases:
    test_path = test_dict[base]
    target_path = target_dict[base]

    # 오디오 파일 로드 (waveform, sample_rate)
    input_waveform, sr_input = torchaudio.load(test_path)
    target_waveform, sr_target = torchaudio.load(target_path)

    if sr_input != sr_target:
        print(f"[WARNING] {base}: Sample rate 불일치 (test: {sr_input}, target: {sr_target}). 건너뜁니다.")
        continue

    # 스테레오인 경우 첫 채널만 사용
    if input_waveform.shape[0] > 1:
        input_waveform = input_waveform[0:1, :]
    if target_waveform.shape[0] > 1:
        target_waveform = target_waveform[0:1, :]

    # 길이를 desired_length로 맞추기
    if input_waveform.size(1) > desired_length:
        input_waveform = input_waveform[:, :desired_length]
    if target_waveform.size(1) > desired_length:
        target_waveform = target_waveform[:, :desired_length]

    # 모델 입력을 위한 전처리: (채널, T) -> (1, T)
    input_model = input_waveform.unsqueeze(0).squeeze(1)  # shape: (1, T)
    target_model = target_waveform.unsqueeze(0).squeeze(1)  # shape: (1, T)

    # 모델 추론: 입력을 넣고 target을 conditioner로 사용하여 예측 생성
    with torch.no_grad():
        t = torch.randint(0, len(params.noise_schedule), [input_model.shape[0]], device=input_model.device)
        conditioner = target_model.unsqueeze(1)  # (B, 1, T)
        predicted_waveform = model(input_model, t, conditioner)
        # 예측된 waveform 길이가 target과 다르면 크롭
        if predicted_waveform.size(2) > target_model.size(1):
            predicted_waveform = predicted_waveform[:, :, :target_model.size(1)]
        elif predicted_waveform.size(2) < target_model.size(1):
            target_model = target_model[:, :predicted_waveform.size(2)]

    # tensor를 numpy로 변환 (채널 차원 제거)
    input_np = input_waveform.squeeze().cpu().numpy()
    target_np = target_waveform.squeeze().cpu().numpy()
    output_np = predicted_waveform.squeeze().cpu().numpy()

    # Log-Mel Spectrogram 계산 (power spectrogram 후 로그 스케일 변환)
    input_mel = mel_transform(input_waveform.to(device)).cpu().numpy()[0]  # (n_mels, time)
    target_mel = mel_transform(target_waveform.to(device)).cpu().numpy()[0]
    output_mel = mel_transform(predicted_waveform.squeeze(1)).cpu().numpy()[0]

    input_log_mel = librosa.power_to_db(input_mel, ref=np.max)
    target_log_mel = librosa.power_to_db(target_mel, ref=np.max)
    output_log_mel = librosa.power_to_db(output_mel, ref=np.max)

    # 2행 3열 subplot: 상단은 Waveform, 하단은 Log-Mel Spectrogram (좌: Input, 중: Target, 우: Output)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"File Pair: {base}", fontsize=16)

    # Waveform 플롯
    axs[0, 0].plot(input_np)
    axs[0, 0].set_title("Input Waveform")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Amplitude")

    axs[0, 1].plot(target_np)
    axs[0, 1].set_title("Target Waveform")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Amplitude")

    axs[0, 2].plot(output_np)
    axs[0, 2].set_title("Output Waveform")
    axs[0, 2].set_xlabel("Time")
    axs[0, 2].set_ylabel("Amplitude")

    # Log-Mel Spectrogram 플롯
    im0 = axs[1, 0].imshow(input_log_mel, aspect='auto', origin='lower', interpolation='none')
    axs[1, 0].set_title("Input Log-Mel Spectrogram")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Mel Bands")
    fig.colorbar(im0, ax=axs[1, 0])

    im1 = axs[1, 1].imshow(target_log_mel, aspect='auto', origin='lower', interpolation='none')
    axs[1, 1].set_title("Target Log-Mel Spectrogram")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Mel Bands")
    fig.colorbar(im1, ax=axs[1, 1])

    im2 = axs[1, 2].imshow(output_log_mel, aspect='auto', origin='lower', interpolation='none')
    axs[1, 2].set_title("Output Log-Mel Spectrogram")
    axs[1, 2].set_xlabel("Time")
    axs[1, 2].set_ylabel("Mel Bands")
    fig.colorbar(im2, ax=axs[1, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/hdd_ext/hdd3/joowoniese/diffwave4/testset/{base}.png")
    plt.show()
