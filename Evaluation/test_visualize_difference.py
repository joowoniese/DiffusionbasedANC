import os
import torch
import torchaudio
import numpy as np
import librosa
import pandas as pd
import random
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 입력 크기가 동일하면 속도 개선 효과

# === 폰트 크기 설정 ===
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 15

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

model = DiffWave(params)
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

model_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/model_logs_norm2/"
os.makedirs(model_dir, exist_ok=True)

learner = DiffWaveLearner(model_dir, model, None, optimizer, params, fp16=False)
if learner.restore_from_checkpoint('weights'):
    print("Checkpoint 불러오기 성공!")
else:
    print("Checkpoint 불러오기 실패: 파일 경로와 이름을 확인하세요.")

device = next(model.parameters()).device

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=params.sample_rate,
    n_mels=params.n_mels,
    n_fft=params.n_fft,
    hop_length=params.hop_samples
).to(device)

desired_length = params.audio_len

visualization_bases = random.sample(sorted(common_bases), min(5, len(common_bases)))

for base in visualization_bases:
    test_path = test_dict[base]
    target_path = target_dict[base]

    input_waveform, sr_input = torchaudio.load(test_path)
    target_waveform, sr_target = torchaudio.load(target_path)

    if sr_input != sr_target:
        print(f"[WARNING] {base}: Sample rate 불일치 (test: {sr_input}, target: {sr_target}). 건너뜁니다.")
        continue

    if input_waveform.shape[0] > 1:
        input_waveform = input_waveform[0:1, :]
    if target_waveform.shape[0] > 1:
        target_waveform = target_waveform[0:1, :]

    if input_waveform.size(1) > desired_length:
        input_waveform = input_waveform[:, :desired_length]
    if target_waveform.size(1) > desired_length:
        target_waveform = target_waveform[:, :desired_length]

    input_model = input_waveform.unsqueeze(0).squeeze(1)
    target_model = target_waveform.unsqueeze(0).squeeze(1)

    with torch.no_grad():
        t = torch.randint(0, len(params.noise_schedule), [input_model.shape[0]], device=input_model.device)
        conditioner = target_model.unsqueeze(1)
        predicted_waveform = model(input_model, t, conditioner)
        if predicted_waveform.size(2) > target_model.size(1):
            predicted_waveform = predicted_waveform[:, :, :target_model.size(1)]
        elif predicted_waveform.size(2) < target_model.size(1):
            target_model = target_model[:, :predicted_waveform.size(2)]

    input_np = input_waveform.squeeze().cpu().numpy()
    target_np = target_waveform.squeeze().cpu().numpy()
    output_np = predicted_waveform.squeeze().cpu().numpy()
    diff_waveform = target_np - output_np

    global_ymin = min(np.min(input_np), np.min(target_np), np.min(output_np), np.min(diff_waveform))
    global_ymax = max(np.max(input_np), np.max(target_np), np.max(output_np), np.max(diff_waveform))
    dynamic_diff_ymin = np.min(diff_waveform)
    dynamic_diff_ymax = np.max(diff_waveform)

    input_mel = mel_transform(input_waveform.to(device)).cpu().numpy()[0]
    target_mel = mel_transform(target_waveform.to(device)).cpu().numpy()[0]
    output_mel = mel_transform(predicted_waveform.squeeze(1)).cpu().numpy()[0]

    input_log_mel = librosa.power_to_db(input_mel, ref=np.max)
    target_log_mel = librosa.power_to_db(target_mel, ref=np.max)
    output_log_mel = librosa.power_to_db(output_mel, ref=np.max)
    diff_log_mel = target_log_mel - output_log_mel

    global_db_min = min(np.min(input_log_mel), np.min(target_log_mel), np.min(output_log_mel), np.min(diff_log_mel))
    global_db_max = max(np.max(input_log_mel), np.max(target_log_mel), np.max(output_log_mel), np.max(diff_log_mel))
    dynamic_diff_db_min = np.min(diff_log_mel)
    dynamic_diff_db_max = np.max(diff_log_mel)

    common_wave_x = (0, len(target_np))
    spec_h, spec_w = target_log_mel.shape
    common_spec_xlim = (0, spec_w)
    common_spec_ylim = (0, spec_h)

    fig, axs = plt.subplots(2, 5, figsize=(30, 10))
    fig.suptitle(f"File Pair: {base}", fontsize=TITLE_FONT_SIZE)

    axs[0, 0].plot(input_np)
    axs[0, 0].set_title("Input Waveform", fontsize=TITLE_FONT_SIZE)
    axs[0, 0].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    axs[0, 0].set_ylabel("Amplitude", fontsize=LABEL_FONT_SIZE)
    axs[0, 0].set_xlim(common_wave_x)
    axs[0, 0].set_ylim(global_ymin, global_ymax)
    axs[0, 0].tick_params(labelsize=TICK_FONT_SIZE)

    axs[0, 1].plot(target_np)
    axs[0, 1].set_title("Target Waveform", fontsize=TITLE_FONT_SIZE)
    axs[0, 1].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    axs[0, 1].set_ylabel("Amplitude", fontsize=LABEL_FONT_SIZE)
    axs[0, 1].set_xlim(common_wave_x)
    axs[0, 1].set_ylim(global_ymin, global_ymax)
    axs[0, 1].tick_params(labelsize=TICK_FONT_SIZE)

    axs[0, 2].plot(output_np)
    axs[0, 2].set_title("Output Waveform", fontsize=TITLE_FONT_SIZE)
    axs[0, 2].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    axs[0, 2].set_ylabel("Amplitude", fontsize=LABEL_FONT_SIZE)
    axs[0, 2].set_xlim(common_wave_x)
    axs[0, 2].set_ylim(global_ymin, global_ymax)
    axs[0, 2].tick_params(labelsize=TICK_FONT_SIZE)

    axs[0, 3].plot(diff_waveform)
    axs[0, 3].set_title("Waveform Diff (Fixed)", fontsize=TITLE_FONT_SIZE)
    axs[0, 3].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    axs[0, 3].set_ylabel("Amplitude", fontsize=LABEL_FONT_SIZE)
    axs[0, 3].set_xlim(common_wave_x)
    axs[0, 3].set_ylim(global_ymin, global_ymax)
    axs[0, 3].tick_params(labelsize=TICK_FONT_SIZE)

    axs[0, 4].plot(diff_waveform)
    axs[0, 4].set_title("Waveform Diff (Dynamic)", fontsize=TITLE_FONT_SIZE)
    axs[0, 4].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    axs[0, 4].set_ylabel("Amplitude", fontsize=LABEL_FONT_SIZE)
    axs[0, 4].set_xlim(common_wave_x)
    axs[0, 4].set_ylim(dynamic_diff_ymin, dynamic_diff_ymax)
    axs[0, 4].tick_params(labelsize=TICK_FONT_SIZE)

    im_titles = ["Input Log-Mel", "Target Log-Mel", "Output Log-Mel", "Log-Mel Diff (Fixed)", "Log-Mel Diff (Dynamic)"]
    log_mels = [input_log_mel, target_log_mel, output_log_mel, diff_log_mel, diff_log_mel]
    db_ranges = [(global_db_min, global_db_max)] * 4 + [(dynamic_diff_db_min, dynamic_diff_db_max)]

    for i in range(5):
        im = axs[1, i].imshow(log_mels[i], aspect='auto', origin='lower', interpolation='none',
                              vmin=db_ranges[i][0], vmax=db_ranges[i][1])
        axs[1, i].set_title(im_titles[i], fontsize=TITLE_FONT_SIZE)
        axs[1, i].set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
        axs[1, i].set_ylabel("Mel Bands", fontsize=LABEL_FONT_SIZE)
        axs[1, i].set_xlim(common_spec_xlim)
        axs[1, i].set_ylim(common_spec_ylim)
        axs[1, i].tick_params(labelsize=TICK_FONT_SIZE)
        fig.colorbar(im, ax=axs[1, i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=3, h_pad=2)
    plt.savefig(f"/hdd_ext/hdd3/joowoniese/diffwave4/testset/sample_5/{base}.png")
    plt.show()