import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import time

# 실제 프로젝트 모듈 경로에 맞게 임포트하세요.
from conditionalDiffwave.model import DiffWave
from conditionalDiffwave.params import AttrDict, params as default_params
from conditionalDiffwave.learner_condition_withloss_norm import DiffWaveLearner

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 입력 크기가 동일하면 속도 개선 효과

# test와 target 오디오가 위치한 디렉터리 경로
test_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/anc_output_mixed/"
target_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/ideal_output_mixed/"

# 두 디렉터리의 파일 목록 불러오기
test_files = os.listdir(test_dir)
target_files = os.listdir(target_dir)


def compute_snr(target, prediction):
    """
    SNR (Signal-to-Noise Ratio)를 dB 단위로 계산합니다.
    :param target: Ground-truth waveform (Tensor or np.array)
    :param prediction: Predicted waveform (Tensor or np.array)
    :return: SNR in dB
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    noise = target - prediction
    signal_power = np.sum(target ** 2)
    noise_power = np.sum(noise ** 2) + 1e-9  # 분모 0 방지

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# 파일명의 공통 접두어 추출 함수
def get_common_base(filename, marker):
    if filename.endswith(marker):
        return filename.replace(marker, '')
    return None


# test 파일 매핑 (예: '_anc_mixed.wav')
test_dict = {}
for f in test_files:
    base = get_common_base(f, 'mixed.wav')
    if base is not None:
        test_dict[base] = os.path.join(test_dir, f)

# target 파일 매핑 (예: '_anti_mixed.wav')
target_dict = {}
for f in target_files:
    base = get_common_base(f, 'mixed.wav')
    if base is not None:
        target_dict[base] = os.path.join(target_dir, f)

# 공통 접두어를 가진 파일 쌍 선택
common_bases = set(test_dict.keys()) & set(target_dict.keys())
print(f"총 {len(common_bases)} 쌍의 파일이 발견되었습니다.")

# 제공된 파라미터 설정 (여기서는 params.audio_len=22050*5)
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

# 모델 및 옵티마이저 초기화 후 GPU로 이동
model = DiffWave(params).to(device)
model.eval()  # 평가 모드
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

# 체크포인트 저장 디렉터리
model_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/model_logs_norm2/"
os.makedirs(model_dir, exist_ok=True)

# DiffWaveLearner 인스턴스 생성 (테스트 목적으로 weight 불러오기)
learner = DiffWaveLearner(model_dir, model, None, optimizer, params, fp16=False)

# 저장된 checkpoint 불러오기
if learner.restore_from_checkpoint('weights'):
    print("Checkpoint 불러오기 성공!")
else:
    print("Checkpoint 불러오기 실패: 파일 경로와 이름을 확인하세요.")

# 평가 모드 재확인
model.eval()
results = []
processing_times = []  # 처리 시간 저장 리스트

# 멜 스펙트로그램 transform 생성 (GPU로 이동)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=params.sample_rate,
    n_mels=params.n_mels,
    n_fft=params.n_fft,
    hop_length=params.hop_samples
).to(device)

# 각 파일 쌍에 대해 테스트 실행 (최대 3000개만 처리)
for base in sorted(common_bases)[:3000]:
    start_time = time.time()  # 시작 시간 기록

    test_path = test_dict[base]
    target_path = target_dict[base]

    # torchaudio.load: (waveform, sample_rate) 반환 (CPU에서 로드됨)
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

    # 배치 차원 추가 및 채널 차원 제거: (채널, T) -> (1, T)
    input_waveform = input_waveform.unsqueeze(0).squeeze(1)
    target_waveform = target_waveform.unsqueeze(0).squeeze(1)

    # 길이를 params.audio_len로 맞추기
    desired_length = params.audio_len
    if input_waveform.size(1) > desired_length:
        input_waveform = input_waveform[:, :desired_length]
    if target_waveform.size(1) > desired_length:
        target_waveform = target_waveform[:, :desired_length]

    # GPU로 데이터 전송
    input_waveform = input_waveform.to(device)
    target_waveform = target_waveform.to(device)

    # 평가: GPU에서 fp16 사용 옵션 (원하는 경우 사용)
    with torch.inference_mode():
        # t 생성 및 조건자(conditioner) 준비
        t = torch.randint(0, len(params.noise_schedule), [input_waveform.shape[0]], device=device)
        conditioner = target_waveform.unsqueeze(1)  # (B, 1, T)

        # autocast 사용 (fp16 연산으로 속도 개선, 단 모델이 지원해야 합니다)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            predicted_waveform = model(input_waveform, t, conditioner)

        # 처리 시간 측정
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        processing_times.append(elapsed)
        print(f"[{base}] 처리 시간: {elapsed:.3f}초")

        # 예측된 waveform 길이 조정 (target과 맞추기)
        if predicted_waveform.size(2) > target_waveform.size(1):
            predicted_waveform = predicted_waveform[:, :, :target_waveform.size(1)]
        elif predicted_waveform.size(2) < target_waveform.size(1):
            target_waveform = target_waveform[:, :predicted_waveform.size(2)]

        # L1 손실 계산
        loss_fn = torch.nn.L1Loss()
        original_loss = loss_fn(target_waveform, predicted_waveform.squeeze(1))

        # custom loss 계산: 멜 스펙트로그램 사용 (GPU 상에서 계산)
        predicted_mel = mel_transform(predicted_waveform.squeeze(1))
        target_mel = mel_transform(target_waveform)
        custom_loss = learner.compute_custom_loss(predicted_mel, target_mel, params.n_mels, params.sample_rate)
        total_loss = original_loss.item() + custom_loss.item()

        # SNR 계산
        snr = compute_snr(target_waveform.squeeze(), predicted_waveform.squeeze())

        print("SNR (dB):", snr)
        print(f"\n[파일 쌍: {base}]")
        print("예측된 waveform shape:", predicted_waveform.shape)
        print("L1 손실:", original_loss.item())
        print("커스텀 손실:", custom_loss.item())
        print("총합 손실:", total_loss)

        # 결과 저장 (나중에 CSV로 저장)
        results.append({
            "filename": base,
            "l1_loss": original_loss.item(),
            "custom_loss": custom_loss.item(),
            "total_loss": total_loss,
            "time": elapsed,
            "snr_db": snr
        })

        # 모든 결과를 CSV로 한 번에 저장 (반복마다 저장하지 않음)
        results_df = pd.DataFrame(results)
        output_csv_path = os.path.join(model_dir, "/hdd_ext/hdd3/joowoniese/diffwave4/testset/evaluation_losses.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"\n[✅ 모든 결과가 CSV 파일로 저장되었습니다: {output_csv_path}]")


# 평균 처리 시간 출력
if processing_times:
    avg_time = sum(processing_times) / len(processing_times)
    print(f"\n✅ 평균 처리 시간: {avg_time:.3f}초 (총 {len(processing_times)}개 샘플)")
