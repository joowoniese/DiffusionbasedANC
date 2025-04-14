import os
import glob
import re
import torch
import numpy as np
import soundfile as sf

# ---------- 1. GPU 설정 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 현재 디바이스: {device}")

# ---------- FxLMM 필터 (PyTorch + GPU) ----------
class FxLMM_GPU:
    def __init__(self, filter_len=256, mu=0.005, delta=1.0):
        self.mu = mu
        self.delta = delta
        self.filter_len = filter_len
        self.device = device

        self.w = torch.zeros(filter_len, device=device)
        self.x_buffer = torch.zeros(filter_len, device=device)
        self.s = torch.ones(filter_len, device=device) * 0.1  # secondary path

    def filtered_x(self, x):
        # 입력 x는 1D 텐서 → (B, C, T)로 reshape
        x_ = x.view(1, 1, -1)
        s_ = self.s.flip(0).view(1, 1, -1)
        y_ = torch.nn.functional.conv1d(x_, s_, padding='same')
        return y_.view(-1)

    def huber_derivative(self, e):
        abs_e = torch.abs(e)
        return torch.where(abs_e <= self.delta, e, self.delta * torch.sign(e))

    def adapt(self, x, d):
        self.x_buffer = torch.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        x_filt = self.filtered_x(self.x_buffer)
        y = torch.dot(self.w, x_filt)
        e = d - y

        grad = self.huber_derivative(e)
        norm = torch.dot(x_filt, x_filt) + 1e-6
        self.w += (self.mu / norm) * grad * x_filt

        return -y.item(), e.item()

# ---------- 2. 경로 설정 ----------
mixed_dir = '/hdd_ext/hdd3/joowoniese/diffwave4/testset/ideal_output_mixed/'
noise_dir = '/home/joowoniese/NoiseCancelling_Antinoise/noise/'

# 🔽 저장할 경로 지정 (여기만 바꾸면 됨!)
antinoise_output_dir = '/home/joowoniese/NoiseCancelling_Antinoise/FxLMM_GPU/'
os.makedirs(antinoise_output_dir, exist_ok=True)

# ---------- 3. 대상 파일 필터링 ----------
wav_files = glob.glob(os.path.join(mixed_dir, '*.wav'))
pattern = re.compile(r'(N-10_[^_]+_[^_]+_[^_]+_[^_]+_[^_]+)')
filtered_ids = []

for filepath in wav_files:
    filename = os.path.basename(filepath).replace('_mixed', '').replace('.wav', '')
    match = pattern.search(filename)
    if match:
        filtered_ids.append(match.group(1))
    else:
        print(f"매치 안 됨: {filename}")

print(f"[INFO] 총 {len(filtered_ids)}개의 유효한 ID가 추출됨")

# ---------- 4. FxLMM GPU 적용 ----------
for file_id in filtered_ids:
    input_path = os.path.join(noise_dir, file_id + '.wav')
    output_path = os.path.join(antinoise_output_dir, file_id + '.wav')

    if not os.path.exists(input_path):
        print(f"[경고] {file_id}.wav 파일 없음. 건너뜀.")
        continue

    print(f"[처리 중] {file_id}.wav")

    # 오디오 불러오기
    x_np, sr = sf.read(input_path)
    if x_np.ndim > 1:
        x_np = x_np[:, 0]  # 모노 처리

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    fxlmm = FxLMM_GPU(filter_len=255, mu=0.01, delta=1.0)
    antinoise = []

    for i in range(len(x)):
        noise_sample = x[i]
        desired = torch.tensor(0.0, device=device)
        anti, _ = fxlmm.adapt(noise_sample, desired)
        antinoise.append(anti)

    antinoise = np.array(antinoise)
    antinoise = np.clip(antinoise, -1.0, 1.0)  # [-1, 1] 클리핑
    sf.write(output_path, antinoise, sr)

    print(f"[저장 완료] {output_path}")

print("✅ 모든 anti-noise(FxLMM GPU 기반) 생성 및 저장 완료!")
