import os
import librosa
import numpy as np

# Noise와 Clean 데이터 디렉토리 설정
noise_dir = "/hdd_ext/hdd3/joowoniese/diffwave2/audio_data/noise_audio"  # Noise 데이터 디렉토리 경로
clean_dir = "/hdd_ext/hdd3/joowoniese/diffwave2/audio_data/clean_audio"  # Clean 데이터 디렉토리 경로


# 주파수 대역 설정 함수
def calculate_frequency_bins(sr, n_mels):
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)
    low_freq_bins = np.where((mel_frequencies >= 20) & (mel_frequencies < 100))[0]
    mid_freq_bins = np.where((mel_frequencies >= 250) & (mel_frequencies <= 2000))[0]
    high_freq_bins = np.where(mel_frequencies > 2000)[0]
    return low_freq_bins, mid_freq_bins, high_freq_bins


# 주파수 대역별 차이 계산 함수
def calculate_frequency_difference(noise_file, clean_file, sr=22050, n_mels=128):
    # Noise 데이터 로드
    noise, _ = librosa.load(noise_file, sr=sr)
    noise_mel = librosa.feature.melspectrogram(y=noise, sr=sr, n_mels=n_mels)

    # Clean 데이터 로드
    clean, _ = librosa.load(clean_file, sr=sr)
    clean_mel = librosa.feature.melspectrogram(y=clean, sr=sr, n_mels=n_mels)

    # 주파수 대역 계산
    low_freq_bins, mid_freq_bins, high_freq_bins = calculate_frequency_bins(sr, n_mels)

    # 주파수 대역별 차이 계산
    low_diff = np.mean(np.abs(noise_mel[low_freq_bins, :] - clean_mel[low_freq_bins, :]))
    mid_diff = np.mean(np.abs(noise_mel[mid_freq_bins, :] - clean_mel[mid_freq_bins, :]))
    high_diff = np.mean(np.abs(noise_mel[high_freq_bins, :] - clean_mel[high_freq_bins, :]))

    return low_diff, mid_diff, high_diff


# 모든 파일 처리 및 결과 저장
low_diff_list = []
mid_diff_list = []
high_diff_list = []

for file_name in os.listdir(noise_dir):
    if file_name.endswith('.wav'):  # .wav 파일만 처리
        noise_file = os.path.join(noise_dir, file_name)
        clean_file = os.path.join(clean_dir, file_name)

        if os.path.exists(clean_file):  # Clean 파일이 존재하는 경우
            try:
                low_diff, mid_diff, high_diff = calculate_frequency_difference(noise_file, clean_file)
                low_diff_list.append(low_diff)
                mid_diff_list.append(mid_diff)
                high_diff_list.append(high_diff)
                print(f"{file_name}: Low: {low_diff:.2f}, Mid: {mid_diff:.2f}, High: {high_diff:.2f}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file_name}, 오류: {e}")
        else:
            print(f"Clean 파일 없음: {file_name}")

# 대역별 평균값 계산
low_diff_avg = np.mean(low_diff_list) if low_diff_list else 0
mid_diff_avg = np.mean(mid_diff_list) if mid_diff_list else 0
high_diff_avg = np.mean(high_diff_list) if high_diff_list else 0

# 평균값 출력
print("\n전체 파일에 대한 주파수 대역별 평균 차이값:")
print(f"저주파 평균 차이값 (20Hz~100Hz): {low_diff_avg:.2f}")
print(f"중간 주파수 평균 차이값 (250Hz~2000Hz): {mid_diff_avg:.2f}")
print(f"고주파 평균 차이값 (2000Hz~): {high_diff_avg:.2f}")