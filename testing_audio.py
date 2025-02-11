import librosa

file_path = "/hdd_ext/hdd3/joowoniese/diffwave2/audio_data/noise_audio/Watch This - ARIZONATEARS Pluggnb Remix_mixed.wav"
audio, sr = librosa.load(file_path, sr=22050)  # 샘플링 레이트 22.05 kHz로 로드
print(f"Sample rate: {sr}, Length (samples): {len(audio)}, Duration (seconds): {len(audio) / sr}")
