import os
from pydub import AudioSegment

def mix_audio_by_db(file1, file2, output_file, gain_db1=0, gain_db2=0):
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    audio1 = audio1 + gain_db1
    audio2 = audio2 + gain_db2

    mixed_audio = audio1.overlay(audio2)
    mixed_audio.export(output_file, format="wav")
    print(f"✅ 믹스 완료: {output_file}")

# 디렉토리 경로 설정
music_dir = "/hdd_ext/hdd3/hyundaiProject/Dataset/Color-Music/Red/waveform/"
antinoise_dir = "/home/joowoniese/NoiseCancelling_Antinoise/antinoise/"
anc_antinoise_dir = "/home/joowoniese/NoiseCancelling_Antinoise/ANC_antinoise/"
output_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/anc_output_mixed/"

os.makedirs(output_dir, exist_ok=True)

# 기준: antinoise 디렉토리에서 '_anti.wav' 파일들
antinoise_files = [f for f in os.listdir(antinoise_dir) if f.endswith("_anti.wav")]

# ANC_antinoise에 대응되는 '_anc.wav' 파일 있는지 확인
anc_files = set(os.listdir(anc_antinoise_dir))
valid_noise_pairs = []

for anti_file in antinoise_files:
    base = anti_file.replace("_anti.wav", "")
    anc_file = base + "_anc.wav"
    if anc_file in anc_files:
        valid_noise_pairs.append((base, anc_file))

# music 파일들
music_files = [f for f in os.listdir(music_dir) if f.endswith(".wav")]

# 믹싱 시작
for music_file in music_files:
    music_path = os.path.join(music_dir, music_file)
    song_base = os.path.splitext(music_file)[0].replace(" ", "").replace("-", "_")

    for base, anc_file in valid_noise_pairs:
        noise_path = os.path.join(anc_antinoise_dir, anc_file)
        output_filename = f"{song_base}_{base}_mixed.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[MIXING] {music_file} + {anc_file}")
        mix_audio_by_db(music_path, noise_path, output_path, gain_db1=-3, gain_db2=-10)

print("\n✅ 모든 오디오 믹싱이 완료되었습니다.")
