import os
from pydub import AudioSegment

def mix_audio_by_db(file1, file2, output_file, gain_db1=0, gain_db2=0):
    """
    두 개의 오디오 파일을 데시벨(dB) 기반으로 조절하여 믹스하고 저장하는 함수.
    """
    # 오디오 파일 로드
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    # 두 오디오 파일 길이를 맞추기 위해 짧은 길이 기준으로 자르기
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # dB 단위로 볼륨 조절
    audio1 = audio1 + gain_db1
    audio2 = audio2 + gain_db2

    # 오디오 믹스
    mixed_audio = audio1.overlay(audio2)

    # 결과 저장
    mixed_audio.export(output_file, format="wav")
    print(f"✅ 믹스된 오디오가 저장되었습니다: {output_file}")

def standardize_name(name):
    """
    파일 이름에서 확장자 제거, 공백과 '-'를 제거하여 base name 통일.
    """
    base = os.path.splitext(name)[0]
    base = base.replace(" ", "").replace("-", "_")
    return base

# music source 파일 목록 (mix할 음악 파일들)
songNames = [
    "HYUKOH - TOMBOY.wav",
    "Yerin Baek - 0310.wav",
    "The Black Skirts - EVERYTHING.wav",
    "Post Malone - Psycho.wav",
    "Harry Styles - Falling.wav"
]

# 경로 설정
# antinoisesource_dir: 기준이 되는 파일명이 있는 디렉터리
antinoisesource_dir = "/home/joowoniese/NoiseCancelling_Antinoise/antinoise/"
# noisesource_dir: 실제로 믹스에 사용할 노이즈 파일들이 있는 디렉터리
noisesource_dir = "/home/joowoniese/NoiseCancelling_Antinoise/ANC_antinoise/"
# music source 파일들이 있는 디렉터리
musicsource_dir = "/hdd_ext/hdd3/hyundaiProject/Dataset/Color-Music/Red/waveform/"
# 믹스된 오디오 출력 디렉터리
output_dir = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/anc_output_mixed/"
os.makedirs(output_dir, exist_ok=True)

# antinoisesource_dir의 모든 .wav 파일 가져오기
antinoise_files = [f for f in os.listdir(antinoisesource_dir) if f.endswith(".wav")]
antinoise_dict = {}
for af in antinoise_files:
    base = standardize_name(af)  # 여기서는 파일명 전체를 표준화
    antinoise_dict[base] = af

# noisesource_dir의 파일 중 _anc.wav 또는 _anti.wav 파일만 선택
noise_files = [f for f in os.listdir(noisesource_dir) if f.endswith("_anc.wav") or f.endswith("_anti.wav")]
noise_dict = {}
for nf in noise_files:
    # _anc와 _anti를 제거하여 표준 base name 생성
    base = standardize_name(nf.replace("_anc", "").replace("_anti", ""))
    # 여러 파일이 있을 경우, 첫 번째 파일만 사용
    if base not in noise_dict:
        noise_dict[base] = nf

# music source 파일들도 base name 기준으로 dict 생성
music_dict = {}
for song in songNames:
    base = standardize_name(song)
    music_dict[base] = song

# 이제 antinoise_dict와 noise_dict의 공통 base name을 기준으로,
# 그리고 music_dict에도 존재하는 경우에만 믹스 진행
common_bases = set(antinoise_dict.keys()) & set(noise_dict.keys()) & set(music_dict.keys())
print(f"총 {len(common_bases)} 쌍의 파일이 발견되었습니다.")

for base in sorted(common_bases):
    music_file = os.path.join(musicsource_dir, music_dict[base])
    noise_file = os.path.join(noisesource_dir, noise_dict[base])
    output_file = os.path.join(output_dir, f"{base}_mixed.wav")

    print(f"\n[ 믹스 진행 ] {music_file} + {noise_file} → {output_file}")
    mix_audio_by_db(music_file, noise_file, output_file, gain_db1=-3, gain_db2=-10)

print("\n✅ 모든 오디오 믹싱이 완료되었습니다.")
