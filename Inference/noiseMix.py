from pydub import AudioSegment

def mix_audio_by_ratio(file1, file2, output_file, ratio1=0.5, ratio2=0.5):
    """
    두 개의 오디오 파일을 비율 기반으로 믹스하고 저장하는 함수.

    :param file1: 첫 번째 오디오 파일 경로
    :param file2: 두 번째 오디오 파일 경로
    :param output_file: 믹스된 오디오 파일 저장 경로
    :param ratio1: 첫 번째 오디오의 믹스 비율 (0.0 ~ 1.0)
    :param ratio2: 두 번째 오디오의 믹스 비율 (0.0 ~ 1.0)
    """
    # 오디오 파일 로드
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    # 두 오디오 파일 길이를 맞추기 위해 짧은 길이 기준으로 자르기
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # 비율 합이 1이 되도록 정규화
    total_ratio = ratio1 + ratio2
    ratio1 /= total_ratio
    ratio2 /= total_ratio

    # 볼륨 가중치 적용 (dB 변환)
    gain1 = 20 * (1 - ratio1)  # 비율이 작을수록 음량 감소
    gain2 = 20 * (1 - ratio2)

    audio1 = audio1 - gain1
    audio2 = audio2 - gain2

    # 오디오 믹스
    mixed_audio = audio1.overlay(audio2)

    # 결과 저장
    mixed_audio.export(output_file, format="wav")
    print(f"믹스된 오디오가 {output_file}로 저장되었습니다.")


output = "/hdd_ext/hdd3/hyundaiProject/Dataset/Color-Music/Red/waveform/LANY - Malibu Nights.wav"
noisesource = "/home/joowoniese/NoiseCancelling_Antinoise/antinoise/"
# 사용 예시
noisesourcefile = noisesource.replace(".wav", "")
mix_audio_by_ratio(output, noisesource, f"/hdd_ext/hdd3/joowoniese/diffwave4/testset/ideal_output_mixed/{noisesource}_mixed.wav", ratio1=0.7, ratio2=0.3)
