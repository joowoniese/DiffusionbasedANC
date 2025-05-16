import pandas as pd

# CSV 파일 경로
csv_path = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/evaluation_losses.csv"

# CSV 읽기
df = pd.read_csv(csv_path)

# 평균 time 계산
mean_time = df["time"].mean()

print(f"✅ 테스트셋 추론 시간 평균: {mean_time:.4f}초")
