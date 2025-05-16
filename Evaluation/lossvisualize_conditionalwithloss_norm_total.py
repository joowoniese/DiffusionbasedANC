import pandas as pd
import matplotlib.pyplot as plt
import re

# tensor 값 추출 함수
def extract_tensor_value(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        pass
    s = str(x)
    match = re.search(r"tensor\(\[?([-+]?\d*\.\d+|\d+)\]?", s)
    if match:
        return float(match.group(1))
    return None

# CSV 파일 읽기
file_path = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/event_logs_norm2/losslog.csv"
df = pd.read_csv(file_path)

# 사용할 열
columns = ['loss', 'original_loss', 'custom_loss']

# tensor 문자열 → float 변환
for col in columns:
    df[col] = df[col].apply(extract_tensor_value)

# 유효한 값만 남기기 (NaN 제거)
df = df.dropna(subset=columns)

# 🔥 하나의 그래프에 세 곡선을 다른 색으로 그리기
plt.figure(figsize=(10, 6))

plt.plot(df['loss'], label='Total Loss', color='red', linewidth=1)
plt.plot(df['original_loss'], label='Reconstructive Loss', color='blue', linewidth=1)
plt.plot(df['custom_loss'], label='Custom Loss', color='green', linewidth=1)

plt.title("Train Loss Curve (Total, Reconstructive, Custom)", fontsize=20)
plt.xlabel("Epoch | Iteration", fontsize=18)
plt.ylabel("Loss Value", fontsize=18)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/all_losses.png")
plt.show()
