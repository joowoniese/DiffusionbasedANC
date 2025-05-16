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
columns = ['original_loss', 'custom_loss']

# tensor 문자열 → float 변환
df['loss'] = df['loss'].apply(extract_tensor_value)
for col in columns:
    df[col] = df[col].apply(extract_tensor_value)

# 유효한 값만 남기기
df = df.dropna(subset=['step'] + columns)

# -------------------------------
# 1. original + custom loss 함께 그리기
# -------------------------------
plt.figure(figsize=(16, 6))
plt.plot(df['step'], df['original_loss'], label='Reconstructive Loss', color='#c40000', linewidth=2)
plt.plot(df['step'], df['custom_loss'], label='Custom Loss', color='#008302', linewidth=1)
plt.tick_params(axis='both', labelsize=20)
plt.title("Train Loss Curve", fontsize=30)
plt.xlabel("Step", fontsize=20)
plt.ylabel("Loss Value", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/original_custom_losses_by_step.png")
plt.show()
plt.close()

# -------------------------------
# 2. original_loss 단독 그래프
# -------------------------------
plt.figure(figsize=(18, 6))
plt.plot(df['step'], df['original_loss'], label='Reconstructive Loss', color='red', linewidth=1.5)
plt.tick_params(axis='both', labelsize=14)
plt.title("Original Loss Curve", fontsize=20)
plt.xlabel("Step", fontsize=18)
plt.ylabel("Loss Value", fontsize=18)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/original_loss_by_step.png")
plt.show()
plt.close()

# -------------------------------
# 3. custom_loss 단독 그래프
# -------------------------------
plt.figure(figsize=(18, 6))
plt.plot(df['step'], df['custom_loss'], label='Custom Loss', color='blue', linewidth=1.5)
plt.tick_params(axis='both', labelsize=14)
plt.title("Custom Loss Curve", fontsize=20)
plt.xlabel("Step", fontsize=18)
plt.ylabel("Loss Value", fontsize=18)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/custom_loss_by_step.png")
plt.show()
plt.close()
