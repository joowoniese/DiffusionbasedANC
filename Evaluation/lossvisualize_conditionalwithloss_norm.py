import pandas as pd
import matplotlib.pyplot as plt
import re

# 개선된 tensor 값 추출 함수: 이미 숫자면 그대로, 아니면 tensor 문자열에서 숫자를 추출
def extract_tensor_value(x):
    try:
        # 이미 숫자형이면 그대로 반환
        return float(x)
    except (ValueError, TypeError):
        pass
    s = str(x)
    # 예: "tensor(1.2730, device='cuda:0')" 또는 "tensor([0.5798], device='cuda:0', grad_fn=<AddBackward0>)"
    match = re.search(r"tensor\(\[?([-+]?\d*\.\d+|\d+)\]?", s)
    if match:
        return float(match.group(1))
    return None  # 추출 실패 시 None 반환

# CSV 파일 읽기 (파일 경로에 맞게 수정)
file_path = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/event_logs_norm2/losslog.csv"
df = pd.read_csv(file_path)

# 사용할 열 목록
columns = ['loss', 'original_loss', 'custom_loss']

# 각 열에 대해 값 확인 (첫 5행 출력)
print("데이터 확인:")
print(df[columns].head())

# 각 열의 데이터를 float형으로 변환 (tensor 값 추출)
for col in columns:
    df[col] = df[col].apply(extract_tensor_value)

# 변환 후 다시 값 확인
print("\n변환 후 데이터 확인:")
print(df[columns].head())

# 각 열별로 별도의 그래프 생성
for col in columns:
    # 만약 해당 열에 모두 None이나 NaN이면 그래프가 비어있을 수 있으므로 체크
    if df[col].dropna().empty:
        print(f"{col} 열에 변환 가능한 데이터가 없습니다.")
        continue

    plt.figure(figsize=(8, 6))
    plt.plot(df[col])
    plt.title(col, fontsize=16)
    plt.xlabel('Epoch | Iteration')
    plt.ylabel('값')
    plt.grid(True)
    plt.savefig(f"/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/{col}.png")
    plt.show()
