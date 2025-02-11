import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 불러오기
file_path = '/hdd_ext/hdd3/joowoniese/diffwave4/event_logs_mel/loss_log_mel.csv'
data = pd.read_csv(file_path, sep=',')  # 쉼표로 구분된 파일 처리

# 헤더 확인
print(data.columns)

# 열 이름에서 공백 제거 (필요시)
data.columns = data.columns.str.strip()

# loss 값이 10 이상인 행 제거
data = data[data['loss'] < 2]

# step 순서대로 정렬
data = data.sort_values(by='step')

# Loss 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['loss'], label='Loss', color='blue')
plt.title('Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('/hdd_ext/hdd3/joowoniese/diffwave4/train_logs/mel_loss_graph.png')  # Loss 그래프 저장
plt.show()

# Grad Norm 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['grad_norm'], label='Grad Norm', color='green')
plt.title('Grad Norm over Steps')
plt.xlabel('Step')
plt.ylabel('Grad Norm')
plt.legend()
plt.grid()
plt.savefig('/hdd_ext/hdd3/joowoniese/diffwave4/train_logs/mel_grad_norm_graph.png')  # Grad Norm 그래프 저장
plt.show()
