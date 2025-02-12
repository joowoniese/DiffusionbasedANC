import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로 설정
csv_file = "/hdd_ext/hdd3/joowoniese/diffwave4/event_logs_mel/loss_log_mel.csv"

# CSV 파일 로드 (헤더 없음)
try:
    data = pd.read_csv(csv_file, header=None)  # 헤더 없음
    data.columns = ['step', 'loss', 'grad_norm']  # 열 이름 직접 할당
    print(f"[INFO] CSV 파일 로드 완료: {csv_file}")
except FileNotFoundError:
    print(f"[ERROR] CSV 파일을 찾을 수 없습니다: {csv_file}")
    exit()

print(data)
# Loss 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['loss'], label='Loss', color='blue', linewidth=1)
plt.ylim(0, 0.3)  # y축 범위 설정
plt.title('Loss over Steps', fontsize=16)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
# 그래프 저장
loss_image_path = "/hdd_ext/hdd3/joowoniese/diffwave4/train_logs/loss_graph.png"
plt.savefig(loss_image_path)
print(f"[INFO] Loss 그래프가 저장되었습니다: {loss_image_path}")
plt.show()

# Grad Norm 그래프
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['grad_norm'], label='Grad Norm', color='orange', linewidth=1)
plt.ylim(0, 0.3)  # y축 범위 설정
plt.title('Gradient Norm over Steps', fontsize=16)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Gradient Norm', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
# 그래프 저장
grad_norm_image_path = "/hdd_ext/hdd3/joowoniese/diffwave4/train_logs/grad_norm_graph.png"
plt.savefig(grad_norm_image_path)
print(f"[INFO] Grad Norm 그래프가 저장되었습니다: {grad_norm_image_path}")
plt.show()