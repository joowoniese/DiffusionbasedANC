import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = "/hdd_ext/hdd3/joowoniese/diffwave4/ConditionalDiffwavewithLoss2/event_logs_norm2/losslog.csv"

# CSV 불러오기
df = pd.read_csv(csv_path)

# 그래프 그리기
plt.figure(figsize=(18, 6))
plt.plot(df["step"], df["weight_original"], label="Original Loss Weight", color="blue", linewidth=2)
plt.plot(df["step"], df["weight_custom"], label="Custom Loss Weight", color="green", linewidth=2)
plt.tick_params(axis='both', labelsize=20)

# 라벨 & 제목
plt.xlabel("Step", fontsize=20)
plt.ylabel("Weight Value", fontsize=20)
plt.title("Weight Trend Over Steps", fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

# 저장
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/weight_trend_by_step.png")

# 출력
plt.show()
