import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# SNR 값이 저장된 CSV 불러오기
df = pd.read_csv("/hdd_ext/hdd3/joowoniese/diffwave4/testset/evaluation_losses.csv")

plt.figure(figsize=(8, 5))
# plt.bar(df["filename"], df["snr_db"])
# plt.xticks(rotation=90)
# plt.xlabel("Audio Filename")
# plt.ylabel("SNR (dB)")
# plt.title("SNR per Audio File")
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# plt.hist(df["snr_db"], bins=20, color='skyblue', edgecolor='grey')
# # plt.hist(df["snr_db"], bins=20, color='skyblue')
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.ylabel("Number of Files", fontsize=12)
# plt.title("Distribution of SNR across Test Set", fontsize=16)
# plt.grid(True)
# plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/SNRDistribution.png")
# plt.show()
#
# # 바이올린 플롯 예시
# sns.violinplot(x=df["snr_db"], color='skyblue')
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.title("Violin Plot of SNR", fontsize=16)
# plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/SNRviolin.png")
# plt.show()
#
# sns.kdeplot(data=df, x="snr_db", fill=True, color='green')
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.title("Kernel Density Estimate of SNR", fontsize=16)
# plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/SNRkernelDensity.png")
# plt.show()
#
# top = df.sort_values("snr_db", ascending=False).iloc[0]
# bottom = df.sort_values("snr_db").iloc[0]
# print("📢 Best SNR:", top["filename"], "=", top["snr_db"])
# print("📉 Worst SNR:", bottom["filename"], "=", bottom["snr_db"])
plt.tick_params(axis='both', labelsize=10)

plt.scatter(df["snr_db"], df["l1_loss"], c='green', s=10, alpha=0.5)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("L1 Loss", fontsize=12)
plt.title("SNR according to the L1 Loss", fontsize=16)
plt.grid(True)
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/SNRscatter.png")
plt.show()

