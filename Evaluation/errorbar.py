import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV íŒŒì¼ ë¶ˆëŸ¬ì˜¤ê¸°
csv_path = "/hdd_ext/hdd3/joowoniese/diffwave4/testset/evaluation_losses.csv"
df = pd.read_csv(csv_path)

# ì†ì‹¤ í•­ëª© ì •ì˜
loss_types = ['l1_loss', 'custom_loss', 'total_loss']
bar_colors = ['#1f77b4', '#2ca02c', '#d62728']  # íŒŒëž‘, ì´ˆë¡, ë¹¨ê°•

# â–¶ï¸ ê° ì†ì‹¤ í•­ëª©ì— ëŒ€í•œ í†µê³„ ë¶„ì„ ì¶œë ¥
print("ðŸ” Loss Statistics:")
for loss in loss_types:
    print(f"\nâ–¶ {loss.upper()}")
    print(f"  - Mean       : {df[loss].mean():.6f}")
    print(f"  - Std Dev    : {df[loss].std():.6f}")
    print(f"  - Min        : {df[loss].min():.6f}")
    print(f"  - 25% (Q1)   : {df[loss].quantile(0.25):.6f}")
    print(f"  - Median (Q2): {df[loss].median():.6f}")
    print(f"  - 75% (Q3)   : {df[loss].quantile(0.75):.6f}")
    print(f"  - Max        : {df[loss].max():.6f}")
    print(f"  - IQR        : {df[loss].quantile(0.75) - df[loss].quantile(0.25):.6f}")

# âœ… 1. Error bar ë§‰ëŒ€ ê·¸ëž˜í”„
means = df[loss_types].mean()
stds = df[loss_types].std()

plt.figure(figsize=(8, 5))
plt.bar(
    loss_types,
    means,
    yerr=stds,
    capsize=10,
    alpha=0.8,
    edgecolor='black',
    color=bar_colors,
    error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=1)
)
plt.ylabel("Loss Value", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Evaluation Loss Comparison (Â±1 STD)", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/errorbar.png")
plt.show()
plt.close()

# âœ… 2. Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[loss_types], palette=bar_colors)
plt.title("Evaluation Loss Distribution", fontsize=16)
plt.ylabel("Loss Value", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/hdd_ext/hdd3/joowoniese/diffwave4/testset/plot/boxplot.png")
plt.show()
plt.close()
