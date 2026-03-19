import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========= 在这里粘贴你图中的四个 7x7 矩阵 =========
# 例子里先放占位，直接替换为你的数据即可（必须保持 7x7 且数值在 0~1）
cm1 = np.array([
    [0.93, 0.00, 0.04, 0.02, 0.00, 0.00, 0.00],
    [0.00, 0.83, 0.00, 0.00, 0.00, 0.17, 0.00],
    [0.11, 0.00, 0.80, 0.00, 0.02, 0.08, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    [0.07, 0.00, 0.02, 0.05, 0.82, 0.02, 0.02],
    [0.04, 0.00, 0.02, 0.01, 0.00, 0.94, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
cm2 = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.96, 0.04, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.04, 0.96, 0.00, 0.00],
    [0.03, 0.00, 0.01, 0.09, 0.01, 0.86, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
])
cm3 = np.array([
    [0.96, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.08, 0.00, 0.84, 0.03, 0.00, 0.05, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.05, 0.91, 0.04, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
cm4 = np.array([
    [0.96, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.07, 0.00, 0.91, 0.00, 0.00, 0.02, 0.00],
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.99, 0.01, 0.00],
    [0.00, 0.00, 0.01, 0.00, 0.00, 0.99, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])
# ===============================================

class_names = ["GN", "CP", "FN", "AB", "TN", "BC", "TC"]
cms = [cm1, cm2, cm3, cm4]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 统一色阶范围，确保共享色条含义一致
vmin, vmax = 0.0, 1.0
mappable_for_cbar = None

for ax, cm in zip(axes.ravel(), cms):
    # 用 seaborn 画热力图（也可换成 imshow）
    hm = sns.heatmap(
        cm, ax=ax, cmap="Blues", vmin=vmin, vmax=vmax,
        cbar=False,  # 这里先不画独立色条
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.0, linecolor=None, annot=False
    )
    # 数值标注：非零显示到小数点后两位，零就显示“0”
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = "0" if abs(val) < 1e-12 else f"{val:.2f}"

            # 根据数值大小选择文字颜色
            text_color = "white" if val > 0.5 else "black"

            ax.text(j + 0.5, i + 0.5, text,
                    ha="center", va="center", fontsize=8, color=text_color)
    ax.tick_params(axis='x', labelrotation=0)
    mappable_for_cbar = hm.collections[0]  # 取一个 mappable 用于共享色条

plt.subplots_adjust(left=0.20, bottom=0.10, right=0.80, top=0.95, wspace=0.25, hspace=0.25)

# 关键：整张图底部放一条共享色阶
cbar = fig.colorbar(
    mappable_for_cbar,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.06,       # 色阶与子图群之间的距离
    fraction=0.03,  # 色阶厚度（相对高度）
)
cbar.set_label("")  # 需要的话可以写单位/说明

# 如果你有每个子图的标题（模型名/方法名），在这里设置：
axes[0,0].set_title("cnnnet")
axes[0,1].set_title("gnnnet")
axes[1,0].set_title("transformer")
axes[1,1].set_title("NeuroDynamicNet")

plt.show()
