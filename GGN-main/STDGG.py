import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色方案
colors = {
    'input': '#6BAED6',  # 蓝色
    'temporal': '#74C476',  # 绿色
    'similarity': '#FD8D3C',  # 橙色
    'decay': '#FB6A4A',  # 红色
    'output': '#9E9AC8'  # 紫色
}

# ============ 1. 输入层 ============
# 3D立方体效果的输入数据
input_box = FancyBboxPatch((0.5, 3.5), 2, 3,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['input'],
                           facecolor=colors['input'],
                           alpha=0.3, linewidth=2)
ax.add_patch(input_box)

# 绘制3D效果
ax.plot([0.5, 0.8], [6.5, 6.8], 'k-', linewidth=1.5)
ax.plot([2.5, 2.8], [6.5, 6.8], 'k-', linewidth=1.5)
ax.plot([2.5, 2.8], [3.5, 3.8], 'k-', linewidth=1.5)
ax.plot([0.8, 2.8], [6.8, 6.8], 'k-', linewidth=1.5)
ax.plot([0.8, 2.8], [3.8, 3.8], 'k-', linewidth=1.5)
ax.plot([2.8, 2.8], [3.8, 6.8], 'k-', linewidth=1.5)

ax.text(1.5, 5.5, r'$\mathbf{X}$', fontsize=16, ha='center', weight='bold')
ax.text(1.5, 5.0, r'$\mathbb{R}^{N \times T \times F}$', fontsize=11, ha='center')
ax.text(1.5, 4.5, 'N: 节点数', fontsize=9, ha='center')
ax.text(1.5, 4.2, 'T: 时间步', fontsize=9, ha='center')
ax.text(1.5, 3.9, 'F: 特征维度', fontsize=9, ha='center')
ax.text(1.5, 7.2, '输入神经信号', fontsize=12, ha='center', weight='bold', color=colors['input'])

# ============ 2. 时间编码模块 ============
tcn_x = 3.5
# TCN层 1
for i, (k, d) in enumerate([(3, 1), (3, 2), (3, 4), (3, 8)]):
    y_pos = 6.5 - i * 0.8
    tcn_box = FancyBboxPatch((tcn_x, y_pos - 0.3), 2.2, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors['temporal'],
                             facecolor=colors['temporal'],
                             alpha=0.2 + i * 0.1, linewidth=2)
    ax.add_patch(tcn_box)
    ax.text(tcn_x + 1.1, y_pos, f'TCN Layer {i + 1}', fontsize=10, ha='center', weight='bold')
    ax.text(tcn_x + 0.3, y_pos - 0.15, f'k={k}', fontsize=8, ha='left')
    ax.text(tcn_x + 0.9, y_pos - 0.15, f'd={d}', fontsize=8, ha='left')
    ax.text(tcn_x + 1.5, y_pos - 0.15, f'→F{i + 1}', fontsize=8, ha='left')

# 位置编码
pe_box = FancyBboxPatch((tcn_x + 2.5, 6), 1.2, 0.8,
                        boxstyle="round,pad=0.05",
                        edgecolor='gold',
                        facecolor='gold',
                        alpha=0.3, linewidth=2, linestyle='--')
ax.add_patch(pe_box)
ax.text(tcn_x + 3.1, 6.5, '位置编码', fontsize=9, ha='center', weight='bold')
ax.text(tcn_x + 3.1, 6.2, 'PE(t)', fontsize=8, ha='center', style='italic')

# 输出
ax.text(tcn_x + 1.1, 3.2, r'$\mathbf{H}^{(L)}$', fontsize=14, ha='center', weight='bold')
ax.text(tcn_x + 1.1, 2.9, r'$\mathbb{R}^{N \times T \times D_h}$', fontsize=10, ha='center')

ax.text(tcn_x + 1.1, 7.5, '时间编码器', fontsize=12, ha='center', weight='bold', color=colors['temporal'])

# 箭头：输入到TCN
arrow1 = FancyArrowPatch((2.5, 5), (3.5, 6.2),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2.5, color='black')
ax.add_patch(arrow1)

# ============ 3. 相似度计算模块 ============
sim_x = 7.5
# 时间切片提取
slice_box = FancyBboxPatch((sim_x - 0.5, 5.5), 1.5, 1.2,
                           boxstyle="round,pad=0.05",
                           edgecolor=colors['similarity'],
                           facecolor=colors['similarity'],
                           alpha=0.2, linewidth=2)
ax.add_patch(slice_box)
ax.text(sim_x + 0.25, 6.3, '提取时刻 t', fontsize=10, ha='center', weight='bold')
ax.text(sim_x + 0.25, 5.9, r'$\mathbf{h}_i^t, \mathbf{h}_j^t$', fontsize=10, ha='center')

# 双线性变换
bilinear_box = FancyBboxPatch((sim_x + 1.5, 5.5), 2, 1.2,
                              boxstyle="round,pad=0.05",
                              edgecolor=colors['similarity'],
                              facecolor=colors['similarity'],
                              alpha=0.3, linewidth=2)
ax.add_patch(bilinear_box)
ax.text(sim_x + 2.5, 6.3, '双线性注意力', fontsize=10, ha='center', weight='bold')
ax.text(sim_x + 2.5, 6.0, r'$s_{ij}^t = \frac{(\mathbf{h}_i^t)^T \mathbf{W}_s \mathbf{h}_j^t}{\sqrt{D_h}}$',
        fontsize=9, ha='center')
ax.text(sim_x + 2.5, 5.7, r'$\mathbf{W}_s$: 可学习', fontsize=8, ha='center', style='italic')

# 多头注意力（可选）
multihead_box = FancyBboxPatch((sim_x + 1.5, 4.0), 2, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor=colors['similarity'],
                               facecolor='white',
                               alpha=0.5, linewidth=1.5, linestyle='--')
ax.add_patch(multihead_box)
ax.text(sim_x + 2.5, 4.4, '多头机制(可选)', fontsize=8, ha='center', style='italic')

# 相似度矩阵
sim_matrix = FancyBboxPatch((sim_x + 1.8, 2.8), 1.4, 1.4,
                            boxstyle="round,pad=0.02",
                            edgecolor=colors['similarity'],
                            facecolor=colors['similarity'],
                            alpha=0.4, linewidth=2)
ax.add_patch(sim_matrix)
ax.text(sim_x + 2.5, 3.5, r'$\mathbf{S}^t$', fontsize=13, ha='center', weight='bold')
ax.text(sim_x + 2.5, 3.1, '相似度矩阵', fontsize=8, ha='center')

ax.text(sim_x + 2, 7.5, '相似度计算', fontsize=12, ha='center', weight='bold', color=colors['similarity'])

# 箭头：TCN到相似度
arrow2 = FancyArrowPatch((5.7, 4.5), (7, 6.1),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2.5, color='black')
ax.add_patch(arrow2)

# ============ 4. 时间衰减与稀疏化 ============
decay_x = 11
# 衰减函数曲线
decay_box = FancyBboxPatch((decay_x - 0.5, 5.2), 2.5, 2,
                           boxstyle="round,pad=0.08",
                           edgecolor=colors['decay'],
                           facecolor='white',
                           alpha=0.8, linewidth=2)
ax.add_patch(decay_box)

# 绘制衰减曲线
t_vals = np.linspace(0, 3, 50)
decay_vals = np.exp(-0.8 * t_vals)
t_scaled = decay_x + 0.2 + t_vals * 0.6
decay_scaled = 5.5 + decay_vals * 1.4
ax.plot(t_scaled, decay_scaled, color=colors['decay'], linewidth=2.5)
ax.text(decay_x + 1.2, 7.0, '时间衰减', fontsize=10, ha='center', weight='bold')
ax.text(decay_x + 1.2, 6.6, r'$w(t,t\')= e^{-\lambda |t-t\'|}$', fontsize=9, ha='center')
ax.text(decay_x + 1.2, 5.4, r'$|t-t\'|$', fontsize=8, ha='center')
ax.arrow(decay_x + 0.3, 5.45, 1.8, 0, head_width=0.08, head_length=0.1, fc='gray', ec='gray')

# 稀疏化操作
sparse_box = FancyBboxPatch((decay_x - 0.5, 2.5), 2.5, 2,
                            boxstyle="round,pad=0.08",
                            edgecolor=colors['decay'],
                            facecolor=colors['decay'],
                            alpha=0.2, linewidth=2)
ax.add_patch(sparse_box)
ax.text(decay_x + 1.2, 4.2, '稀疏化处理', fontsize=10, ha='center', weight='bold')
ax.text(decay_x + 1.2, 3.8, 'Top-K / 阈值化', fontsize=9, ha='center')
ax.text(decay_x + 1.2, 3.4, '对称化', fontsize=9, ha='center')
ax.text(decay_x + 1.2, 3.0, '归一化', fontsize=9, ha='center')
ax.text(decay_x + 1.2, 2.7, r'$\bar{\mathbf{A}}^t$', fontsize=11, ha='center', style='italic')

ax.text(decay_x + 1.2, 7.8, '时间衰减与稀疏化', fontsize=12, ha='center', weight='bold', color=colors['decay'])

# 箭头：相似度到衰减
arrow3 = FancyArrowPatch((9.5, 3.5), (10.5, 6),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2.5, color='black')
ax.add_patch(arrow3)

# ============ 5. 输出：时空图序列 ============
output_x = 14
# 输出邻接矩阵
output_box = FancyBboxPatch((output_x - 0.3, 4.5), 1.8, 2.5,
                            boxstyle="round,pad=0.08",
                            edgecolor=colors['output'],
                            facecolor=colors['output'],
                            alpha=0.3, linewidth=2.5)
ax.add_patch(output_box)

# 热力图效果
heatmap = Rectangle((output_x, 5.2), 1.2, 1.2,
                    facecolor=colors['output'], alpha=0.6, edgecolor='black', linewidth=1)
ax.add_patch(heatmap)
for i in range(4):
    for j in range(4):
        alpha_val = 0.2 + 0.2 * np.random.rand()
        small_rect = Rectangle((output_x + i * 0.3, 5.2 + j * 0.3), 0.3, 0.3,
                               facecolor=colors['output'], alpha=alpha_val,
                               edgecolor='white', linewidth=0.5)
        ax.add_patch(small_rect)

ax.text(output_x + 0.6, 6.6, r'$\bar{\mathbf{A}}^t$', fontsize=14, ha='center', weight='bold')
ax.text(output_x + 0.6, 5.0, '时刻 t', fontsize=9, ha='center', style='italic')
ax.text(output_x + 0.6, 7.5, '时变邻接矩阵', fontsize=12, ha='center', weight='bold', color=colors['output'])

# 时间轴展开
timeline_y = 1.5
ax.plot([1, 15], [timeline_y, timeline_y], 'k-', linewidth=2)
ax.text(0.5, timeline_y, '时间轴:', fontsize=10, ha='right', weight='bold')

for t_idx, t_pos in enumerate([3, 6, 9, 12]):
    # 时间点标记
    ax.plot(t_pos, timeline_y, 'ko', markersize=8)
    ax.text(t_pos, timeline_y - 0.3, f't={t_idx + 1}', fontsize=9, ha='center')

    # 小网络图
    if t_idx < 3:
        for node_i in range(3):
            theta = node_i * 2 * np.pi / 3
            x_node = t_pos + 0.3 * np.cos(theta)
            y_node = timeline_y + 0.8 + 0.3 * np.sin(theta)
            circle = plt.Circle((x_node, y_node), 0.08, color=colors['output'], alpha=0.6)
            ax.add_patch(circle)

        # 连线
        ax.plot([t_pos + 0.3, t_pos - 0.15], [timeline_y + 0.8 + 0.26, timeline_y + 0.8 - 0.15],
                color=colors['output'], alpha=0.4, linewidth=1.5)
        ax.plot([t_pos - 0.15, t_pos - 0.15], [timeline_y + 0.8 - 0.15, timeline_y + 0.8 + 0.26],
                color=colors['output'], alpha=0.4, linewidth=1.5)

ax.text(14, timeline_y + 0.5, r'$\{\mathcal{G}^1, \mathcal{G}^2, ..., \mathcal{G}^T\}$',
        fontsize=11, ha='left', style='italic')

# 箭头：衰减到输出
arrow4 = FancyArrowPatch((13, 3.5), (13.7, 5.5),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2.5, color='black')
ax.add_patch(arrow4)

# ============ 标题 ============
ax.text(8, 9.3, 'STDGG: 时空衰减图生成器架构',
        fontsize=18, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

# ============ 图例 ============
legend_elements = [
    mpatches.Patch(facecolor=colors['input'], edgecolor=colors['input'],
                   label='输入数据', alpha=0.5),
    mpatches.Patch(facecolor=colors['temporal'], edgecolor=colors['temporal'],
                   label='时间编码', alpha=0.5),
    mpatches.Patch(facecolor=colors['similarity'], edgecolor=colors['similarity'],
                   label='相似度计算', alpha=0.5),
    mpatches.Patch(facecolor=colors['decay'], edgecolor=colors['decay'],
                   label='时间衰减', alpha=0.5),
    mpatches.Patch(facecolor=colors['output'], edgecolor=colors['output'],
                   label='输出图', alpha=0.5),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('STDGG_architecture.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('STDGG_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

print("STDGG架构图已保存为 'STDGG_architecture.pdf' 和 'STDGG_architecture.png'")
