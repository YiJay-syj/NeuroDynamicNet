import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
file_path = 'F:/scientific_research/GGN-main/GGN-main/seizure_x_from_begin.npy'
data = np.load(file_path)

print(f"数据形状: {data.shape}")  # (7841, 139, 20, 48)
print(f"数据类型: {data.dtype}")

# 选择要可视化的样本和通道
sample_idx = 0  # 选择第一个样本
channel_names = [f'Channel_{i + 1}' for i in range(20)]  # 假设有20个通道


def plot_eeg_time_series(data, sample_idx=0, channels_to_plot=None, figsize=(15, 12)):
    """
    绘制脑电信号随时间变化的图形

    参数:
    - data: 四维脑电数据 (样本, 时间步, 通道, 特征)
    - sample_idx: 要绘制的样本索引
    - channels_to_plot: 要绘制的通道列表，None表示绘制所有通道
    - figsize: 图形大小
    """

    # 提取指定样本的数据
    sample_data = data[sample_idx]  # 形状: (139, 20, 48)

    # 如果未指定通道，使用所有通道
    if channels_to_plot is None:
        channels_to_plot = range(sample_data.shape[1])

    # 计算时间轴 (假设采样频率，您需要根据实际情况调整)
    # 这里使用时间步作为x轴，您可能需要根据实际采样率转换为真实时间
    time_steps = np.arange(sample_data.shape[0])

    # 创建图形
    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=figsize, sharex=True)
    if len(channels_to_plot) == 1:
        axes = [axes]

    # 为每个通道绘制时间序列
    for i, channel_idx in enumerate(channels_to_plot):
        # 提取该通道的所有特征并计算均值（或选择特定特征）
        channel_data = sample_data[:, channel_idx, :]  # 形状: (139, 48)

        # 方法1: 绘制所有特征的均值
        mean_signal = channel_data.mean(axis=1)

        # 方法2: 或者绘制第一个特征（根据您的需求选择）
        # mean_signal = channel_data[:, 0]

        axes[i].plot(time_steps, mean_signal, linewidth=1.0, color='blue', alpha=0.8)
        axes[i].set_ylabel(f'{channel_names[channel_idx]}\n信号强度', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(mean_signal.min() - 0.1, mean_signal.max() + 0.1)

    # 设置公共x轴标签
    axes[-1].set_xlabel('时间步', fontsize=12)

    plt.suptitle(f'脑电信号强度随时间变化 - 样本 {sample_idx + 1}', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()

    return fig


# 绘制所有通道
print("绘制所有通道的脑电信号...")
plot_eeg_time_series(data, sample_idx=0)


# 如果通道太多，可以选择部分通道显示
def plot_selected_channels(data, sample_idx=0, selected_channels=[0, 1, 2, 3, 4], figsize=(15, 10)):
    """
    绘制选定的几个通道
    """
    sample_data = data[sample_idx][:100]    #只取前100个时间步
    time_steps = np.arange(sample_data.shape[0])

    fig, axes = plt.subplots(len(selected_channels), 1, figsize=figsize, sharex=True)
    if len(selected_channels) == 1:
        axes = [axes]

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, channel_idx in enumerate(selected_channels):
        channel_data = sample_data[:, channel_idx, :]
        mean_signal = channel_data.mean(axis=1)

        axes[i].plot(time_steps, mean_signal, linewidth=1.2,
                     color=colors[i % len(colors)], alpha=0.8)
        axes[i].set_ylabel(f'通道{channel_idx + 1}\n信号强度', fontsize=10)
        axes[i].grid(True, alpha=0.3)

        # # 添加统计信息
        # axes[i].text(0.02, 0.95, f'均值: {mean_signal.mean():.3f}\n标准差: {mean_signal.std():.3f}',
        #              transform=axes[i].transAxes, fontsize=8,
        #              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[-1].set_xlabel('时间步', fontsize=12)
    # plt.suptitle(f'脑电信号强度随时间变化 - 样本 {sample_idx + 1} (选定通道)', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig("selected_channels.png")
    plt.show()


# 绘制选定的通道
print("绘制选定通道的脑电信号...")
plot_selected_channels(data, sample_idx=0, selected_channels=[0, 3, 9, 15])


# 单个通道的详细分析
def plot_single_channel_detailed(data, sample_idx=0, channel_idx=0, figsize=(15, 8)):
    """
    对单个通道进行详细分析
    """
    sample_data = data[sample_idx]
    time_steps = np.arange(sample_data.shape[0])
    channel_data = sample_data[:, channel_idx, :]

    # 计算均值信号
    mean_signal = channel_data.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # 上子图: 时间序列
    ax1.plot(time_steps, mean_signal, linewidth=1.5, color='blue', alpha=0.8)
    ax1.set_ylabel('信号强度', fontsize=12)
    ax1.set_title(f'通道 {channel_idx + 1} 脑电信号时间序列 - 样本 {sample_idx + 1}', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'统计信息:\n均值: {mean_signal.mean():.4f}\n标准差: {mean_signal.std():.4f}\n最大值: {mean_signal.max():.4f}\n最小值: {mean_signal.min():.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
             verticalalignment='top')

    # 下子图: 数据分布直方图
    ax2.hist(mean_signal, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('信号强度', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('信号强度分布直方图', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("single_channel_detailed.png")
    plt.show()

    # 打印详细统计信息
    print(f"\n=== 通道 {channel_idx + 1} 详细统计 ===")
    print(f"信号范围: [{mean_signal.min():.4f}, {mean_signal.max():.4f}]")
    print(f"均值: {mean_signal.mean():.4f}")
    print(f"标准差: {mean_signal.std():.4f}")
    print(f"中位数: {np.median(mean_signal):.4f}")


# 绘制单个通道的详细分析
print("绘制单个通道的详细分析...")
plot_single_channel_detailed(data, sample_idx=0, channel_idx=0)


# 多样本对比
def compare_multiple_samples(data, sample_indices=[0, 1, 2], channel_idx=0, figsize=(15, 10)):
    """
    比较不同样本的同一通道
    """
    # time_steps = np.arange(data.shape[1])
    time_steps = np.arange(100) #只取前100个时间步

    plt.figure(figsize=figsize)

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, sample_idx in enumerate(sample_indices):
        sample_data = data[sample_idx][:100]    #只取前100个时间步
        channel_data = sample_data[:, channel_idx, :]
        mean_signal = channel_data.mean(axis=1)

        plt.plot(time_steps, mean_signal, linewidth=1.2,
                 color=colors[i % len(colors)], alpha=0.7,
                 label=f'样本 {sample_idx + 1}')

    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('信号强度', fontsize=12)
    plt.title(f'不同样本的通道 {channel_idx + 1} 脑电信号对比', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 设置 y 轴的范围
    plt.ylim(-1, 1)

    plt.tight_layout()
    plt.savefig("compare_multiple.png")
    plt.show()


# 比较不同样本
print("比较不同样本的同一通道...")
compare_multiple_samples(data, sample_indices=[0,69,89,139], channel_idx=0)