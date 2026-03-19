# NeuroDynamicNet：基于图神经网络的 EEG 癫痫发作分类框架

## 项目简介

NeuroDynamicNet 是一个面向 **EEG（脑电）癫痫发作分类** 的深度学习项目。项目以 TUH 类脑电数据处理流程为核心，支持多种模型进行对比实验，包括：

- **GGN**（Graph Generative Network）
- **EnhancedGGN**（带动态连接重要性加权的增强版 GGN）
- **ST-HGGN**（结合时空衰减与层次化图生成的模型）
- **DCRNN / gnnnet**
- **CNNNet / cnnnet**
- **Transformer**

代码同时包含数据加载、归一化、训练、测试、混淆矩阵绘制、F1 统计、可靠性分析（Top-1 / Top-2 ECE）、t-SNE 可视化以及多次重复实验统计等功能。项目主训练入口位于 `eeg_main.py`，并通过命令行参数统一管理实验配置。fileciteturn0file0 fileciteturn1file5 fileciteturn1file14

---

## 主要特性

- 支持 **多模型 EEG 癫痫分类实验**，便于进行基线与改进模型对比。fileciteturn1file5
- 支持 **图结构建模**，包括固定邻接图、随机图、潜在图生成等机制。fileciteturn1file5 fileciteturn0file1
- 提供 **增强型动态图建模模块**，包括：
  - 动态连接重要性加权（Dynamic Connection Weighting）
  - 时空衰减图生成（Spatio-Temporal Decay Graph）
  - 层次化图生成（Hierarchical Graph Generator）fileciteturn1file4 fileciteturn1file12 fileciteturn1file13
- 支持 **多次独立重复训练** 并统计均值与方差。fileciteturn1file18 fileciteturn1file14
- 自动输出 **损失曲线、混淆矩阵、可靠性图、t-SNE 可视化、metrics.json、预测概率文件**。fileciteturn1file17 fileciteturn1file11 fileciteturn1file19

---

## 项目结构

当前代码快照中，核心文件包括：

```text
.
├── eeg_main.py              # 主入口：训练 / 测试 / 数据生成 / 可视化
├── eeg_util.py              # 参数管理、数据加载器、图工具、评估工具
├── baseline_models.py       # DCRNN、CNN、Transformer 等基线模型
├── encoder_decoder.py       # 编码器 / 解码器 / 图池化等模块
├── ggn.py                   # GGN、EnhancedGGN、ST-HGGN 等核心模型
├── graph_conv_layer.py      # 图卷积与图池化基础层
└── ...                      # 数据、邻接矩阵、日志、模型权重等目录
```


---

## 任务说明

本项目聚焦 **EEG 癫痫发作类型分类**。当前代码中定义了 7 个类别标签：

- GNSZ
- CPSZ
- FNSZ
- ABSZ
- TNSZ
- BCKG
- TCSZ

并在训练与评估中使用对应的标签映射。fileciteturn0file0

---

## 数据格式

主流程默认读取预先保存好的 NumPy 数据文件：

- `seizure_x*_from_begin.npy`
- `seizure_y*_from_begin.npy`

在 `load_tuh_data()` 中，输入数据会被读取后打乱，并按类别进行分层划分，其中每一类约 **1/3 用于测试，2/3 用于训练**。随后数据从原始形状 `B, T, N, C` 转换为模型使用的 `B, C, N, T`。fileciteturn1file3

如果需要从原始频域数据重新生成样本，可通过 `generate_tuh_data()` 处理 pickle 文件，并导出 `.npy` 数据。该过程会按频段目录读取样本、过滤部分类别、对齐长度并拼接保存。fileciteturn0file0

> **说明**
> 
> 当前仓库代码默认数据集名称为 `TUH`，因此 README 中也按 TUH EEG 任务进行描述；实际使用时，请根据你本地的数据组织方式修改 `--data_path` 和相关文件路径。fileciteturn0file1

---

## 环境依赖

从代码可见，项目主要依赖以下 Python 库：

- Python 3.9+
- PyTorch
- torch-geometric
- NumPy
- Pandas
- SciPy
- scikit-learn
- matplotlib
- seaborn
- networkx
- tensorboard

这些依赖分别出现在训练主脚本、图模型、基线模型和工具函数中。fileciteturn0file0 fileciteturn0file1 fileciteturn0file2 fileciteturn0file3 fileciteturn0file4 fileciteturn0file5

建议创建 `requirements.txt`：

```txt
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
networkx
tensorboard
torch
torch-geometric
```

安装示例：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 训练模型

项目主入口在 `eeg_main.py`。默认情况下，若不启用 `--testing`，程序会保存参数并执行 `multi_train(args, runs=args.runs)`，即重复训练多次并统计结果。fileciteturn1file14

示例：

```bash
python eeg_main.py \
  --task EnhancedGGN \
  --data_path ./data/TUH \
  --adj_file ./adjs/raw_adj.npy \
  --adj_type origin \
  --best_model_save_path ./saved_models/enhancedggn.pt \
  --fig_filename ./figs/enhancedggn \
  --epochs 50 \
  --batch_size 32 \
  --lr 5e-5 \
  --runs 5 \
  --cuda \
  --lgg
```

### 2. 测试 / 可视化

当启用 `--testing` 时，程序会：

1. 加载测试数据；
2. 恢复指定权重；
3. 执行 t-SNE 可视化；
4. 将结果输出到 `fig_filename` 对应目录。fileciteturn1file11 fileciteturn1file19

示例：

```bash
python eeg_main.py \
  --task EnhancedGGN \
  --testing \
  --data_path ./data/TUH \
  --best_model_save_path ./saved_models/enhancedggn.pt \
  --fig_filename ./figs/test_run \
  --cuda
```

### 3. 生成预处理数据

```bash
python eeg_main.py \
  --task generate_data \
  --data_path ./data/TUH
```

该模式会调用 `generate_tuh_data()` 从原始特征文件生成训练所需的 `.npy` 文件。fileciteturn1file14 fileciteturn0file0

---

## 关键参数

项目参数定义集中在 `eeg_util.py:get_common_args()`，包括数据、训练、图结构和模型控制项。主要参数如下：fileciteturn0file1

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task` | 任务 / 模型名称，如 `GGN`、`EnhancedGGN`、`ST-HGGN`、`transformer`、`gnnnet`、`cnnnet` | `seizure` |
| `--dataset` | 数据集名称 | `TUH` |
| `--data_path` | 数据路径 | `./data/METR-LA` |
| `--adj_file` | 邻接矩阵文件路径 | `./adjs/raw_adj.npy` |
| `--adj_type` | 邻接矩阵类型 | `origin` |
| `--batch_size` | 批大小 | `32` |
| `--epochs` | 训练轮数 | `10` |
| `--lr` | 学习率 | `0.00005` |
| `--dropout` | dropout 比例 | `0.6` |
| `--predict_class_num` | 分类类别数 | `7` |
| `--feature_len` | 输入特征维度 | `48` |
| `--cuda` | 启用 GPU | 关闭 |
| `--testing` | 测试模式 | 关闭 |
| `--multi_train` | 多次训练 | 关闭 |
| `--lgg` | 启用潜在图生成 | 关闭 |
| `--use_dynamic_in_st_hggn` | 是否在 ST-HGGN 中启用动态连接加权 | 关闭 |

---

## 支持的模型

### 1. GGN

GGN 由编码器、图生成/图解码器和分类预测器组成。编码器支持 RNN、LSTM、CNN2d 等形式，解码部分支持图卷积、GAT-CNN、LGG-CNN 等模式。fileciteturn0file4 fileciteturn0file3

### 2. EnhancedGGN

EnhancedGGN 在 GGN 基础上加入 **Dynamic Connection Weighting**，将生成图与类别相关连接模式进行自适应融合，以增强判别能力。fileciteturn1file10 fileciteturn0file4

### 3. ST-HGGN

ST-HGGN 进一步结合：

- 基础 GGN 编码/解码；
- 时空衰减图生成器；
- 层次化图生成器；
- 可选动态连接加权模块。

其目标顺序在代码注释中明确为：
**GGN → 时空衰减图生成 → 层次化图生成 → （可选）动态连接加权**。fileciteturn1file7 fileciteturn1file13

### 4. 其他基线模型

项目还集成了以下可对比模型：

- `gnnnet`：DCRNN 分类模型
- `cnnnet`：CNNNet
- `transformer`：EEG Transformer

这些模型可通过 `chose_model()` 统一切换。fileciteturn1file5 fileciteturn0file2

---

## 输出结果

训练完成后，项目会自动保存以下结果：

- 最优模型权重（`best_model_save_path`）
- TensorBoard 日志
- loss 曲线图
- 混淆矩阵图
- 可靠性分析图（Top-1 / Top-2 ECE）
- t-SNE 可视化图
- `metrics.json`
- `*_test_proba_labels.npz`

对应逻辑位于 `train_eeg()`、测试分支以及指标计算相关函数中。fileciteturn0file0 fileciteturn1file17 fileciteturn1file11 fileciteturn1file19

---

## 复现实验建议

为便于复现实验，建议：

1. 固定随机种子；
2. 保存参数 JSON；
3. 使用 `--runs` 进行多次重复训练；
4. 同时报告 ACC、Macro-F1、Weighted-F1、Balanced Accuracy、Kappa、MCC、ROC-AUC、PR-AUC 等指标。fileciteturn1file14 fileciteturn1file18 fileciteturn1file6

---

## 注意事项

1. 代码中存在 `from models.xxx import ...` 的导入方式，因此上传 GitHub 时建议整理目录结构，确保 `models/` 包存在。fileciteturn0file0 fileciteturn0file3 fileciteturn0file4
2. 当前代码片段中引用了如 `reliability_utils`、`load_eeg_adj()` 等外部模块/函数，上传仓库前请确认这些文件已一并整理。fileciteturn0file0
3. 默认参数中的部分路径仍是占位路径，例如 `./data/METR-LA`，实际使用前需替换为你的 EEG 数据路径。fileciteturn0file1
4. 若你希望仓库更易于开源复现，建议额外补充：
   - `requirements.txt`
   - `LICENSE`
   - `data/README.md`
   - 示例邻接矩阵说明
   - 训练与测试脚本（如 `scripts/train.sh`）

---

## 引用

如果这个项目对你的研究有帮助，欢迎在论文或项目中引用，并注明该项目用于 EEG 癫痫发作分类与动态图建模实验。

---

## 致谢

部分基线模型实现参考了 DCRNN 相关开源实现；代码中也在 `baseline_models.py` 中保留了相关说明。fileciteturn0file2
