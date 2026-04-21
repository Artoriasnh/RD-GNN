# Module 4 — GPU 训练指南

## 准备

### 1. 环境
```bash
# PyTorch + PyG (按你的 CUDA 版本调)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install lightgbm pandas numpy  # 基线评估会用到
```

### 2. 目录结构
把所有 m4 文件和 Module 1/2 的产出放一起：
```
your_working_dir/
├── m4_dataset.py
├── m4_model.py
├── m4_train.py
├── m4_explain.py
├── run_m4.sh
├── package_results.sh
├── out_samples/           # Module 2 产出
│   ├── samples.csv
│   ├── graph_states.npz
│   ├── event_windows.npz
│   ├── legal_masks.npz
│   ├── labels.npz
│   ├── mov_features.npz
│   └── vocab.pkl
└── out_graph/             # Module 1 产出
    └── graph.pkl
```

## 运行

### 第一步：冒烟测试（10 分钟）
**一定先跑这个**，确认代码在你的环境下能跑通。
```bash
bash run_m4.sh smoke
```
预期输出：
- 显示 GPU 信息
- 跑 500 个训练样本 1 epoch
- 输出一个 val top-1 数字（冒烟测试的数字不准，能出结果就行）

### 第二步：完整训练
```bash
bash run_m4.sh full
```

这会跑 5 epochs、d=128、batch=64、AMP（bfloat16）。

**预计时长**（按 GPU 算力）：
| GPU | 显存 | 预估时长 |
|---|---|---|
| RTX 3060 12GB | 12GB | ~2-3 小时 |
| RTX 3090 / 4080 | 24GB / 16GB | ~1-1.5 小时 |
| A100 | 40/80GB | ~30-45 分钟 |
| RTX 2060 / 2070 | 6-8GB | ~4-5 小时 |

**如果显存不够** (OOM)：
```bash
# 手动减 batch
python m4_train.py ... --batch-size 32
# 或者减模型
python m4_train.py ... --batch-size 32 --d 96
# 或者关 AMP (某些老 GPU 的 bf16 有问题)
python m4_train.py ... --batch-size 32  # 去掉 --amp
```

### 第三步：打包结果回传
```bash
bash package_results.sh m4_runs/20240421_143022_full
```
输出 `<timestamp>_full_results.tar.gz`（约 10-30 MB），把这个上传给我。

## 预期结果

| 指标 | baseline (LightGBM+图) | **M4 (预期)** |
|---|---|---|
| top-1 | 71.5% | **76-80%** |
| top-3 | 84.3% | **88-92%** |
| top-5 | 86.6% | **92-94%** |
| MRR | 0.79 | **0.83-0.86** |

**如果 1 epoch 后 val top-1 < 50%**：说明学习率或数据管线有问题，看下面的 troubleshoot。

## Troubleshooting

### OOM
- `--batch-size 32` 或更小
- `--d 96` 或更小
- 去掉 `--amp`（AMP 在某些显卡上反而吃更多显存）
- 关闭 `pin_memory`（改 m4_train.py 里 `pin=False`）

### 训练 loss 不降
- 检查日志里显示的 `[model] vocabs:` 数字是否和你 Module 2 一致（type=10, id=625, train=1317）
- 检查日志里 `mask_hit` 或样本总数是否合理
- 试 `--lr 1e-4`（可能默认学习率太大）

### GPU 利用率低（<30%）
- 加 `--num-workers 6` 或 8
- 显存够的话把 `--batch-size` 加到 128

### 日志看起来 loss 跳动很大
- 加大 `--warmup 1000`
- 梯度裁剪已经开启（1.0），如果还跳就 `--lr 1e-4`

## 运行模式一览

| 模式 | batch | d | epochs | 用途 |
|---|---|---|---|---|
| `smoke` | 16 | 64 | 1 (500样本) | 10分钟内验证代码能跑 |
| `small` | 64 | 64 | 2 | 1-2小时快速实验 |
| `full` | 64 | 128 | 5 | **默认，推荐值** |
| `large` | 128 | 192 | 10 | 最终论文数字（4-6小时） |

从 `full` 开始就够了。如果 `full` 结果好，再考虑 `large`。

## 结果里我要看什么

你训完回传给我的 `.tar.gz` 里有：
1. `best.pt` — 训练好的模型权重，我可以加载做解释分析
2. `history.json` — 每 epoch 的 val 指标 + 最终 test 指标 + 训练配置
3. `train.log` — 完整训练日志

我拿到这三个文件后会：
- 核对 test 指标是否超过基线 71.5%
- 用 `m4_explain.py` 对几个有代表性的 test 样本跑解释（L1规则/L2重要边/L3反事实/时间attention）
- 看 val/train top-1 曲线判断是否需要调参重训
- 根据 layered 报告看模型在"大分叉"决策点（n_legal ∈ [16,300]，97% 样本）的表现

## 一个实用提示

训练时用 `tmux` 或 `screen`，防止 SSH 断开训练也丢：
```bash
tmux new -s derby
# 进去之后:
bash run_m4.sh full 2>&1 | tee train.out
# Ctrl+B D 分离会话; 重新连接: tmux attach -t derby
```
