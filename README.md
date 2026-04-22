# 方案 A+C+E 联合：使用指南

## 三个方案在做什么

- **方案 A**：按 `(area, headcode_class)` 组合加权 loss——弱势类（EC 段、空车）拿更大 loss 权重
- **方案 C**：事件窗 K 从 64 → 128，给模型更长的历史上下文
- **方案 E**：MOV 特征从 23 维 → 35 维，加入 `minutes_until_next_planned`、`next_report_stanox`、`terminated`、`offroute` 等信号

## 变更的文件

| 文件 | 改动 |
|---|---|
| `module2_replay.py` | MovContext 扩到 35 维；`--K` 默认 128 |
| `m4_dataset.py` | 曝露每样本的 `combo_idx (area\|hc)` |
| `m4_train.py` | 自动推断 mov_dim 和 K；加 `--class-weights` 开关；loss 加权 |
| `m4_model.py` | 不变（原生支持通过参数传 mov_dim 和 K）|
| `m4_eval.py` | 兼容新 ckpt 的 mov_dim/K 字段 |
| `m4_explain.py` | 不变 |

## 执行步骤

### 第 1 步：覆盖文件

把 `plan_ACE/` 里的 7 个文件覆盖到你项目里（替换旧版本）。

### 第 2 步：重跑 Module 2（生成新样本 v2）

```powershell
python module2_replay.py `
    --graph out_graph\graph.pkl `
    --td    withtrainid.csv `
    --mov   mov_91_127.csv `
    --out   out_samples_v2 `
    --K     128
```

预计 **10-15 分钟**。产物：`out_samples_v2/` 下的 7 个文件（.npz + samples.csv + vocab.pkl）。
- `event_windows.npz` 会翻倍（~360 MB）
- `mov_features.npz` 会略增（~24 MB）
- 其他文件大小不变

### 第 3 步：转成 .npy（加速）

```powershell
python convert_npz_to_npy.py --samples-dir out_samples_v2
```

预计 **30 秒**。这步之后 DataLoader 用 mmap 读 .npy。

### 第 4 步：训练 A+C+E 联合版

```powershell
python m4_train.py `
    --samples-dir out_samples_v2 `
    --graph-pkl   out_graph\graph.pkl `
    --out         m4_runs\v2_ACE `
    --batch-size  32 `
    --epochs      10 `
    --d           128 `
    --hgt-layers  2 `
    --tf-layers   3 `
    --heads       4 `
    --lr          3e-4 `
    --warmup      500 `
    --weight-decay 0.05 `
    --time-weight 0.3 `
    --amp `
    --class-weights `
    --num-workers 0
```

### 第 5 步：评估

```powershell
python m4_eval.py `
    --ckpt m4_runs\v2_ACE\best.pt `
    --samples-dir out_samples_v2 `
    --graph-pkl   out_graph\graph.pkl `
    --out eval_reports\v2_ACE
```

## 预期收益（基于分析）

| 项 | 当前 (v1, 79.1%) | 预期 (v2 A+C+E) |
|---|---|---|
| Overall top-1 | 79.1% | **81-83%** |
| EC 段 top-1 | 42.9% | **60-70%** ← 方案 A 的直接目标 |
| 空车 top-1 | 56.9% | **70%+** ← 方案 A 的直接目标 |
| MRR | 0.852 | **0.87+** |
| Time NLL | 5.83 | **5.5-5.7** ← 方案 E 的新 MOV 特征帮时间预测 |

**警告**：如果 `--class-weights` 没开、只开了 C+E，提升会在 +1-2%。**必须加 `--class-weights` 才能激活方案 A**。

## 训练日志里会看到什么

启用 `--class-weights` 后，训练开始时会打印每个 `(area, headcode)` 组合的样本数和权重。例如：

```
[class-weights] median_n=3412  clip=[0.5,3.0]
[class-weights] per (area|hc) weights (train):
  DC|passenger           n=29844  w=0.500
  DW|passenger           n=22183  w=0.500
  TD|passenger           n=19576  w=0.500
  ...
  EC|freight             n=487    w=3.000  ← 拉满权重的小样本组合
  OTHER|empty_ecs        n=78     w=3.000
```

这就是方案 A 的核心——**样本少的组合权重被拉到 3.0**，让 model 在这些难例上 loss 贡献翻倍。

## Sanity check

训练开始时留意这三行：
1. `[model] inferred from data: mov_dim=35  K=128` — 确认 v2 特征生效
2. `n_combos=24` 左右 — 确认 combo 被正确枚举
3. 某个弱势 combo 权重=3.0 — 确认加权起效

如果看到 `mov_dim=23` 或 `K=64`，说明用的还是旧样本，去确认 `--samples-dir` 指向新目录。

## 如果要回退（控制实验）

想做对照实验，看 A/C/E 哪个贡献大？

- **只测 E（不加权，K 保持 64）**：重跑 Module 2 用 `--K 64`；训练不加 `--class-weights`。MOV 新特征的独立贡献。
- **只测 C（不加权，默认 MOV）**：改 `module2_replay.py` 里 `use_v2=False`；`--K 128`。K 扩大的独立贡献。
- **只测 A**：用 v1 数据（mov_dim=23, K=64），训练加 `--class-weights`。加权 loss 的独立贡献。

但这需要 3 次额外训练（每次 30-45 分钟），如果时间有限，直接上 A+C+E 联合是合理的。

## 时长预估

- Module 2 重跑：~15 min
- 训练 10 epochs：**~30-40 min**（K=128 使每步比 v1 慢 ~30%，但仍在你 RTX 5070 的 GPU 能力内）
- Eval：2-3 min

整套流程 **约 50-60 分钟**。
