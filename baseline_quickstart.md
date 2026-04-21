# 基线运行指引

## 安装
```bash
pip install lightgbm pandas numpy
```
(pandas/numpy 应该已经有了)

## 运行
```bash
python module3_baseline.py \
    --samples-dir out_samples \
    --graph-pkl   out_graph/graph.pkl \
    --out         baseline_reports
```

笔记本 CPU 上约 3-5 分钟。

## 输出
```
baseline_reports/
├── metrics.json            — 全部数字
├── per_fork.csv            — 按分叉度分层的 top-1
├── feature_importance.csv  — LightGBM 特征重要性
├── model_mark.txt          — LightGBM 分类器 (mark head)
└── model_dt.txt            — LightGBM 回归器 (dt head)
```

## 预期基线数字 (我在你这份数据上跑出来的)

| method        | top-1 | top-3 | top-5 | MRR   |
|---------------|-------|-------|-------|-------|
| freq prior    | 0.420 | 0.778 | 0.879 | 0.609 |
| **LightGBM**  | **0.635** | **0.802** | **0.836** | **0.729** |

dt 回归: median|err| ≈ 25 秒 (但 MAE 受长尾污染, 看中位数更可靠)

## 分层看 (LightGBM top-1):
- `n_legal ∈ [16, 300]` (97.7% 样本): **0.637**
- `n_legal ∈ [7, 15]`: 0.875 (样本少, 参考)
- `n_legal ∈ [4, 6]`: 0.706

## 读数提示

1. **真正要打败的基线 = 63.5% top-1**. SOTA (HGT+THP) 如果压不过 68-70%, 说明复杂度没回报.
2. **LightGBM 第 2 轮就早停** — 这不是 bug, 是特征信号量不够的证据. 拍扁到 tabular 丢了图结构.
3. **特征重要性: 时间周期 > 网络负载 > MOV 晚点 > berth/prev_pr**. `prev_pr` 排第 21 名 — 说明序列关系 tabular 抓不住. GNN+Transformer 架构的优势就在这里.
