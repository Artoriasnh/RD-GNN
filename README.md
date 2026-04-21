# Derby Signalling Data Pipeline — Module 1 & 2

两个模块都基于 **pandas / numpy**，不依赖 PyTorch。Module 1 的输出可以一行代码转成 PyG `HeteroData`（见 `graph_to_pyg.py`）。

## 文件清单

| 文件 | 作用 |
|---|---|
| `module1_build_graph.py` | 构建 Derby 静态异构图 |
| `module2_replay.py` | 事件流重放 + PR 样本抽取 |
| `graph_to_pyg.py` | 把 Module 1 产物转成 PyG HeteroData |

## 运行

```bash
# 1) 构建静态图
python module1_build_graph.py \
  --infra /path/to/route_to_tc_all.csv \
  --td    /path/to/withtrainid.csv \
  --out   ./out_graph

# 2) 抽样本 (全量约 10 分钟, 内存峰值 ~4GB)
python module2_replay.py \
  --graph ./out_graph/graph.pkl \
  --td    /path/to/withtrainid.csv \
  --mov   /path/to/mov_91_127.csv \
  --out   ./out_samples \
  --K     64
```

## 产出的数据结构

### Module 1 → `out_graph/graph.pkl`

```python
{
  'node_ids':  {'berth':[...], 'tc':[...], 'route':[...], 'signal':[...]},
  'id_to_idx': {'berth':{id:idx}, 'tc':..., 'route':..., 'signal':...},
  'node_features': {
      'berth':  (167, 4)  float32,   # [is_platform, is_yard, is_numeric4, is_special]
      'tc':     (276, k)  float32,   # one-hot 区域码
      'route':  (277, 22) float32,   # one-hot(区域) + one-hot(kind) + [n_tc, parse_ok]
      'signal': (115, k)  float32,   # one-hot 区域码
  },
  'edge_index': {                     # 8 种边类型
      ('berth','starts','route'):     (2, 279),
      ('route','ends','berth'):       (2, 290),
      ('route','covers','tc'):        (2, 1702),
      ('tc','covered_by','route'):    (2, 1702),
      ('berth','step','berth'):       (2, 386),
      ('tc','adj','tc'):              (2, 548),
      ('signal','protects','berth'):  (2, 115),
      ('berth','protected_by','signal'): (2, 115),
  },
  'legal_routes_from_berth': {berth_str: [route_idx, ...]},   # 查找表
  'route_tcs':       {route_idx: [tc_str, ...]},
  'route_end_berth': {route_idx: berth_str},
  'berth_reach':     {berth_str: set(berth_str)},   # 12-hop 前向可达
}
```

### Module 2 → `out_samples/*`

每个 PR 事件对应一个样本（约 180k 条）。**所有 npz 文件的第 0 维 = 样本索引，已对齐**。

| 文件 | shape | 说明 |
|---|---|---|
| `samples.csv` | 180k × 9 | 元数据（time, trainid, current_berth, pr_route_id, pr_route_idx, mask_includes_label, n_legal, dt_prev_sec, sample_idx） |
| `graph_states.npz` | berth:(N,167,5), tc:(N,276,2), route:(N,277,2), signal:(N,115,2) | 每个样本的**动态**节点特征快照 |
| `event_windows.npz` | X:(N,64,4) | 每样本过去 64 个事件的 token (type_idx, id_idx, train_idx, rel_time_norm) |
| `legal_masks.npz` | M:(N,277) bool | 合法候选集掩码 |
| `labels.npz` | mark:(N,) int64, dt_log:(N,) float32 | A 标签 + B 标签 (log(1+Δt秒)) |
| `mov_features.npz` | X:(N,23) float32 | 焦点车在 PR 时刻最近一条 MOV 的上下文 |
| `vocab.pkl` | — | 事件 token 的字典 (type, id, train 三个) |

## 训练时如何拼起来

伪代码（PyTorch）：

```python
import numpy as np, pickle, torch
from torch.utils.data import Dataset

class DerbyPR(Dataset):
    def __init__(self, sample_dir, graph_pkl):
        gs = np.load(f'{sample_dir}/graph_states.npz')
        self.gs_b, self.gs_tc, self.gs_r, self.gs_s = gs['berth'], gs['tc'], gs['route'], gs['signal']
        self.ev = np.load(f'{sample_dir}/event_windows.npz')['X']
        self.mask = np.load(f'{sample_dir}/legal_masks.npz')['M']
        lab = np.load(f'{sample_dir}/labels.npz')
        self.mark, self.dt = lab['mark'], lab['dt_log']
        self.mov = np.load(f'{sample_dir}/mov_features.npz')['X']
        # 静态图特征 + 边 (一次加载, 所有样本共享)
        g = pickle.load(open(graph_pkl,'rb'))
        self.static = {k: torch.tensor(v) for k,v in g['node_features'].items()}
        self.edges = {k: torch.tensor(v) for k,v in g['edge_index'].items()}

    def __len__(self): return len(self.mark)

    def __getitem__(self, i):
        # 动态特征 concat 到静态特征
        x_dict = {
            'berth':  torch.cat([self.static['berth'],  torch.tensor(self.gs_b[i])],  dim=-1),
            'tc':     torch.cat([self.static['tc'],     torch.tensor(self.gs_tc[i])], dim=-1),
            'route':  torch.cat([self.static['route'],  torch.tensor(self.gs_r[i])],  dim=-1),
            'signal': torch.cat([self.static['signal'], torch.tensor(self.gs_s[i])], dim=-1),
        }
        return dict(
            x_dict=x_dict,
            edge_index_dict=self.edges,                       # 所有样本共享同一个图
            events=torch.tensor(self.ev[i]),                  # (K, 4)
            mov=torch.tensor(self.mov[i]),                    # (23,)
            legal_mask=torch.tensor(self.mask[i]),            # (277,) bool
            label_mark=int(self.mark[i]),                     # scalar
            label_dt=float(self.dt[i]),                       # scalar
        )
```

## 关键设计决定 / 已知事项

**1. 合法性定义 — 重要**
原本以为"合法候选 = 当前 berth 出发的 route"。数据显示信号员会**提前为整条路径批量设进路**（例：车在 APUT，10 秒内连发 5328/5300/5061/5075/5083/5093/5101 共 7 条 PR）。因此合法性改为：
- 从 `current_berth` **前向 BFS 12 跳**可达的所有 berth 出发的 route
- 减去其 TC 被**他车**占用的 route（自车占着是允许的）
- 实测 `mask_hit`（标签在掩码内）从 11% → 96.4%

**2. current_berth 推断**
从每辆车最近一次 `CA/CC` 事件的 `to_berth` 推，99.5% 的 PR 能推到。剩下 0.5% 是"焦点车刚进入 Derby 区域、尚无 berth 事件"的冷启动情况，合法掩码置全 0，训练时应作为特殊样本处理。

**3. tc_occupier 启发式**
TC 占用者 = 最近一个 `type=Track, state=1` 事件的 `trainid_filled`。这不完全精确（Track 事件偶尔没有 trainid），但足够 98% 的 case 用。

**4. Event token 的 Track 降采样**
Track 事件占 TD 的 56%（2.17M / 3.87M），全部塞进 token 窗口会把 PR 事件挤掉。默认 `--K 64` 时 Track 不入 token（`downsample_track_in_tokens=True`），但 TC 状态仍会更新。如果要改回全保留，编辑 `main()` 末尾那段 `if downsample_track_in_tokens and typ == 'Track': continue`。

**5. Route state 的业务语义**
数据里 `type=Route, state=0/1` 的语义我按"1=已锁定"处理。如果你在现场或 Network Rail 文档确认是相反（0=已设/锁定，1=已解锁），把 `update_state` 里 Route 分支的赋值反一下就行——不影响流水线的其它部分。

**6. 稀有 PR 标签**
237 种 PR 里有长尾。`samples.csv` 里可以按 `pr_route_idx` 做频次统计，把出现 <50 次的合并成 `<RARE>` 类。模型端 `num_classes = 277 -> ~150` 会更稳。

**7. 数据切分**
98 天按时间切：
```python
train = samples[samples['time'] <  '2023-11-16']   # 76 天
val   = samples[(samples['time']>='2023-11-16') & (samples['time']<'2023-12-01')]  # 15 天
test  = samples[samples['time'] >= '2023-12-01']   # 7 天
```
**不要随机切**，否则同一列车的事件会污染。

## 从这里可以接的下一步

1. **基线** — 频率先验（计数表 `P(PR|prev_PR, current_berth, hour)`）+ LightGBM 多分类，用掩码限制到合法集内。拿到一个 top-1 / top-3 基线数字。
2. **SOTA 模型** — HGT（空间）+ THP（时间）+ 双头（mark + log-normal mixture 强度）。参考我们上一轮讨论的架构图。
3. **解释性** — GNNExplainer 对 HGT 做子图解释；THP attention 做时间解释；合法性掩码本身对"为何不选 X"提供规则级解释。
