"""
Module 4.1: Dataset / DataLoader
================================
把 Module 2 产出的 npz/csv 封装成 PyTorch Dataset.

设计要点:
  1) 静态图 (HeteroData) 全局共享, 不在样本里重复拷贝.
  2) 每个样本返回: dynamic 节点特征 (4 张 tensor) + event window + mov + legal_mask + labels.
  3) 训练期可以选择 "保留 label 不在 mask 里的样本" (给这些样本把 label 处 mask=1).
     测试期严格按原 mask 评估.

用法:
    from m4_dataset import DerbyDataset, load_graph_hetero
    g_hetero, meta = load_graph_hetero('out_graph/graph.pkl')
    ds = DerbyDataset('out_samples', split='train',
                       train_end='2023-11-16', val_end='2023-12-01',
                       relax_mask_in_train=True)
    sample = ds[0]  # dict
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_graph_hetero(graph_pkl):
    """把 Module 1 的 pkl 转成 PyG HeteroData + 元数据字典."""
    from torch_geometric.data import HeteroData
    g = pickle.load(open(graph_pkl, 'rb'))
    h = HeteroData()
    for nt, feat in g['node_features'].items():
        h[nt].x = torch.tensor(feat, dtype=torch.float32)
        h[nt].num_nodes = feat.shape[0]
    for et, ei in g['edge_index'].items():
        h[et].edge_index = torch.tensor(ei, dtype=torch.long).contiguous()
    meta = {k: g[k] for k in ['node_ids', 'id_to_idx',
                              'legal_routes_from_berth',
                              'route_tcs', 'route_end_berth',
                              'berth_reach']}
    return h, meta


class DerbyDataset(Dataset):
    """
    一个样本 = 一次 PR 事件. 返回:
        'dyn_berth':    (N_B, 5)  float
        'dyn_tc':       (N_T, 2)
        'dyn_route':    (N_R, 2)
        'dyn_signal':   (N_S, 2)
        'events':       (K, 4)    前 3 维 long, 第 4 维 float
        'mov':          (23,)     float
        'legal_mask':   (N_R,)    bool
        'label_mark':   scalar long
        'label_dt':     scalar float   log(1 + Δt sec)
        'n_legal':      scalar long    仅用于评估分层

    Windows 兼容: npz 文件句柄 lazy-open (Windows 用 spawn 启动 DataLoader worker,
    主进程持有的文件句柄无法 pickle 传给子进程, 这里改为每个进程首次访问时才打开).
    """
    def __init__(self, samples_dir, split='train',
                 train_end='2023-11-16', val_end='2023-12-01',
                 relax_mask_in_train=True):
        assert split in ('train', 'val', 'test')
        self.dir = samples_dir
        self.split = split
        self.relax = relax_mask_in_train and (split == 'train')

        sp = pd.read_csv(os.path.join(samples_dir, 'samples.csv'))
        sp['time'] = pd.to_datetime(sp['time'])
        t = sp['time'].values
        tr = t < np.datetime64(train_end)
        va = (t >= np.datetime64(train_end)) & (t < np.datetime64(val_end))
        te = t >= np.datetime64(val_end)
        if split == 'train':
            self.idx = np.where(tr)[0]
        elif split == 'val':
            self.idx = np.where(va)[0]
        else:
            self.idx = np.where(te)[0]

        # 不在 __init__ 里打开 npz (无法 pickle 到 worker 进程)
        # 改为 lazy: 首次 __getitem__ 时每个进程自己打开
        self._gs = None
        self._ev = None
        self._mk = None
        self._mov = None

        # labels 和 meta 是小数据, 预加载到内存没问题
        lb = np.load(os.path.join(samples_dir, 'labels.npz'))
        self.mark = lb['mark']
        self.dt = lb['dt_log']
        self.mask_includes_label = sp['mask_includes_label'].values
        self.n_legal = sp['n_legal'].values
        self.vocab = pickle.load(open(os.path.join(samples_dir, 'vocab.pkl'), 'rb'))

        print(f"[dataset/{split}] {len(self.idx)} samples  "
              f"relax_mask_in_train={self.relax}")

    def _ensure_open(self):
        """每个 worker 进程首次访问时调用, 打开 npz 句柄."""
        if self._gs is None:
            self._gs = np.load(os.path.join(self.dir, 'graph_states.npz'), mmap_mode='r')
            self._ev = np.load(os.path.join(self.dir, 'event_windows.npz'), mmap_mode='r')['X']
            self._mk = np.load(os.path.join(self.dir, 'legal_masks.npz'), mmap_mode='r')['M']
            self._mov = np.load(os.path.join(self.dir, 'mov_features.npz'), mmap_mode='r')['X']

    def __getstate__(self):
        """pickle 时去掉文件句柄 — Windows DataLoader 需要."""
        state = self.__dict__.copy()
        for k in ('_gs', '_ev', '_mk', '_mov'):
            state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        self._ensure_open()
        k = self.idx[i]
        dyn_berth = torch.from_numpy(np.asarray(self._gs['berth'][k]).copy())
        dyn_tc = torch.from_numpy(np.asarray(self._gs['tc'][k]).copy())
        dyn_route = torch.from_numpy(np.asarray(self._gs['route'][k]).copy())
        dyn_signal = torch.from_numpy(np.asarray(self._gs['signal'][k]).copy())
        ev = np.asarray(self._ev[k]).copy()
        events_cat = torch.from_numpy(ev[:, :3]).long()   # type, id, train
        events_t = torch.from_numpy(ev[:, 3:4]).float()    # rel_time_norm
        mov = torch.from_numpy(np.asarray(self._mov[k]).copy()).float()
        mask = torch.from_numpy(np.asarray(self._mk[k]).copy()).bool()
        mark = int(self.mark[k])
        dt = float(self.dt[k])

        # 训练期放宽: 如果 label 不在 mask 里, 把 label 那一位置 True
        if self.relax and not self.mask_includes_label[k]:
            mask = mask.clone()
            mask[mark] = True

        return dict(
            dyn_berth=dyn_berth, dyn_tc=dyn_tc,
            dyn_route=dyn_route, dyn_signal=dyn_signal,
            events_cat=events_cat, events_t=events_t,
            mov=mov, legal_mask=mask,
            label_mark=torch.tensor(mark, dtype=torch.long),
            label_dt=torch.tensor(dt, dtype=torch.float32),
            n_legal=torch.tensor(int(self.n_legal[k]), dtype=torch.long),
        )


def collate_fn(batch):
    """标准 default_collate 对 dict 有效, 只需堆叠 tensor."""
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


if __name__ == '__main__':
    # 快速冒烟测试
    import sys
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else 'out_samples'
    graph_pkl = sys.argv[2] if len(sys.argv) > 2 else 'out_graph/graph.pkl'
    g, meta = load_graph_hetero(graph_pkl)
    print("HeteroData:")
    print(g)
    ds = DerbyDataset(samples_dir, split='val')
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else v}")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(dl))
    print("\nbatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")
