"""
Module 4.1: Dataset / DataLoader

Fast version for large local datasets.

Main idea:
1. Prefer uncompressed .npy files + mmap for random access.
2. Fall back to old .npz layout if .npy files are not available.
3. Keep Windows DataLoader compatibility via lazy-open per process.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------------
# Graph loader
# -------------------------------
def load_graph_hetero(graph_pkl):
    """把 Module 1 的 pkl 转成 PyG HeteroData + 元数据字典。"""
    from torch_geometric.data import HeteroData

    g = pickle.load(open(graph_pkl, 'rb'))
    h = HeteroData()

    for nt, feat in g['node_features'].items():
        h[nt].x = torch.tensor(feat, dtype=torch.float32)
        h[nt].num_nodes = feat.shape[0]

    for et, ei in g['edge_index'].items():
        h[et].edge_index = torch.tensor(ei, dtype=torch.long).contiguous()

    meta = {
        k: g[k]
        for k in [
            'node_ids',
            'id_to_idx',
            'legal_routes_from_berth',
            'route_tcs',
            'route_end_berth',
            'berth_reach',
        ]
    }
    return h, meta


class DerbyDataset(Dataset):
    """
    一个样本 = 一次 PR 事件。

    返回:
      dyn_berth : (N_B, 5) float
      dyn_tc    : (N_T, 2) float
      dyn_route : (N_R, 2) float
      dyn_signal: (N_S, 2) float
      events_cat: (K, 3) long
      events_t  : (K, 1) float
      mov       : (23,) float
      legal_mask: (N_R,) bool
      label_mark: scalar long
      label_dt  : scalar float
      n_legal   : scalar long
    """

    def __init__(
        self,
        samples_dir,
        split='train',
        train_end='2023-11-16',
        val_end='2023-12-01',
        relax_mask_in_train=True,
        prefer_npy=True,
    ):
        assert split in ('train', 'val', 'test')
        self.dir = Path(samples_dir)
        self.split = split
        self.relax = relax_mask_in_train and (split == 'train')
        self.prefer_npy = prefer_npy

        sp = pd.read_csv(self.dir / 'samples.csv')
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

        # Lazy-open handles per process.
        self._use_npy = False
        self._gs_npz = None
        self._ev_npz = None
        self._mk_npz = None
        self._mov_npz = None

        self._gs_berth = None
        self._gs_tc = None
        self._gs_route = None
        self._gs_signal = None
        self._ev = None
        self._mk = None
        self._mov = None

        lb = np.load(self.dir / 'labels.npz')
        self.mark = lb['mark']
        self.dt = lb['dt_log']
        self.mask_includes_label = sp['mask_includes_label'].values
        self.n_legal = sp['n_legal'].values
        self.vocab = pickle.load(open(self.dir / 'vocab.pkl', 'rb'))

        # ---- (方案 A) 曝露 area / headcode_class 供加权 loss 用 ----
        # 每个样本的 "area": 由 pr_route_id 解析 (例如 RDC5061A(M) -> 'DC')
        import re
        _RE = re.compile(r'^R([A-Z]{2,3})\d+\w?\([MSC]\)$')
        def _area(rid):
            m = _RE.match(str(rid)); return m.group(1) if m else 'OTHER'
        sp_area = sp['pr_route_id'].astype(str).map(_area).values
        # headcode class: passenger / freight / empty_ecs / other
        hc_first = sp['trainid'].fillna('').astype(str).str.slice(0, 1)
        def _hc_cls(c):
            if c in ('1', '2'): return 'passenger'
            if c in ('6', '4', '7'): return 'freight'
            if c in ('0', '5'): return 'empty_ecs'
            return 'other'
        sp_hc = hc_first.map(_hc_cls).values
        # 做成 (area, hc) 组合字符串, 再映射到 int
        combo = np.array([f'{a}|{h}' for a, h in zip(sp_area, sp_hc)])
        combos_unique = sorted(set(combo.tolist()))
        combo_to_idx = {c: i for i, c in enumerate(combos_unique)}
        self.combo_idx_all = np.array([combo_to_idx[c] for c in combo],
                                       dtype=np.int64)
        self.combo_names = combos_unique   # 长度 = n_combo
        # 在 __getitem__ 里只返回样本自己的 combo_idx

        print(
            f"[dataset/{split}] {len(self.idx)} samples  "
            f"relax_mask_in_train={self.relax}  prefer_npy={self.prefer_npy}  "
            f"n_combos={len(combos_unique)}"
        )

    def _has_fast_npy_layout(self):
        needed = [
            self.dir / 'graph_states_berth.npy',
            self.dir / 'graph_states_tc.npy',
            self.dir / 'graph_states_route.npy',
            self.dir / 'graph_states_signal.npy',
            self.dir / 'event_windows_X.npy',
            self.dir / 'legal_masks_M.npy',
            self.dir / 'mov_features_X.npy',
        ]
        return all(p.exists() for p in needed)

    def _ensure_open(self):
        if self._gs_berth is not None or self._gs_npz is not None:
            return

        use_npy = self.prefer_npy and self._has_fast_npy_layout()
        self._use_npy = use_npy

        if use_npy:
            self._gs_berth = np.load(self.dir / 'graph_states_berth.npy', mmap_mode='r')
            self._gs_tc = np.load(self.dir / 'graph_states_tc.npy', mmap_mode='r')
            self._gs_route = np.load(self.dir / 'graph_states_route.npy', mmap_mode='r')
            self._gs_signal = np.load(self.dir / 'graph_states_signal.npy', mmap_mode='r')
            self._ev = np.load(self.dir / 'event_windows_X.npy', mmap_mode='r')
            self._mk = np.load(self.dir / 'legal_masks_M.npy', mmap_mode='r')
            self._mov = np.load(self.dir / 'mov_features_X.npy', mmap_mode='r')
        else:
            self._gs_npz = np.load(self.dir / 'graph_states.npz', mmap_mode='r')
            self._ev_npz = np.load(self.dir / 'event_windows.npz', mmap_mode='r')
            self._mk_npz = np.load(self.dir / 'legal_masks.npz', mmap_mode='r')
            self._mov_npz = np.load(self.dir / 'mov_features.npz', mmap_mode='r')
            self._ev = self._ev_npz['X']
            self._mk = self._mk_npz['M']
            self._mov = self._mov_npz['X']

        backend = 'npy+mmap' if use_npy else 'npz fallback'
        print(f'[dataset/{self.split}] opened backend: {backend}')

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in (
            '_gs_npz', '_ev_npz', '_mk_npz', '_mov_npz',
            '_gs_berth', '_gs_tc', '_gs_route', '_gs_signal',
            '_ev', '_mk', '_mov',
        ):
            state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        self._ensure_open()
        k = self.idx[i]

        if self._use_npy:
            dyn_berth = torch.from_numpy(np.asarray(self._gs_berth[k]).copy())
            dyn_tc = torch.from_numpy(np.asarray(self._gs_tc[k]).copy())
            dyn_route = torch.from_numpy(np.asarray(self._gs_route[k]).copy())
            dyn_signal = torch.from_numpy(np.asarray(self._gs_signal[k]).copy())
        else:
            dyn_berth = torch.from_numpy(np.asarray(self._gs_npz['berth'][k]).copy())
            dyn_tc = torch.from_numpy(np.asarray(self._gs_npz['tc'][k]).copy())
            dyn_route = torch.from_numpy(np.asarray(self._gs_npz['route'][k]).copy())
            dyn_signal = torch.from_numpy(np.asarray(self._gs_npz['signal'][k]).copy())

        ev = np.asarray(self._ev[k]).copy()
        events_cat = torch.from_numpy(ev[:, :3]).long()
        events_t = torch.from_numpy(ev[:, 3:4]).float()

        mov = torch.from_numpy(np.asarray(self._mov[k]).copy()).float()
        mask = torch.from_numpy(np.asarray(self._mk[k]).copy()).bool()

        mark = int(self.mark[k])
        dt = float(self.dt[k])

        if self.relax and not self.mask_includes_label[k]:
            mask = mask.clone()
            mask[mark] = True

        return {
            'dyn_berth': dyn_berth,
            'dyn_tc': dyn_tc,
            'dyn_route': dyn_route,
            'dyn_signal': dyn_signal,
            'events_cat': events_cat,
            'events_t': events_t,
            'mov': mov,
            'legal_mask': mask,
            'label_mark': torch.tensor(mark, dtype=torch.long),
            'label_dt': torch.tensor(dt, dtype=torch.float32),
            'n_legal': torch.tensor(int(self.n_legal[k]), dtype=torch.long),
            'combo_idx': torch.tensor(int(self.combo_idx_all[k]), dtype=torch.long),
        }


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader

    samples_dir = sys.argv[1] if len(sys.argv) > 1 else 'out_samples'
    graph_pkl = sys.argv[2] if len(sys.argv) > 2 else 'out_graph/graph.pkl'

    g, meta = load_graph_hetero(graph_pkl)
    print('HeteroData:')
    print(g)

    ds = DerbyDataset(samples_dir, split='val')
    sample = ds[0]
    for k, v in sample.items():
        print(f'  {k}: {tuple(v.shape) if hasattr(v, "shape") else v}')

    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))
    print('\nbatch shapes:')
    for k, v in batch.items():
        print(f'  {k}: {tuple(v.shape)}')
