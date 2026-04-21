"""
Module 2: 事件流重放 + PR 样本抽取
============================================
输入:  graph.pkl            (Module 1 的产物)
       withtrainid.csv      (TD)
       mov_91_127.csv       (MOV)
输出:  <out>/samples.parquet  — 每行一个 PR 样本的元信息
       <out>/graph_states.npz — 每个样本对应的动态节点特征快照 (堆叠成 tensor)
       <out>/event_windows.npz — 每个样本前 K 个事件的 token 序列
       <out>/legal_masks.npz   — 每个样本的合法候选集 bool mask
       <out>/labels.npz        — mark 标签 + time-to-previous-PR 的 Δt
       <out>/vocab.pkl        — 事件 token 的字典

样本定义:
    在每条 type=Panel_Request 的事件 e_i 发生之前的瞬间 t_i^- 取快照.
    标签:
        mark    = e_i 的 id 映射到 r2i 的整数
        dt_prev = e_i.time - 上一个 PR (同一焦点车) 的 time (秒, log 变换)
                  如果该车之前没 PR, 用该车首次出现在 TD 至今的秒数

动态节点特征 (本模块产出):
    berth:  [one_hot_train_class(4), occupancy_dwell_norm]    — 每节点 5 维
    tc:     [occupied(1), since_change_norm(1)]               — 每节点 2 维
    route:  [locked(1), since_last_request_norm(1)]           — 每节点 2 维
    signal: [aspect(1), since_change_norm(1)]                 — 每节点 2 维
    * 这些会和 Module 1 的静态特征在训练时 concat

事件 token 词表 (给 Temporal 分支用):
    type:  {Track, Signal, CA, Route, Panel_Request, CC, CB, TRTS}  (+<PAD>)
    id:    所有 TC / Signal / Route id + 特殊 berth-step id  (+<UNK>, <PAD>)
    train: trainid_filled 的 headcode 首字母+数字前缀  (+<UNK>, <PAD>)

    降采样策略: 为了省 token, Track 事件在序列里只保留焦点车相关的;
              其他焦点无关的 Track 被丢弃 (但图状态照常更新).
"""
import argparse, os, pickle, sys
from collections import defaultdict, deque
import numpy as np
import pandas as pd


# ---------- 工具: 归一化时间差 ----------
def log1p_norm(secs, cap=3600.0):
    """秒 → log(1+s)/log(1+cap), 截到 [0,1] 以上可超过."""
    return np.log1p(max(secs, 0.0)) / np.log1p(cap)


# ---------- 焦点车 MOV 上下文 ----------
class MovContext:
    """给定 (headcode, t), 返回最近一条 MOV 的特征向量."""
    TOC_CATS = [28, 27, 42, 5, 7, 54, 9, 34, 16, 97]  # top 10, 其他归 OTHER
    SRV_PREFIXES = ['1', '2', '5', '6', '9', '0']     # UK 标准列车代码首位

    def __init__(self, mov_df):
        mov_df = mov_df.copy()
        mov_df['t'] = pd.to_datetime(mov_df['actual_timestamp'])
        mov_df['headcode'] = mov_df['train_id'].astype(str).str.slice(2, 6)
        mov_df = mov_df.sort_values(['headcode', 't'])
        # 用 groupby + merge_asof 的 per-group 做近邻查找
        self.by_hc = {k: v[['t', 'timetable_variation', 'variation_status',
                            'toc_id', 'train_service_code']].reset_index(drop=True)
                      for k, v in mov_df.groupby('headcode')}
        # feature dim: delay(1) + status_onehot(4) + toc_onehot(11) + srv_onehot(7) = 23
        self.dim = 1 + 4 + (len(self.TOC_CATS) + 1) + (len(self.SRV_PREFIXES) + 1)

    def get(self, headcode, t):
        vec = np.zeros(self.dim, dtype=np.float32)
        rec = self.by_hc.get(headcode)
        if rec is None:
            return vec
        # 找最近一条 t_mov <= t 的记录
        idx = rec['t'].searchsorted(t, side='right') - 1
        if idx < 0:
            return vec
        row = rec.iloc[idx]
        vec[0] = np.tanh(float(row['timetable_variation']) / 30.0)  # 压缩到 (-1,1)
        status_map = {'EARLY': 0, 'ON TIME': 1, 'LATE': 2, 'OFF ROUTE': 3}
        s = status_map.get(row['variation_status'], None)
        if s is not None:
            vec[1 + s] = 1.0
        # toc
        try:
            toc = int(row['toc_id'])
            vec[5 + (self.TOC_CATS.index(toc) if toc in self.TOC_CATS else len(self.TOC_CATS))] = 1.0
        except Exception:
            vec[5 + len(self.TOC_CATS)] = 1.0
        # service prefix
        srv = str(row['train_service_code'])[:1]
        pidx = self.SRV_PREFIXES.index(srv) if srv in self.SRV_PREFIXES else len(self.SRV_PREFIXES)
        vec[5 + len(self.TOC_CATS) + 1 + pidx] = 1.0
        return vec


# ---------- headcode → 车种 one-hot (for berth feature) ----------
TRAIN_CLASSES = ['0', '1', '2', '5', '6']  # 空车/客车/支线/EMPTY/货运 (UK 标准)
def train_class_onehot(headcode):
    vec = np.zeros(4, dtype=np.float32)
    if not headcode:
        return vec
    c = headcode[0]
    # 映射到 4 类: passenger(1,2), freight(6,4,7), empty(0,5), other
    if c in ('1', '2'):   vec[0] = 1.0  # passenger
    elif c in ('6', '4', '7'): vec[1] = 1.0  # freight
    elif c in ('0', '5'): vec[2] = 1.0  # empty / ECS
    else:                 vec[3] = 1.0  # other
    return vec


# ---------- 词表构建 ----------
class Vocab:
    def __init__(self, items, specials=('<PAD>', '<UNK>')):
        self.itos = list(specials) + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk = self.stoi['<UNK>'] if '<UNK>' in self.stoi else None
        self.pad = self.stoi['<PAD>'] if '<PAD>' in self.stoi else None

    def enc(self, x):
        return self.stoi.get(x, self.unk if self.unk is not None else 0)

    def __len__(self):
        return len(self.itos)


def build_vocabs(td, graph):
    # type vocab
    types = Vocab(td['type'].dropna().unique().tolist())
    # id vocab: 所有可能出现在事件里的 id (tc / signal / route / 数字 berth-step id / 0)
    ids_all = td['id'].dropna().astype(str).unique().tolist()
    ids_vocab = Vocab(ids_all)
    # train vocab: trainid_filled  (1315 个)
    train_vocab = Vocab(td['trainid_filled'].dropna().astype(str).unique().tolist())
    return dict(type=types, id=ids_vocab, train=train_vocab)


# ---------- 核心: 事件流重放 ----------
class Replayer:
    def __init__(self, graph, mov_ctx, K=64):
        self.g = graph
        self.mov = mov_ctx
        self.K = K

        self.b2i = graph['id_to_idx']['berth']
        self.tc2i = graph['id_to_idx']['tc']
        self.r2i = graph['id_to_idx']['route']
        self.s2i = graph['id_to_idx']['signal']

        # 动态状态 (按 index 存)
        self.berth_train = [''] * len(self.b2i)       # trainid 字符串
        self.berth_since = [None] * len(self.b2i)     # 进入时间 (Timestamp or None)
        self.tc_state = np.zeros(len(self.tc2i), dtype=np.int8)
        self.tc_since = [None] * len(self.tc2i)
        self.tc_occupier = [''] * len(self.tc2i)      # 当前 TC 占用车的 trainid (启发式)
        self.signal_state = np.zeros(len(self.s2i), dtype=np.int8)
        self.signal_since = [None] * len(self.s2i)
        self.route_state = np.zeros(len(self.r2i), dtype=np.int8)
        self.route_since = [None] * len(self.r2i)

        # 每辆车最近一次 PR 的时间, 以及最后已知 berth
        self.train_last_pr = {}           # trainid -> Timestamp
        self.train_last_berth = {}        # trainid -> berth_str
        self.train_first_seen = {}        # trainid -> Timestamp

        # 事件 token 滚动窗口
        self.event_buf = deque(maxlen=K)  # each: (type_idx, id_idx, train_idx, rel_t_sec)

        self.route_tcs = graph['route_tcs']
        self.legal_table = graph['legal_routes_from_berth']
        self.berth_reach = graph.get('berth_reach', {})   # 若无, 退化为只看当前 berth
        self.n_routes = len(self.r2i)

    # ----- 动态特征 snapshot -----
    def snapshot(self, now_ts):
        # berth: class_onehot(4) + dwell_norm(1)
        B = np.zeros((len(self.b2i), 5), dtype=np.float32)
        for i, tid in enumerate(self.berth_train):
            if tid:
                B[i, :4] = train_class_onehot(tid)
                if self.berth_since[i] is not None:
                    dwell = (now_ts - self.berth_since[i]).total_seconds()
                    B[i, 4] = log1p_norm(dwell)
        # tc: occ(1) + since(1)
        TC = np.zeros((len(self.tc2i), 2), dtype=np.float32)
        TC[:, 0] = self.tc_state
        for i, ts in enumerate(self.tc_since):
            if ts is not None:
                TC[i, 1] = log1p_norm((now_ts - ts).total_seconds())
        # route: locked(1) + since(1)
        R = np.zeros((len(self.r2i), 2), dtype=np.float32)
        R[:, 0] = self.route_state
        for i, ts in enumerate(self.route_since):
            if ts is not None:
                R[i, 1] = log1p_norm((now_ts - ts).total_seconds())
        # signal: aspect(1) + since(1)
        S = np.zeros((len(self.s2i), 2), dtype=np.float32)
        S[:, 0] = self.signal_state
        for i, ts in enumerate(self.signal_since):
            if ts is not None:
                S[i, 1] = log1p_norm((now_ts - ts).total_seconds())
        return dict(berth=B, tc=TC, route=R, signal=S)

    # ----- 合法候选集 -----
    # 定义: 从 current_berth 前向 BFS 12 跳可达的所有 berth 出发的 route, 且其覆盖的 TC
    #       不被 *其他车* 占用 (被焦点车自己占用是允许的).
    def legal_mask(self, current_berth, focus_tid):
        m = np.zeros(self.n_routes, dtype=np.bool_)
        if current_berth is None:
            return m
        cb = str(current_berth)
        reach = self.berth_reach.get(cb, {cb}) if self.berth_reach else {cb}
        for b in reach:
            for ri in self.legal_table.get(b, []):
                tcs = self.route_tcs.get(ri, [])
                ok = True
                for tc in tcs:
                    ti = self.tc2i.get(tc)
                    if ti is None:
                        continue
                    if self.tc_state[ti] == 1:
                        # 被占: 若占用者是焦点车本身, 仍允许; 否则屏蔽
                        if self.tc_occupier[ti] and self.tc_occupier[ti] != focus_tid:
                            ok = False
                            break
                if ok:
                    m[ri] = True
        return m

    # ----- 事件 token -----
    def encode_event(self, ev, vocabs, now_ts):
        type_idx = vocabs['type'].enc(ev['type'])
        id_idx = vocabs['id'].enc(str(ev['id'])) if pd.notna(ev['id']) else vocabs['id'].pad
        train_idx = vocabs['train'].enc(str(ev['trainid_filled'])) \
                    if pd.notna(ev['trainid_filled']) else vocabs['train'].pad
        rel_t = (now_ts - ev['time']).total_seconds()
        return (type_idx, id_idx, train_idx, log1p_norm(rel_t))

    def window_array(self, vocabs, now_ts):
        # 从缓冲里取 K 个最新 token, 不足则 PAD
        K = self.K
        arr = np.zeros((K, 4), dtype=np.float32)
        arr[:, 0] = vocabs['type'].pad
        arr[:, 1] = vocabs['id'].pad
        arr[:, 2] = vocabs['train'].pad
        buf = list(self.event_buf)
        for i, tok in enumerate(buf[-K:]):
            arr[K - len(buf[-K:]) + i] = tok
        # 第 4 列是相对时间 (log 归一), 前面 encode 时已经放过了
        return arr

    # ----- 更新 state -----
    def update_state(self, ev):
        t = ev['time']; typ = ev['type']; eid = ev['id']; state = ev['state']
        tid = ev['trainid_filled'] if pd.notna(ev['trainid_filled']) else ''
        if tid and tid not in self.train_first_seen:
            self.train_first_seen[tid] = t

        if typ == 'CA':
            fb, tb = ev['from_berth'], ev['to_berth']
            if pd.notna(tb) and str(tb) in self.b2i:
                j = self.b2i[str(tb)]
                self.berth_train[j] = tid
                self.berth_since[j] = t
                if tid: self.train_last_berth[tid] = str(tb)
            if pd.notna(fb) and str(fb) in self.b2i:
                j = self.b2i[str(fb)]
                if self.berth_train[j] == tid:
                    self.berth_train[j] = ''
                    self.berth_since[j] = None

        elif typ == 'CC':
            tb = ev['to_berth']
            if pd.notna(tb) and str(tb) in self.b2i:
                j = self.b2i[str(tb)]
                self.berth_train[j] = tid
                self.berth_since[j] = t
                if tid: self.train_last_berth[tid] = str(tb)

        elif typ == 'CB':
            fb = ev['from_berth']
            if pd.notna(fb) and str(fb) in self.b2i:
                j = self.b2i[str(fb)]
                if self.berth_train[j] == tid:
                    self.berth_train[j] = ''
                    self.berth_since[j] = None

        elif typ == 'Track' and pd.notna(eid):
            j = self.tc2i.get(str(eid))
            if j is not None:
                new_state = int(state) if pd.notna(state) else 0
                self.tc_state[j] = new_state
                self.tc_since[j] = t
                # 占用者启发式: Track 事件的 trainid_filled 就是触发这次变化的车
                if new_state == 1:
                    self.tc_occupier[j] = tid if tid else ''
                else:
                    self.tc_occupier[j] = ''

        elif typ == 'Signal' and pd.notna(eid):
            j = self.s2i.get(str(eid))
            if j is not None:
                self.signal_state[j] = int(state) if pd.notna(state) else 0
                self.signal_since[j] = t

        elif typ == 'Route' and pd.notna(eid):
            j = self.r2i.get(str(eid))
            if j is not None:
                # 注意业务语义: state=0 常代表 "进路已解锁"; 这里保持原值, 下游可再调整
                self.route_state[j] = int(state) if pd.notna(state) else 0
                self.route_since[j] = t

        elif typ == 'Panel_Request' and pd.notna(eid):
            if tid:
                self.train_last_pr[tid] = t


# ---------- 主流程 ----------
def main(graph_pkl, td_csv, mov_csv, out_dir, K=64, max_samples=None,
         downsample_track_in_tokens=True):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[load] graph from {graph_pkl}")
    graph = pickle.load(open(graph_pkl, 'rb'))

    print(f"[load] MOV from {mov_csv}")
    mov_df = pd.read_csv(mov_csv, usecols=['train_id', 'actual_timestamp',
                                           'timetable_variation', 'variation_status',
                                           'toc_id', 'train_service_code'],
                         low_memory=False)
    mov_ctx = MovContext(mov_df)
    print(f"[mov] feature dim = {mov_ctx.dim}")

    print(f"[load] TD from {td_csv}  (this is the big file...)")
    td = pd.read_csv(td_csv, low_memory=False)
    td['time'] = pd.to_datetime(td['time'])
    td = td.sort_values('time', kind='mergesort').reset_index(drop=True)
    print(f"[td] rows={len(td)}  range={td['time'].min()}  ~  {td['time'].max()}")

    print("[vocab] building...")
    vocabs = build_vocabs(td, graph)
    for k, v in vocabs.items():
        print(f"  {k}: size={len(v)}")

    replayer = Replayer(graph, mov_ctx, K=K)

    # 预分配存储
    samples_meta = []
    graph_states = {'berth': [], 'tc': [], 'route': [], 'signal': []}
    event_wins = []
    legal_masks = []
    labels_mark = []
    labels_dt = []
    mov_feats = []

    r2i = graph['id_to_idx']['route']
    n_pr_total = (td['type'] == 'Panel_Request').sum()
    print(f"[samples] {n_pr_total} PR events to process")

    n_done = 0
    n_skipped = 0
    for ev in td.itertuples(index=False):
        evd = ev._asdict()
        t = evd['time']; typ = evd['type']

        # --- 在 PR 处先取快照 (before update state, before pushing to buffer) ---
        if typ == 'Panel_Request' and pd.notna(evd['id']):
            rid = str(evd['id'])
            if rid not in r2i:
                n_skipped += 1
                # 仍然更新 state/buf
                replayer.update_state(evd)
                tok = replayer.encode_event(evd, vocabs, t)
                replayer.event_buf.append(tok)
                continue

            tid = str(evd['trainid_filled']) if pd.notna(evd['trainid_filled']) else ''
            current_berth = replayer.train_last_berth.get(tid, None)

            # ==== 写样本 ====
            snap = replayer.snapshot(t)
            win = replayer.window_array(vocabs, t)
            mask = replayer.legal_mask(current_berth, tid)

            mark = r2i[rid]
            # label 里的合法性: 必须 mask[mark] = True 或者允许 "当前无候选 fallback"
            # 如果 mask[mark]=False, 仍保留样本, 但打标记 —— 可能是数据噪声/图不全
            dt_prev_sec = None
            if tid and tid in replayer.train_last_pr:
                dt_prev_sec = (t - replayer.train_last_pr[tid]).total_seconds()
            elif tid and tid in replayer.train_first_seen:
                dt_prev_sec = (t - replayer.train_first_seen[tid]).total_seconds()

            samples_meta.append({
                'sample_idx': n_done,
                'time': t,
                'trainid': tid,
                'current_berth': current_berth if current_berth is not None else '',
                'pr_route_id': rid,
                'pr_route_idx': mark,
                'mask_includes_label': bool(mask[mark]),
                'n_legal': int(mask.sum()),
                'dt_prev_sec': dt_prev_sec if dt_prev_sec is not None else -1.0,
            })
            graph_states['berth'].append(snap['berth'])
            graph_states['tc'].append(snap['tc'])
            graph_states['route'].append(snap['route'])
            graph_states['signal'].append(snap['signal'])
            event_wins.append(win)
            legal_masks.append(mask)
            labels_mark.append(mark)
            labels_dt.append(np.log1p(max(dt_prev_sec or 0.0, 0.0)))

            hc = tid[:4] if tid else ''
            mov_feats.append(mov_ctx.get(hc, t))

            n_done += 1
            if n_done % 10000 == 0:
                print(f"  [{n_done}/{n_pr_total}]  legal_mean={np.mean([m.sum() for m in legal_masks[-1000:]]):.2f}  "
                      f"mask_hit={np.mean([s['mask_includes_label'] for s in samples_meta[-1000:]]):.3f}")
            if max_samples and n_done >= max_samples:
                break

        # --- 正常更新 state + buffer ---
        replayer.update_state(evd)
        # token buffer 降采样: 非焦点车的 Track 太多, 默认丢弃
        if downsample_track_in_tokens and typ == 'Track':
            continue  # 不入 token 窗口 (但 state 已更新)
        tok = replayer.encode_event(evd, vocabs, t)
        replayer.event_buf.append(tok)

    print(f"[samples] produced={n_done}  skipped_pr_not_in_graph={n_skipped}")

    # ---- 保存 ----
    meta_df = pd.DataFrame(samples_meta)
    meta_df.to_csv(os.path.join(out_dir, 'samples.csv'), index=False)
    print(f"[write] samples.csv  ({len(meta_df)} rows)")

    np.savez_compressed(os.path.join(out_dir, 'graph_states.npz'),
                        berth=np.stack(graph_states['berth']),
                        tc=np.stack(graph_states['tc']),
                        route=np.stack(graph_states['route']),
                        signal=np.stack(graph_states['signal']))
    np.savez_compressed(os.path.join(out_dir, 'event_windows.npz'),
                        X=np.stack(event_wins))
    np.savez_compressed(os.path.join(out_dir, 'legal_masks.npz'),
                        M=np.stack(legal_masks))
    np.savez_compressed(os.path.join(out_dir, 'labels.npz'),
                        mark=np.array(labels_mark, dtype=np.int64),
                        dt_log=np.array(labels_dt, dtype=np.float32))
    np.savez_compressed(os.path.join(out_dir, 'mov_features.npz'),
                        X=np.stack(mov_feats))
    with open(os.path.join(out_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump({k: v.itos for k, v in vocabs.items()}, f)

    print(f"[done] all artifacts in {out_dir}")
    # 统计报告
    print("\n========== 样本质量报告 ==========")
    print(f"总样本数: {len(meta_df)}")
    print(f"current_berth 可用率: {(meta_df['current_berth']!='').mean():.4f}")
    print(f"标签在合法掩码内的比例 (mask_hit): {meta_df['mask_includes_label'].mean():.4f}")
    print(f"每样本合法候选数分布:\n{meta_df['n_legal'].describe()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--td', required=True)
    ap.add_argument('--mov', required=True)
    ap.add_argument('--out', default='./out_samples')
    ap.add_argument('--K', type=int, default=64, help='event token window size')
    ap.add_argument('--max-samples', type=int, default=None)
    a = ap.parse_args()
    main(a.graph, a.td, a.mov, a.out, K=a.K, max_samples=a.max_samples)
