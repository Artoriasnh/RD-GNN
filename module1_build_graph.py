"""
Module 1: 构建 Derby 地区的静态异构图
=============================================
输入:  route_to_tc_all.csv (infra)       -- 提供 route <-> berth <-> TC 的映射
       withtrainid.csv    (TD, 可选)      -- 用来抽取 Signal id 清单和对齐校验
输出:  <out_dir>/graph.pkl                -- 包含所有节点/边/映射的 dict
       <out_dir>/node_<type>.csv          -- 节点表（供人工检查 / 可视化）
       <out_dir>/edge_<type>.csv          -- 边表

在本地转 PyG HeteroData（只需要几行）:
    import pickle, torch
    from torch_geometric.data import HeteroData
    g = pickle.load(open('graph.pkl','rb'))
    h = HeteroData()
    for nt, feat in g['node_features'].items():
        h[nt].x = torch.tensor(feat, dtype=torch.float32)
    for et, ei in g['edge_index'].items():
        h[et].edge_index = torch.tensor(ei, dtype=torch.long)
"""
import argparse, ast, os, pickle, re
from collections import defaultdict
import numpy as np
import pandas as pd


# ---------- 1. 从 route id / signal id 推属性的启发式 ----------
#
# Derby 地区 Network Rail 的命名约定（我从数据里观察 + 行业常识）：
#   R<AREA><berth_id><suffix_letter>(<M|S|C>)
#     AREA  ∈ {DC, DW, DY, TD, EC, ...}       — 站场区段（Derby Centre/West/York side/...）
#     suffix_letter ∈ {A, B, C, D, E, ...}    — 同一起点的第几条候选
#     (M)=Main (S)=Shunt (C)=Call-on          — 进路类型
#
# 信号机 id:
#   S<AREA><berth_id>   例 SDW5316, SDC5075
#
# 这些属性我们作为静态节点特征 one-hot 编码。

ROUTE_ID_RE = re.compile(r'^R([A-Z]{2,3})(\d+|\w+?)([A-Z])\(([MSC])\)$')
SIGNAL_ID_RE = re.compile(r'^S([A-Z]{2,3})(.+)$')

ROUTE_KINDS = ['M', 'S', 'C']
# 从数据里观察到的区段码 — 跑一次后可能要补
AREA_CODES_ROUTE = ['DC', 'DW', 'DY', 'TD', 'EC', 'ED', 'FD', 'GA', 'NG', 'RJ',
                    'YW', 'PT', 'PK', 'PM', 'PW']  # 足够，不认识的走 "OTHER"


def parse_route_id(rid: str):
    """从 route id 抽属性。返回 dict(area, kind, suffix) 或 None。"""
    m = ROUTE_ID_RE.match(str(rid))
    if not m:
        return None
    return dict(area=m.group(1), berth_ref=m.group(2),
                suffix=m.group(3), kind=m.group(4))


def parse_signal_id(sid: str):
    m = SIGNAL_ID_RE.match(str(sid))
    if not m:
        return None
    return dict(area=m.group(1), berth_ref=m.group(2))


def parse_tc_id(tcid: str):
    """TC id 形如 TDAB / TDCF / TFBY — 前 2 字母是区域码。"""
    if not isinstance(tcid, str) or len(tcid) < 3:
        return dict(area='OTHER')
    area = tcid[1:3] if tcid.startswith('T') else tcid[:2]
    return dict(area=area)


def one_hot(value, categories):
    """把一个分类值变成 one-hot，未知类别归到最后一维 (OTHER)。"""
    vec = np.zeros(len(categories) + 1, dtype=np.float32)
    if value in categories:
        vec[categories.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


# ---------- 2. 读 infra 并分离 命名进路 / 无名 berth-step ----------

def load_infra(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=['start', 'end', 'track'])  # 删完全空行

    def _parse_track(s):
        if not isinstance(s, str) or s.strip() in ('', '[]'):
            return []
        try:
            return list(ast.literal_eval(s))
        except Exception:
            # 容错：少数行形如 "['T886','T888','T892','T894']" 但 ast 也应该能 parse
            return []

    df['track_list'] = df['track'].apply(_parse_track)
    named = df.dropna(subset=['route']).reset_index(drop=True)
    unnamed = df[df['route'].isna()].reset_index(drop=True)
    print(f"[infra] named routes: {len(named)}  |  unnamed berth-steps: {len(unnamed)}")
    return named, unnamed


# ---------- 3. 汇总所有 berth / TC / signal 清单 ----------

def collect_nodes(named, unnamed, td_path=None):
    berths, tcs = set(), set()
    for _, r in named.iterrows():
        berths.add(str(r['start'])); berths.add(str(r['end']))
        tcs.update(r['track_list'])
    for _, r in unnamed.iterrows():
        berths.add(str(r['start'])); berths.add(str(r['end']))
        tcs.update(r['track_list'])

    routes = sorted(named['route'].dropna().unique().tolist())

    # signal 清单：infra 里没给，必须从 TD 里抽
    signals = []
    if td_path and os.path.exists(td_path):
        # 只读必要列，别把 260MB 全读进来
        print(f"[signals] scanning {td_path} for Signal ids (this takes ~30s)...")
        td_it = pd.read_csv(td_path, usecols=['type', 'id'],
                            chunksize=500_000, low_memory=False)
        sig_set = set()
        for ch in td_it:
            sig_set.update(ch.loc[ch['type'] == 'Signal', 'id'].unique())
        signals = sorted(sig_set)
        print(f"[signals] found {len(signals)}")

    berths = sorted(berths)
    tcs = sorted(tcs)
    return berths, tcs, routes, signals


# ---------- 4. 构建节点特征 ----------

def build_node_features(berths, tcs, routes, signals, named):
    # Berth: [is_platform, is_yard, is_numeric4, is_special]
    berth_feat = []
    for b in berths:
        is_plat = 1.0 if 'PLAT' in b or b.startswith('LPL') else 0.0
        is_yard = 1.0 if any(b.startswith(p) for p in ('L', 'M', 'B', 'X', 'C')) \
                        and not is_plat else 0.0
        is_num4 = 1.0 if b.isdigit() and len(b) == 4 else 0.0
        is_spec = 1.0 - max(is_plat, is_yard, is_num4)
        berth_feat.append([is_plat, is_yard, is_num4, is_spec])
    berth_feat = np.array(berth_feat, dtype=np.float32)

    # TC: one-hot(area)
    tc_areas = sorted({parse_tc_id(t)['area'] for t in tcs})
    tc_feat = np.stack([one_hot(parse_tc_id(t)['area'], tc_areas) for t in tcs])

    # Route: one-hot(area) + one-hot(kind) + [n_tc, has_parse_ok]
    r_area = np.zeros((len(routes), len(AREA_CODES_ROUTE) + 1), dtype=np.float32)
    r_kind = np.zeros((len(routes), len(ROUTE_KINDS) + 1), dtype=np.float32)
    r_scalar = np.zeros((len(routes), 2), dtype=np.float32)
    n_tc_map = dict(zip(named['route'], named['track_list'].apply(len)))
    for i, r in enumerate(routes):
        parsed = parse_route_id(r)
        if parsed:
            r_area[i] = one_hot(parsed['area'], AREA_CODES_ROUTE)
            r_kind[i] = one_hot(parsed['kind'], ROUTE_KINDS)
            r_scalar[i, 1] = 1.0
        else:
            r_area[i] = one_hot(None, AREA_CODES_ROUTE)
            r_kind[i] = one_hot(None, ROUTE_KINDS)
        r_scalar[i, 0] = n_tc_map.get(r, 0) / 12.0  # 归一化到 0-1
    route_feat = np.concatenate([r_area, r_kind, r_scalar], axis=1)

    # Signal: one-hot(area) — 复用 route 的区段码空间
    sig_feat = np.stack([one_hot(
        parse_signal_id(s)['area'] if parse_signal_id(s) else None,
        AREA_CODES_ROUTE) for s in signals]) if signals else np.zeros((0, len(AREA_CODES_ROUTE)+1), dtype=np.float32)

    return dict(berth=berth_feat, tc=tc_feat, route=route_feat, signal=sig_feat)


# ---------- 5. 构建边 ----------

def build_edges(named, unnamed, b2i, tc2i, r2i, s2i):
    """返回 {edge_type_tuple: 2xN ndarray}。"""
    edges = defaultdict(list)

    # B --starts--> R   /   R --ends--> B
    for _, r in named.iterrows():
        rid = r['route']
        if rid not in r2i:
            continue
        bs, be = str(r['start']), str(r['end'])
        if bs in b2i:
            edges[('berth', 'starts', 'route')].append((b2i[bs], r2i[rid]))
        if be in b2i:
            edges[('route', 'ends', 'berth')].append((r2i[rid], b2i[be]))
        for tc in r['track_list']:
            if tc in tc2i:
                edges[('route', 'covers', 'tc')].append((r2i[rid], tc2i[tc]))
                edges[('tc', 'covered_by', 'route')].append((tc2i[tc], r2i[rid]))

    # B --step--> B   (用于 CA 步进的连通性；命名 + 无名都算)
    for _, r in pd.concat([named, unnamed], ignore_index=True).iterrows():
        bs, be = str(r['start']), str(r['end'])
        if bs in b2i and be in b2i:
            edges[('berth', 'step', 'berth')].append((b2i[bs], b2i[be]))

    # TC --adj--> TC  (同一 route 内相邻 TC 视为邻接 — 近似)
    adj = set()
    for _, r in named.iterrows():
        tl = r['track_list']
        for a, b in zip(tl[:-1], tl[1:]):
            if a in tc2i and b in tc2i:
                adj.add((tc2i[a], tc2i[b])); adj.add((tc2i[b], tc2i[a]))
    edges[('tc', 'adj', 'tc')] = list(adj)

    # Signal ↔ Berth: 信号机 id 里的 berth_ref 指向它保护的 berth
    if s2i:
        for sid, si in s2i.items():
            p = parse_signal_id(sid)
            if not p: continue
            ref = p['berth_ref']
            if ref in b2i:
                edges[('signal', 'protects', 'berth')].append((si, b2i[ref]))
                edges[('berth', 'protected_by', 'signal')].append((b2i[ref], si))

    # 去重并转成 ndarray
    out = {}
    for et, lst in edges.items():
        if not lst:
            continue
        arr = np.array(sorted(set(lst)), dtype=np.int64).T   # [2, N]
        out[et] = arr
    return out


# ---------- 6. 构建"berth -> 合法 route 集合"查表（规则掩码要用） ----------

def build_legal_lookup(named, r2i):
    table = defaultdict(list)   # berth_str -> list of route indices
    route_tcs = {}              # route_idx -> list of tc ids (strings)
    route_end_berth = {}
    for _, r in named.iterrows():
        rid = r['route']
        if rid not in r2i: continue
        ri = r2i[rid]
        table[str(r['start'])].append(ri)
        route_tcs[ri] = list(r['track_list'])
        route_end_berth[ri] = str(r['end'])
    return dict(table), route_tcs, route_end_berth


# ---------- 7. 主流程 ----------

def main(infra_csv, td_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    named, unnamed = load_infra(infra_csv)
    berths, tcs, routes, signals = collect_nodes(named, unnamed, td_csv)

    b2i = {v: i for i, v in enumerate(berths)}
    tc2i = {v: i for i, v in enumerate(tcs)}
    r2i = {v: i for i, v in enumerate(routes)}
    s2i = {v: i for i, v in enumerate(signals)}

    print(f"[nodes] berth={len(b2i)}  tc={len(tc2i)}  route={len(r2i)}  signal={len(s2i)}")

    feats = build_node_features(berths, tcs, routes, signals, named)
    edges = build_edges(named, unnamed, b2i, tc2i, r2i, s2i)

    for et, ei in edges.items():
        print(f"[edges] {et}  |E|={ei.shape[1]}")

    legal_table, route_tcs, route_end_berth = build_legal_lookup(named, r2i)
    print(f"[legal] berths with >=1 candidate: {len(legal_table)}")
    if legal_table:
        print(f"[legal] max candidates per berth : {max(len(v) for v in legal_table.values())}")

    # 预计算 berth 的 BFS 前向可达集（N跳内），给 Module 2 用作"未来路径候选"
    from collections import defaultdict, deque
    b_adj = defaultdict(set)
    starts_ei = edges.get(('berth', 'starts', 'route'))
    ends_ei = edges.get(('route', 'ends', 'berth'))
    step_ei = edges.get(('berth', 'step', 'berth'))
    route_start_idx = {}
    if starts_ei is not None:
        for b, r in zip(starts_ei[0], starts_ei[1]):
            route_start_idx[r] = b
    if ends_ei is not None and starts_ei is not None:
        for r, b in zip(ends_ei[0], ends_ei[1]):
            if r in route_start_idx:
                b_adj[berths[route_start_idx[r]]].add(berths[b])
    if step_ei is not None:
        for s, d in zip(step_ei[0], step_ei[1]):
            b_adj[berths[s]].add(berths[d])

    def bfs_reachable(b0, max_hops=12):
        if b0 not in b_adj:
            return {b0}
        vis = {b0}
        q = deque([(b0, 0)])
        while q:
            n, h = q.popleft()
            if h >= max_hops:
                continue
            for nb in b_adj.get(n, []):
                if nb not in vis:
                    vis.add(nb); q.append((nb, h + 1))
        return vis

    print("[legal] precomputing BFS reachable sets per berth (hops=12)...")
    berth_reach = {b: bfs_reachable(b, max_hops=12) for b in berths}
    reach_sizes = [len(v) for v in berth_reach.values()]
    print(f"  reach size: min={min(reach_sizes)}  median={sorted(reach_sizes)[len(reach_sizes)//2]}  max={max(reach_sizes)}")

    # 保存主 pickle
    graph = dict(
        node_ids=dict(berth=berths, tc=tcs, route=routes, signal=signals),
        id_to_idx=dict(berth=b2i, tc=tc2i, route=r2i, signal=s2i),
        node_features=feats,
        edge_index=edges,
        legal_routes_from_berth=legal_table,   # berth_str -> [route_idx,...]
        route_tcs=route_tcs,                   # route_idx -> [tc_str,...]
        route_end_berth=route_end_berth,       # route_idx -> berth_str
        berth_reach=berth_reach,               # berth_str -> set of berth_str (12-hop)
    )
    with open(os.path.join(out_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)

    # 额外导出可读 csv 方便人工检查
    for nt in ('berth', 'tc', 'route', 'signal'):
        ids = graph['node_ids'][nt]
        if not ids: continue
        pd.DataFrame({'idx': range(len(ids)), 'id': ids}).to_csv(
            os.path.join(out_dir, f'node_{nt}.csv'), index=False)

    for et, ei in edges.items():
        pd.DataFrame({'src': ei[0], 'dst': ei[1]}).to_csv(
            os.path.join(out_dir, f'edge_{"__".join(et)}.csv'), index=False)

    print(f"[done] saved to {out_dir}/graph.pkl")
    return graph


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--infra', required=True)
    ap.add_argument('--td', default=None, help='optional; used to collect signal ids')
    ap.add_argument('--out', default='./out_graph')
    a = ap.parse_args()
    main(a.infra, a.td, a.out)
