"""
Module 3 (baseline): 频率先验 + LightGBM 基线
=================================================
输入 (全部由 Module 2 生成):
    out_samples/samples.csv
    out_samples/graph_states.npz
    out_samples/event_windows.npz
    out_samples/legal_masks.npz
    out_samples/labels.npz
    out_samples/mov_features.npz
    out_samples/vocab.pkl
    out_graph/graph.pkl

产出:
    baseline_reports/metrics.json     — 汇总所有数字
    baseline_reports/per_fork.csv     — 按"分叉度"分层的准确率
    baseline_reports/feature_importance.csv
    baseline_reports/model_mark.txt   — LightGBM 模型 (可加载复用)
    baseline_reports/model_dt.txt

跑法:
    python module3_baseline.py \
      --samples-dir out_samples \
      --graph-pkl   out_graph/graph.pkl \
      --out         baseline_reports

设计:
  A. 频率先验 (zero-ML): 按 (prev_pr_id, current_berth) 查历史最常见的 PR.
  B. LightGBM (tabular): 把所有特征拍扁到固定长度向量, 多分类到 235 个标签.
  C. 简易 dt 回归: LightGBM regressor 预测 log(1+Δt_sec).

关键细节:
  - 按时间切分 (训练/验证/测试 = 9-11月前半 / 11月后半 / 12月), 不做随机切.
  - 评估 top-k 时, 必须在 legal_mask 上重归一化概率, 这是掩码模型的正确评估方式.
  - top-k 按"分叉度" (n_legal) 分层报告 — 分叉=1 的 case 几乎 100%, 只看均值会骗自己.
"""
import argparse, json, os, pickle, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


# ---------- 数据加载 ----------
def load_all(samples_dir, graph_pkl):
    t0 = time.time()
    sp = pd.read_csv(os.path.join(samples_dir, 'samples.csv'))
    sp['time'] = pd.to_datetime(sp['time'])
    gs = np.load(os.path.join(samples_dir, 'graph_states.npz'))
    ev = np.load(os.path.join(samples_dir, 'event_windows.npz'))['X']
    mk = np.load(os.path.join(samples_dir, 'legal_masks.npz'))['M']
    lb = np.load(os.path.join(samples_dir, 'labels.npz'))
    mv = np.load(os.path.join(samples_dir, 'mov_features.npz'))['X']
    vc = pickle.load(open(os.path.join(samples_dir, 'vocab.pkl'), 'rb'))
    g = pickle.load(open(graph_pkl, 'rb'))
    print(f"[load] {len(sp)} samples in {time.time()-t0:.1f}s")
    return sp, gs, ev, mk, lb, mv, vc, g


# ---------- 时间切分 ----------
def time_split(sp, train_end='2023-11-16', val_end='2023-12-01'):
    t = sp['time'].values
    tr = t < np.datetime64(train_end)
    va = (t >= np.datetime64(train_end)) & (t < np.datetime64(val_end))
    te = t >= np.datetime64(val_end)
    print(f"[split] train={tr.sum()}  val={va.sum()}  test={te.sum()}")
    return np.where(tr)[0], np.where(va)[0], np.where(te)[0]


# ---------- 特征工程 (tabular flat) ----------
def build_tabular_features(sp, gs, ev, mv, g, vc):
    """
    把每个样本压成一个定长特征向量.
    维度构成:
      scalars:                                             ~15
      prev_pr_id (作为 categorical int):                    1
      prev_prev_pr_id:                                      1
      prev_type:                                            1
      current_berth (int, missing=-1):                      1
      train_headcode_first_char:                            1
      mov:                                                 23
      tc_occ_ratio (按区域统计的 TC 占用比例):              10
      route_lock_count:                                     1
      tc_total_occupied:                                    1
      event_window 聚合 (最近 K 个事件里各 type 的数量):    10
      time: hour_sin, hour_cos, dow_sin, dow_cos            4
    """
    N = len(sp)
    berth_state = gs['berth']   # (N, 167, 5)
    tc_state    = gs['tc']      # (N, 276, 2)
    route_state = gs['route']   # (N, 277, 2)

    # --- id 到 index 的映射 ---
    id_itos = vc['id']                      # list
    id_stoi = {s: i for i, s in enumerate(id_itos)}
    type_itos = vc['type']
    type_stoi = {s: i for i, s in enumerate(type_itos)}
    b2i = g['id_to_idx']['berth']

    # --- 从 event_windows 里拿 "上一/上上一 PR" 的 id token ---
    # ev shape: (N, K, 4) -> columns: [type_idx, id_idx, train_idx, rel_t_norm]
    type_pr_idx = type_stoi.get('Panel_Request', -1)

    prev_pr = np.full(N, -1, dtype=np.int32)
    prev_prev_pr = np.full(N, -1, dtype=np.int32)
    prev_type = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        types_i = ev[i, :, 0].astype(np.int32)
        ids_i   = ev[i, :, 1].astype(np.int32)
        pr_positions = np.where(types_i == type_pr_idx)[0]
        # event_windows 是 "past K events, 不含当前"; 最后一位是最近的.
        # 直接取 type==PR 的最后两条
        if len(pr_positions) > 0:
            prev_pr[i] = ids_i[pr_positions[-1]]
        if len(pr_positions) > 1:
            prev_prev_pr[i] = ids_i[pr_positions[-2]]
        # prev_type 只取最新一条非 PAD 的事件
        non_pad = np.where(ids_i > 0)[0]
        if len(non_pad) > 0:
            prev_type[i] = types_i[non_pad[-1]]

    # --- current_berth → int ---
    cb = sp['current_berth'].fillna('').astype(str).values
    cb_int = np.array([b2i.get(x, -1) for x in cb], dtype=np.int32)

    # --- train headcode 首字母 ---
    hc_first = sp['trainid'].fillna('').astype(str).str.slice(0, 1).values
    hc_map = {c: i for i, c in enumerate(sorted(set(hc_first)))}
    hc_int = np.array([hc_map.get(x, -1) for x in hc_first], dtype=np.int32)

    # --- TC 占用按区域聚合 (用节点静态 one-hot 不方便, 简化: 总占用数 + 按 id 前缀 2 字母分组) ---
    tc_ids = g['node_ids']['tc']
    tc_area = np.array([t[1:3] if t.startswith('T') and len(t) >= 3 else 'OT' for t in tc_ids])
    area_cats = sorted(set(tc_area.tolist()))
    # 限制到 top10 以免爆维度
    area_counts = Counter(tc_area.tolist())
    top_areas = [a for a, _ in area_counts.most_common(10)]
    area_masks = {a: (tc_area == a) for a in top_areas}
    tc_occ = tc_state[..., 0]   # (N, 276)

    tc_occ_by_area = np.zeros((N, len(top_areas)), dtype=np.float32)
    for j, a in enumerate(top_areas):
        m = area_masks[a]
        tc_occ_by_area[:, j] = tc_occ[:, m].mean(axis=1)

    tc_total_occupied = tc_occ.sum(axis=1, dtype=np.float32)
    route_lock_count = route_state[..., 0].sum(axis=1, dtype=np.float32)

    # --- event_window 聚合: 各 type 的数量 ---
    n_types = len(type_itos)
    type_counts = np.zeros((N, n_types), dtype=np.float32)
    for i in range(N):
        types_i = ev[i, :, 0].astype(np.int32)
        ids_i   = ev[i, :, 1].astype(np.int32)
        valid = ids_i > 0   # 排除 PAD
        for t_ in types_i[valid]:
            if 0 <= t_ < n_types:
                type_counts[i, t_] += 1
    # 只保留有意义的列, 避免 PAD/UNK 占位 — 但为简单起见, 全保留

    # --- 时间周期 ---
    t = pd.to_datetime(sp['time'])
    hour = t.dt.hour.values + t.dt.minute.values / 60.0
    dow = t.dt.dayofweek.values
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    # --- 拼接 ---
    feat_num = np.column_stack([
        tc_total_occupied, route_lock_count,
        tc_occ_by_area,
        type_counts,
        hour_sin, hour_cos, dow_sin, dow_cos,
        mv,
    ]).astype(np.float32)

    # 分类列 (单独给 LightGBM 当 categorical)
    # LightGBM 对 categorical 要求非负整数, 把 -1 (缺失) 映射成独立的一个大值
    MISSING = 99999
    feat_cat = np.column_stack([
        np.where(prev_pr < 0, MISSING, prev_pr),
        np.where(prev_prev_pr < 0, MISSING, prev_prev_pr),
        np.where(prev_type < 0, MISSING, prev_type),
        np.where(cb_int < 0, MISSING, cb_int),
        np.where(hc_int < 0, MISSING, hc_int),
    ]).astype(np.int32)

    # 命名
    num_cols = (
        ['tc_total_occupied', 'route_lock_count'] +
        [f'tc_occ_{a}' for a in top_areas] +
        [f'type_cnt_{t_}' for t_ in type_itos] +
        ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] +
        [f'mov_{i}' for i in range(mv.shape[1])]
    )
    cat_cols = ['prev_pr', 'prev_prev_pr', 'prev_type', 'current_berth', 'hc_first']

    X = np.concatenate([feat_num, feat_cat.astype(np.float32)], axis=1)
    col_names = num_cols + cat_cols
    cat_idx = list(range(len(num_cols), len(col_names)))

    print(f"[feat] numeric={len(num_cols)}  categorical={len(cat_cols)}  total={X.shape[1]}")
    return X, col_names, cat_idx


# ---------- 评估: top-k on legal candidates ----------
def topk_with_mask(proba, mask, k):
    """
    proba: (N, C) 概率
    mask:  (N, C) bool (True=合法)
    返回每样本的 top-k 预测 index (在合法候选内重归一化).
    """
    N, C = proba.shape
    out = np.full((N, k), -1, dtype=np.int32)
    scored = proba * mask.astype(np.float32)   # 非法置 0
    # 取前 k 大
    if k == 1:
        out[:, 0] = scored.argmax(axis=1)
    else:
        # argpartition 再排序
        part = np.argpartition(-scored, kth=min(k, C-1)-1, axis=1)[:, :k]
        # 按实际分数降序
        row_scores = np.take_along_axis(scored, part, axis=1)
        order = np.argsort(-row_scores, axis=1)
        out = np.take_along_axis(part, order, axis=1)
    return out


def eval_mark(proba, mask, y_true, sp_subset, name='', ks=(1, 3, 5)):
    rep = {}
    for k in ks:
        pred = topk_with_mask(proba, mask, k)
        hit = (pred == y_true[:, None]).any(axis=1)
        rep[f'{name}_top{k}'] = float(hit.mean())
    # MRR
    # scored ranking
    scored = proba * mask.astype(np.float32)
    ranks = (-scored).argsort(axis=1)
    mrr = 0.0
    for i, y in enumerate(y_true):
        pos = np.where(ranks[i] == y)[0]
        mrr += 1.0 / (pos[0] + 1) if len(pos) else 0.0
    rep[f'{name}_mrr'] = mrr / len(y_true)

    # 分叉度分层 top-1
    n_legal = sp_subset['n_legal'].values
    bins = [(1, 1), (2, 3), (4, 6), (7, 15), (16, 300)]
    layered = []
    pred1 = topk_with_mask(proba, mask, 1)[:, 0]
    for lo, hi in bins:
        m = (n_legal >= lo) & (n_legal <= hi)
        if m.sum() == 0:
            layered.append({'bin': f'{lo}-{hi}', 'n': 0, 'top1': None})
            continue
        acc = (pred1[m] == y_true[m]).mean()
        layered.append({'bin': f'{lo}-{hi}', 'n': int(m.sum()), 'top1': float(acc)})
    return rep, layered


# ---------- 基线 A: 频率先验 ----------
def baseline_frequency(sp, tr_idx, te_idx, mask_te, y_te):
    """P(PR | prev_pr, current_berth), 回退到 P(PR | current_berth), 再回退到 P(PR)."""
    tr = sp.iloc[tr_idx]
    # 从 samples 里拿 prev_pr 不方便, 直接用 (current_berth, hour_bin) 做先验 — 这是最简版
    hour_bin = pd.to_datetime(tr['time']).dt.hour.values // 4
    key = list(zip(tr['current_berth'].fillna('').astype(str), hour_bin))
    lbl = tr['pr_route_idx'].values

    table = defaultdict(Counter)
    global_counter = Counter()
    by_berth = defaultdict(Counter)
    for k, y in zip(key, lbl):
        table[k][y] += 1
        by_berth[k[0]][y] += 1
        global_counter[y] += 1

    te_sp = sp.iloc[te_idx]
    hour_bin_te = pd.to_datetime(te_sp['time']).dt.hour.values // 4
    key_te = list(zip(te_sp['current_berth'].fillna('').astype(str), hour_bin_te))

    C = mask_te.shape[1]
    proba = np.zeros((len(te_idx), C), dtype=np.float32)
    for i, (k, berth) in enumerate(zip(key_te, te_sp['current_berth'].fillna(''))):
        c = table.get(k)
        if c is None or sum(c.values()) < 3:
            c = by_berth.get(str(berth), global_counter)
        total = sum(c.values()) or 1
        for cls, cnt in c.items():
            proba[i, cls] = cnt / total
    rep, layered = eval_mark(proba, mask_te, y_te, te_sp, name='freq')
    return rep, layered, proba


# ---------- 基线 B: LightGBM 多分类 ----------
def baseline_lgbm_mark(X_tr, y_tr, X_va, y_va, X_te, mask_te, y_te, sp_te, cat_idx, num_class):
    import lightgbm as lgb
    params = dict(
        objective='multiclass',
        num_class=num_class,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        max_cat_to_onehot=4,   # 防止高基数 cat 被过度 one-hot
        verbose=-1,
        num_threads=-1,
    )
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_idx, free_raw_data=False, reference=dtr)
    print(f"[lgbm-mark] training on {X_tr.shape} with {num_class} classes...")
    model = lgb.train(
        params, dtr, num_boost_round=300,
        valid_sets=[dva], valid_names=['val'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)],
    )
    proba = model.predict(X_te, num_iteration=model.best_iteration)
    rep, layered = eval_mark(proba, mask_te, y_te, sp_te, name='lgbm')
    return rep, layered, proba, model


# ---------- 简易 B 回归: log(1+Δt) ----------
def baseline_lgbm_dt(X_tr, y_tr, X_va, y_va, X_te, y_te, cat_idx):
    import lightgbm as lgb
    params = dict(
        objective='regression',
        metric='rmse',
        learning_rate=0.1,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        verbose=-1,
        num_threads=-1,
    )
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_idx, free_raw_data=False, reference=dtr)
    print(f"[lgbm-dt] training on {X_tr.shape}...")
    model = lgb.train(
        params, dtr, num_boost_round=400,
        valid_sets=[dva], valid_names=['val'],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    pred = model.predict(X_te, num_iteration=model.best_iteration)
    # 还原到秒
    pred_sec = np.expm1(pred)
    true_sec = np.expm1(y_te)
    rmse_log = float(np.sqrt(((pred - y_te) ** 2).mean()))
    mae_sec = float(np.abs(pred_sec - true_sec).mean())
    median_err = float(np.median(np.abs(pred_sec - true_sec)))
    return dict(dt_rmse_log=rmse_log, dt_mae_sec=mae_sec, dt_median_abs_err_sec=median_err), model


# ---------- 主流程 ----------
def main(samples_dir, graph_pkl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sp, gs, ev, mk, lb, mv, vc, g = load_all(samples_dir, graph_pkl)

    y = lb['mark']
    y_dt = lb['dt_log']
    C = mk.shape[1]

    # 过滤掉 "标签不在合法掩码内" 的样本 (约 5%) — 对 baseline 而言这些是无法学的.
    # 保留它们会压低分数但不会教会模型. 训练时剔, 测试时保留并按 n_legal=0 单独看.
    keep = sp['mask_includes_label'].values
    print(f"[filter] keep {keep.sum()}/{len(sp)} with label in mask")

    tr_all, va_all, te_all = time_split(sp)
    tr = np.intersect1d(tr_all, np.where(keep)[0])
    va = np.intersect1d(va_all, np.where(keep)[0])
    te = te_all  # 测试集不过滤, 反映真实表现

    print("[feat] building tabular features (may take ~1min)...")
    X, col_names, cat_idx = build_tabular_features(sp, gs, ev, mv, g, vc)

    reports = {}

    # --- 1. 频率先验 ---
    print("\n[baseline-freq]")
    rep, lay, _ = baseline_frequency(sp, tr_all, te, mk[te], y[te])
    reports['freq'] = rep
    reports['freq_layered'] = lay
    print(rep)

    # --- 2. LightGBM mark ---
    print("\n[baseline-lgbm-mark]")
    rep, lay, proba_lgbm, model_mark = baseline_lgbm_mark(
        X[tr], y[tr], X[va], y[va], X[te], mk[te], y[te], sp.iloc[te],
        cat_idx, num_class=C,
    )
    reports['lgbm'] = rep
    reports['lgbm_layered'] = lay
    print(rep)

    # --- 3. LightGBM dt (B 任务) ---
    print("\n[baseline-lgbm-dt]")
    rep_dt, model_dt = baseline_lgbm_dt(X[tr], y_dt[tr], X[va], y_dt[va],
                                         X[te], y_dt[te], cat_idx)
    reports['dt'] = rep_dt
    print(rep_dt)

    # --- 输出 ---
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(reports, f, indent=2)

    # 按分叉度分层
    all_layered = []
    for bl in ['freq', 'lgbm']:
        for d in reports[f'{bl}_layered']:
            all_layered.append(dict(model=bl, **d))
    pd.DataFrame(all_layered).to_csv(os.path.join(out_dir, 'per_fork.csv'), index=False)

    # 特征重要性
    imp = pd.DataFrame({
        'feature': col_names,
        'importance_gain': model_mark.feature_importance(importance_type='gain'),
        'importance_split': model_mark.feature_importance(importance_type='split'),
    }).sort_values('importance_gain', ascending=False)
    imp.to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)

    model_mark.save_model(os.path.join(out_dir, 'model_mark.txt'))
    model_dt.save_model(os.path.join(out_dir, 'model_dt.txt'))

    # --- 漂亮的 summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'method':<20} {'top1':>8} {'top3':>8} {'top5':>8} {'MRR':>8}")
    for m in ['freq', 'lgbm']:
        r = reports[m]
        print(f"{m:<20} {r[f'{m}_top1']:>8.4f} {r[f'{m}_top3']:>8.4f} {r[f'{m}_top5']:>8.4f} {r[f'{m}_mrr']:>8.4f}")
    print(f"\ndt task:  RMSE(log) = {reports['dt']['dt_rmse_log']:.3f}  "
          f"MAE(sec) = {reports['dt']['dt_mae_sec']:.1f}  "
          f"median|err|(sec) = {reports['dt']['dt_median_abs_err_sec']:.1f}")

    print("\nTop-1 by fork size (LightGBM):")
    for d in reports['lgbm_layered']:
        print(f"  n_legal in {d['bin']:<8}: n={d['n']:<6}  top1={d['top1']}")

    print(f"\nsaved to {out_dir}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-dir', required=True)
    ap.add_argument('--graph-pkl', required=True)
    ap.add_argument('--out', default='./baseline_reports')
    a = ap.parse_args()
    main(a.samples_dir, a.graph_pkl, a.out)
