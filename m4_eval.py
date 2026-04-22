"""
Module 4.5: 综合评估 (A mark + B time + 分层分析)
======================================================
在训练好的 best.pt 上做全面评估, 不动模型, 不重训.

A 侧新增:
  - MRR (mean reciprocal rank)
  - Top-k 对 k=1,2,3,5,10 都算
  - 按 area (DC/DW/DY/EC/...) 的 per-class top-1
  - 按 train 种类 (客运/货运/空车/其他) 的分层
  - 更细的 n_legal 分桶

B 侧新增 (log-normal mixture 的可读指标):
  - MAE, Median AE (秒)                 — 点预测 vs 真实 Δt 秒
  - 80% 区间覆盖率 (P10..P90 是否真的盖住 80%)
  - 50% 区间覆盖率 (P25..P75 盖住 50%)
  - 按 Δt 大小分桶的准确度

用法:
    python m4_eval.py \\
        --ckpt       m4_runs/20260422_003902_full/best.pt \\
        --samples-dir out_samples \\
        --graph-pkl  out_graph/graph.pkl \\
        --out        eval_reports/full
"""
import argparse, json, os, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from m4_dataset import DerbyDataset, load_graph_hetero, collate_fn
from m4_model import DerbyModel


# ============================================================
# LogNormal mixture 的点估计和分位数
# ============================================================
def lnorm_mix_point_prediction(mu, log_sigma, log_w, method='median'):
    """
    从 log-normal mixture 给出点预测 tau (seconds).

    method:
      'mean':   E[τ] = Σ w_k · exp(μ_k + σ_k²/2)  (闭式, 但对大 σ 敏感)
      'median': 数值求解 CDF(τ)=0.5 (鲁棒, 略慢)

    参数:
      mu, log_sigma, log_w: (B, K)  (未归一化 w)
    返回:
      tau_pred: (B,)
    """
    w = F.softmax(log_w, dim=-1)  # (B, K)
    sigma = log_sigma.exp().clamp(max=5.0)   # 防爆: σ > exp(5) 不太可能

    if method == 'mean':
        # E[τ] = Σ w_k · exp(μ_k + σ_k²/2)
        comp_mean = torch.exp(mu + 0.5 * sigma ** 2).clamp(max=1e7)
        return (w * comp_mean).sum(dim=-1)
    elif method == 'median':
        return lnorm_mix_quantile(mu, log_sigma, log_w, q=0.5)
    else:
        raise ValueError(method)


def lnorm_mix_cdf(tau, mu, log_sigma, log_w):
    """
    mixture CDF at tau.
    tau:      (B,) or (B, Q)   正值, 秒
    mu, log_sigma, log_w: (B, K)
    返回 CDF: 同 tau shape
    """
    w = F.softmax(log_w, dim=-1)  # (B, K)
    sigma = log_sigma.exp().clamp(min=1e-3, max=5.0)
    if tau.dim() == mu.dim() - 1:
        tau = tau.unsqueeze(-1)  # (B, 1)
    log_tau = torch.log(tau.clamp(min=1e-3))  # (B, 1) or (B, Q, 1)
    # lognormal CDF: Φ( (log τ - μ) / σ )
    z = (log_tau - mu.unsqueeze(-2) if tau.dim() > mu.dim() else (log_tau - mu)) / \
        (sigma.unsqueeze(-2) if tau.dim() > mu.dim() else sigma)
    # 上面 broadcast 写得丑, 简化一下:
    if log_tau.dim() == 2:           # (B, 1): 求单个 τ 的 cdf
        z = (log_tau - mu) / sigma    # (B, K)
        cdf_comp = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        return (w * cdf_comp).sum(dim=-1)   # (B,)
    elif log_tau.dim() == 3:         # (B, Q, 1): Q 个 τ
        # mu: (B, K) -> (B, 1, K); sigma: 同
        z = (log_tau - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        cdf_comp = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        return (w.unsqueeze(1) * cdf_comp).sum(dim=-1)   # (B, Q)
    else:
        raise ValueError(f"log_tau shape {log_tau.shape}")


def lnorm_mix_quantile(mu, log_sigma, log_w, q, n_bisect=40):
    """
    求 mixture 的 q 分位数: CDF(τ) = q.

    q: scalar in (0, 1)
    返回: (B,)
    """
    B = mu.shape[0]
    device = mu.device
    # 初始上下界 (log 空间)
    # log τ 的范围: 各分量 μ 的 min/max ± 几倍 σ
    sigma = log_sigma.exp().clamp(max=5.0)
    log_lo = (mu - 5 * sigma).min(dim=-1).values  # (B,)
    log_hi = (mu + 5 * sigma).max(dim=-1).values  # (B,)
    # 二分
    lo = log_lo.clone()
    hi = log_hi.clone()
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        tau_mid = mid.exp()
        cdf = lnorm_mix_cdf(tau_mid, mu, log_sigma, log_w)
        mask = cdf < q
        lo = torch.where(mask, mid, lo)
        hi = torch.where(mask, hi, mid)
    return (0.5 * (lo + hi)).exp()


# ============================================================
# Route 属性: 从 node_ids['route'] 里解析 area 等
# ============================================================
import re
ROUTE_RE = re.compile(r'^R([A-Z]{2,3})\d+\w?\([MSC]\)$')


def route_area(rid):
    m = ROUTE_RE.match(str(rid))
    return m.group(1) if m else 'OTHER'


def headcode_class(headcode):
    """UK 标准: 0/5 空车, 1/2 客运, 6/4/7 货运, 其他"""
    if not headcode: return 'unknown'
    c = str(headcode)[:1]
    if c in ('1', '2'): return 'passenger'
    if c in ('6', '4', '7'): return 'freight'
    if c in ('0', '5'): return 'empty_ecs'
    return 'other'


# ============================================================
# 主评估循环: 一次 pass, 收集所有需要的原始张量
# ============================================================
@torch.no_grad()
def run_inference(model, loader, static_x, edge_index_dict, num_nodes_per_type,
                   device, max_batches=None):
    """跑 test loader, 收集 logits, labels, time params, 每个样本的 metadata."""
    model.eval()
    all_logits = []
    all_labels_mark = []
    all_labels_dt = []
    all_n_legal = []
    all_has_legal = []
    all_mu, all_log_s, all_log_w = [], [], []
    all_h_H = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
        logits = out['logits'].float().cpu()
        has_legal = torch.isfinite(logits).any(dim=-1)

        mu, log_s, log_w = out['time_params']
        all_logits.append(logits)
        all_labels_mark.append(batch['label_mark'].cpu())
        all_labels_dt.append(batch['label_dt'].cpu())
        all_n_legal.append(batch['n_legal'].cpu())
        all_has_legal.append(has_legal)
        all_mu.append(mu.float().cpu())
        all_log_s.append(log_s.float().cpu())
        all_log_w.append(log_w.float().cpu())
        all_h_H.append(out['h_H'].float().cpu())

    return dict(
        logits=torch.cat(all_logits),
        labels_mark=torch.cat(all_labels_mark),
        labels_dt=torch.cat(all_labels_dt),
        n_legal=torch.cat(all_n_legal),
        has_legal=torch.cat(all_has_legal),
        mu=torch.cat(all_mu),
        log_s=torch.cat(all_log_s),
        log_w=torch.cat(all_log_w),
        h_H=torch.cat(all_h_H),
    )


# ============================================================
# A 侧指标
# ============================================================
def eval_mark_full(outputs, sp_subset, route_ids,
                    ks=(1, 2, 3, 5, 10)):
    """完整的 mark 评估: top-k, MRR, 按 area / headcode / n_legal 分层."""
    logits = outputs['logits']
    y = outputs['labels_mark']
    hl = outputs['has_legal']
    valid = hl
    denom = valid.float().sum().clamp(min=1).item()

    report = {}

    # top-k
    top_maxk = logits.topk(max(ks), dim=-1).indices
    for k in ks:
        pred_k = top_maxk[:, :k]
        hit = (pred_k == y.unsqueeze(1)).any(dim=1) & valid
        report[f'top{k}'] = float(hit.float().sum() / denom)

    # MRR
    ranks = (-logits).argsort(dim=-1)
    mrr_sum = 0.0
    valid_count = 0
    for i in range(len(y)):
        if not valid[i]: continue
        pos = (ranks[i] == y[i]).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            mrr_sum += 1.0 / (pos[0].item() + 1)
        valid_count += 1
    report['mrr'] = mrr_sum / max(valid_count, 1)

    # 按 n_legal 分桶 (更细)
    n_legal = outputs['n_legal']
    pred_1 = top_maxk[:, 0]
    bins = [(1, 1), (2, 3), (4, 6), (7, 15), (16, 50),
             (51, 150), (151, 300)]
    report['by_n_legal'] = []
    for lo, hi in bins:
        m = (n_legal >= lo) & (n_legal <= hi) & valid
        if m.sum() == 0:
            report['by_n_legal'].append({'bin': f'{lo}-{hi}', 'n': 0, 'top1': None})
        else:
            acc = ((pred_1 == y) & m).float().sum() / m.float().sum()
            report['by_n_legal'].append({
                'bin': f'{lo}-{hi}', 'n': int(m.sum()),
                'top1': float(acc),
            })

    # 按 area 分组 (以 ground truth route 的 area 为准)
    y_area = np.array([route_area(route_ids[int(yi)]) for yi in y])
    pred_area_acc = {}
    for area in np.unique(y_area):
        m = (y_area == area) & valid.numpy()
        if m.sum() == 0: continue
        correct = ((pred_1.numpy() == y.numpy()) & m).sum()
        pred_area_acc[area] = {
            'n': int(m.sum()),
            'top1': float(correct / m.sum()),
        }
    report['by_area'] = pred_area_acc

    # 按车种
    hc = sp_subset['trainid'].fillna('').astype(str).str.slice(0, 1).values
    hc_cls = np.array([headcode_class(h) for h in hc])
    by_hc = {}
    for cls in np.unique(hc_cls):
        m = (hc_cls == cls) & valid.numpy()
        if m.sum() == 0: continue
        correct = ((pred_1.numpy() == y.numpy()) & m).sum()
        by_hc[cls] = {
            'n': int(m.sum()),
            'top1': float(correct / m.sum()),
        }
    report['by_headcode'] = by_hc

    return report


# ============================================================
# B 侧指标
# ============================================================
def eval_time_full(outputs):
    """time head 综合评估."""
    mu = outputs['mu']
    log_s = outputs['log_s']
    log_w = outputs['log_w']
    dt_log = outputs['labels_dt']
    has_legal = outputs['has_legal']

    # 真实 τ (秒)
    tau_true = torch.expm1(dt_log).clamp(min=1e-3)

    # NLL
    # 复现 TimeHead.log_prob
    w = F.softmax(log_w, dim=-1)
    sigma = log_s.exp().clamp(min=1e-3, max=5.0)
    tau_clip = tau_true.clamp(min=1e-3)
    log_tau = torch.log(tau_clip).unsqueeze(-1)
    z = (log_tau - mu) / sigma
    log_normal = -0.5 * z ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    log_w_norm = F.log_softmax(log_w, dim=-1)
    log_mix = torch.logsumexp(log_w_norm + log_normal, dim=-1)
    log_prob = log_mix - log_tau.squeeze(-1)
    nll = -log_prob[has_legal].mean().item()

    # 点预测: median
    tau_med = lnorm_mix_quantile(mu, log_s, log_w, q=0.5)   # (N,)
    tau_med = tau_med[has_legal].numpy()

    # 点预测: mean
    tau_mean = lnorm_mix_point_prediction(mu, log_s, log_w, method='mean')
    tau_mean = tau_mean[has_legal].numpy()

    tau_true_valid = tau_true[has_legal].numpy()

    abs_err_med = np.abs(tau_med - tau_true_valid)
    abs_err_mean = np.abs(tau_mean - tau_true_valid)

    # 分位数覆盖
    p10 = lnorm_mix_quantile(mu, log_s, log_w, q=0.10)
    p25 = lnorm_mix_quantile(mu, log_s, log_w, q=0.25)
    p75 = lnorm_mix_quantile(mu, log_s, log_w, q=0.75)
    p90 = lnorm_mix_quantile(mu, log_s, log_w, q=0.90)
    tau_t = tau_true
    inside_80 = ((tau_t >= p10) & (tau_t <= p90))[has_legal].float().mean().item()
    inside_50 = ((tau_t >= p25) & (tau_t <= p75))[has_legal].float().mean().item()

    # 按 Δt 真值分桶
    bins = [(0, 10), (10, 60), (60, 300), (300, 1800), (1800, 10**9)]
    by_dt = []
    for lo, hi in bins:
        m = (tau_true_valid >= lo) & (tau_true_valid < hi)
        if m.sum() == 0:
            by_dt.append({'range_sec': f'[{lo},{hi})', 'n': 0})
            continue
        by_dt.append({
            'range_sec': f'[{lo},{hi})',
            'n': int(m.sum()),
            'median_abs_err_sec': float(np.median(abs_err_med[m])),
            'mean_abs_err_sec':   float(np.mean(abs_err_med[m])),
        })

    return dict(
        nll=nll,
        mae_sec_median=float(np.mean(abs_err_med)),
        mae_sec_mean=float(np.mean(abs_err_mean)),
        median_abs_err_sec=float(np.median(abs_err_med)),
        coverage_80pct=inside_80,
        coverage_50pct=inside_50,
        by_dt_range=by_dt,
    )


# ============================================================
# 主入口
# ============================================================
def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ck_args = argparse.Namespace(**ckpt['args'])
    print(f"[ckpt] trained at epoch {ckpt['epoch']}  val_top1={ckpt['val_rep']['top1']:.4f}")

    # 图
    g_hetero, meta = load_graph_hetero(args.graph_pkl)
    num_nodes_per_type = {nt: g_hetero[nt].num_nodes for nt in g_hetero.node_types}
    static_x = {nt: g_hetero[nt].x.to(device) for nt in g_hetero.node_types}
    edge_index_dict = {et: g_hetero[et].edge_index.to(device)
                       for et in g_hetero.edge_types}
    route_ids = meta['node_ids']['route']

    # 数据集
    ds_te = DerbyDataset(args.samples_dir, 'test')
    # 注意: DerbyDataset 的 idx 已经按 time 切好了, 但我们要的 sp_subset 是 test 对应的原始 sample.csv 行
    sp_all = pd.read_csv(os.path.join(args.samples_dir, 'samples.csv'))
    sp_all['time'] = pd.to_datetime(sp_all['time'])
    sp_te = sp_all.iloc[ds_te.idx].reset_index(drop=True)

    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0,
                        pin_memory=(device.type == 'cuda'))

    # 模型
    # 兼容 v1 (mov_dim=23, K=64) 和 v2+ (ckpt 里明确保存了)
    mov_dim = ckpt.get('mov_dim', 23)
    K = ckpt.get('K', 64)
    print(f"[model] mov_dim={mov_dim}  K={K}")

    model = DerbyModel(
        node_feat_dims=ckpt['node_feat_dims'],
        metadata=g_hetero.metadata(),
        n_type=len(ckpt['vocabs']['type']),
        n_id=len(ckpt['vocabs']['id']),
        n_train=len(ckpt['vocabs']['train']),
        d=ck_args.d, n_hgt_layers=ck_args.hgt_layers,
        n_tf_layers=ck_args.tf_layers, n_heads=ck_args.heads,
        n_mix=ck_args.n_mix, mov_dim=mov_dim, K=K,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"[model] loaded {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # 推理
    print("[inference] running on test set...")
    t0 = time.time()
    outputs = run_inference(model, dl_te, static_x, edge_index_dict,
                             num_nodes_per_type, device,
                             max_batches=args.max_batches)
    print(f"  done in {time.time()-t0:.1f}s, n={len(outputs['labels_mark'])}")

    # A 评估 — 要把 sp_te 截成和 outputs 实际长度一致 (max_batches 限制时)
    n_collected = len(outputs['labels_mark'])
    sp_te_aligned = sp_te.iloc[:n_collected].reset_index(drop=True)

    print("[eval A] mark prediction...")
    report_A = eval_mark_full(outputs, sp_te_aligned, route_ids)

    # B 评估
    print("[eval B] time prediction...")
    report_B = eval_time_full(outputs)

    # 汇总
    print("\n" + "=" * 60)
    print("A (mark) metrics")
    print("=" * 60)
    print(f"  top-1  = {report_A['top1']:.4f}")
    print(f"  top-2  = {report_A['top2']:.4f}")
    print(f"  top-3  = {report_A['top3']:.4f}")
    print(f"  top-5  = {report_A['top5']:.4f}")
    print(f"  top-10 = {report_A['top10']:.4f}")
    print(f"  MRR    = {report_A['mrr']:.4f}")

    print("\n  by n_legal bucket:")
    for d in report_A['by_n_legal']:
        v = f"{d['top1']:.4f}" if d['top1'] is not None else "N/A"
        print(f"    {d['bin']:<10}  n={d['n']:<6}  top1={v}")

    print("\n  by area (ground-truth route area, top by n):")
    for a, v in sorted(report_A['by_area'].items(),
                         key=lambda kv: -kv[1]['n']):
        print(f"    {a:<8}  n={v['n']:<6}  top1={v['top1']:.4f}")

    print("\n  by headcode class:")
    for c, v in sorted(report_A['by_headcode'].items(),
                         key=lambda kv: -kv[1]['n']):
        print(f"    {c:<12}  n={v['n']:<6}  top1={v['top1']:.4f}")

    print("\n" + "=" * 60)
    print("B (time) metrics")
    print("=" * 60)
    print(f"  NLL                  = {report_B['nll']:.4f}")
    print(f"  MAE (median pred)    = {report_B['mae_sec_median']:.1f} sec")
    print(f"  MAE (mean   pred)    = {report_B['mae_sec_mean']:.1f} sec")
    print(f"  Median |err|         = {report_B['median_abs_err_sec']:.1f} sec")
    print(f"  80% coverage (P10..P90) = {report_B['coverage_80pct']:.4f}  (target 0.80)")
    print(f"  50% coverage (P25..P75) = {report_B['coverage_50pct']:.4f}  (target 0.50)")
    print("\n  by Δt range (true):")
    for d in report_B['by_dt_range']:
        if d['n'] == 0:
            print(f"    {d['range_sec']:<15}  n=0")
        else:
            print(f"    {d['range_sec']:<15}  n={d['n']:<6}  "
                  f"median|err|={d['median_abs_err_sec']:.1f}s  "
                  f"mean|err|={d['mean_abs_err_sec']:.1f}s")

    # 保存
    full = {'A': report_A, 'B': report_B,
             'ckpt_args': ckpt['args'],
             'ckpt_epoch': ckpt['epoch']}
    with open(os.path.join(args.out, 'eval_report.json'), 'w') as f:
        json.dump(full, f, indent=2, default=float)

    # 小巧的 csv 快查表
    pd.DataFrame(report_A['by_n_legal']).to_csv(
        os.path.join(args.out, 'by_n_legal.csv'), index=False)
    pd.DataFrame([{'area': k, **v} for k, v in report_A['by_area'].items()]
                  ).to_csv(os.path.join(args.out, 'by_area.csv'), index=False)
    pd.DataFrame([{'headcode': k, **v} for k, v in report_A['by_headcode'].items()]
                  ).to_csv(os.path.join(args.out, 'by_headcode.csv'), index=False)
    pd.DataFrame(report_B['by_dt_range']).to_csv(
        os.path.join(args.out, 'by_dt_range.csv'), index=False)

    print(f"\n[done] saved to {args.out}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--samples-dir', required=True)
    ap.add_argument('--graph-pkl', required=True)
    ap.add_argument('--out', default='./eval_reports')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-batches', type=int, default=None,
                    help='limit to first N batches (for dev/debug)')
    a = ap.parse_args()
    main(a)
