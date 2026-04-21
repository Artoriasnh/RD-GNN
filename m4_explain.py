"""
Module 4.4: 解释性 (GNNExplainer + time attention + 反事实)
===========================================================

三层解释:
  L1 规则层: "这条 route 非法, 因为 TC_X 被 train_Y 占着"
             — 这在 Module 2 / 模型的 legal_mask 里已经隐式实现,
               这里只做"为什么 mask 里不包含 route R"的可读报告.
  L2 GNNExplainer: 对模型选中的 route, 找出最小子图 (节点+边),
             保留子图仍能让模型给出同样的预测.
  L3 反事实: 手动扰动节点特征 (例如把某辆他车从 berth 上移开),
             看预测如何变. 这是最可读的解释给信号员.

注意: GNNExplainer 对异构图的支持在 PyG 2.3+ 存在, 但相对 fragile.
      这里实现一个简化版: 对 batch 大小=1 的样本, 用 edge-mask 扰动法
      找出对 "被选中 route 的 logit" 贡献最大的边.

用法:
    python m4_explain.py \
      --ckpt m4_runs/exp1/best.pt \
      --samples-dir out_samples \
      --graph-pkl out_graph/graph.pkl \
      --sample-idx 100 \
      --k-edges 8
"""
import argparse, os, pickle, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from m4_dataset import DerbyDataset, load_graph_hetero, collate_fn
from m4_model import DerbyModel


def load_checkpoint(ckpt_path, hetero_graph, vocabs, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    node_feat_dims = ckpt['node_feat_dims']
    a = argparse.Namespace(**ckpt['args'])
    model = DerbyModel(
        node_feat_dims=node_feat_dims,
        metadata=hetero_graph.metadata(),
        n_type=len(vocabs['type']), n_id=len(vocabs['id']),
        n_train=len(vocabs['train']),
        d=a.d, n_hgt_layers=a.hgt_layers,
        n_tf_layers=a.tf_layers, n_heads=a.heads,
        n_mix=a.n_mix,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['args']


# ---------- L1: 规则解释 ----------
def rule_explanation(graph_meta, dyn_tc, current_berth,
                     candidate_routes, model_topk_route,
                     route_node_id_map):
    """
    说明: 为什么某些 route 不在 legal_mask 里, 或者为什么 top-k 的选择有变化.
    candidate_routes: list of route indices (model logits 的 top-k)
    返回一个字符串 list.
    """
    tc_ids = graph_meta['node_ids']['tc']
    tc_occ = (dyn_tc[:, 0] > 0.5).numpy()   # (N_T,)
    rules = []
    for r_idx in candidate_routes:
        route_name = graph_meta['node_ids']['route'][r_idx]
        blocking_tcs = []
        for tc_name in graph_meta['route_tcs'].get(r_idx, []):
            ti = graph_meta['id_to_idx']['tc'].get(tc_name)
            if ti is not None and tc_occ[ti]:
                blocking_tcs.append(tc_name)
        if blocking_tcs:
            rules.append(f"route {route_name} has TC occupied: {blocking_tcs}")
        else:
            rules.append(f"route {route_name} — all TCs clear")
    return rules


# ---------- L2: Edge-level importance via perturbation ----------
@torch.no_grad()
def edge_importance(model, batch, static_x, edge_index_dict, num_nodes_per_type,
                     target_route_idx, top_k_edges=8):
    """
    对每种边类型, 逐条边 mask 掉, 观察 target_route_idx 的 logit 变化.
    返回按重要性排序的 top-k 边, 带边类型和端点 id.

    batch_size 必须为 1.
    """
    assert batch['dyn_berth'].shape[0] == 1, "edge_importance requires batch_size=1"
    device = batch['dyn_berth'].device

    # Baseline
    out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
    base_logit = out['logits'][0, target_route_idx].item()

    records = []
    for et, ei in edge_index_dict.items():
        E = ei.shape[1]
        if E == 0: continue
        # 分块 mask: 逐条边太慢. 按 10 条一组做 occlusion 评估重要性.
        group = max(1, E // 30)
        for g_start in range(0, E, group):
            g_end = min(E, g_start + group)
            keep = torch.ones(E, dtype=torch.bool, device=device)
            keep[g_start:g_end] = False
            ei_pert = ei[:, keep]
            eidx_pert = dict(edge_index_dict)
            eidx_pert[et] = ei_pert
            out_p = model(batch, static_x, eidx_pert, num_nodes_per_type)
            pert_logit = out_p['logits'][0, target_route_idx].item()
            drop = base_logit - pert_logit
            # 把 drop 平摊给组内每条边
            for e_idx in range(g_start, g_end):
                records.append(dict(
                    edge_type=et,
                    edge_idx=e_idx,
                    src=int(ei[0, e_idx].item()),
                    dst=int(ei[1, e_idx].item()),
                    importance=drop / (g_end - g_start),
                ))

    records.sort(key=lambda r: -abs(r['importance']))
    return records[:top_k_edges], base_logit


def edges_to_human(records, graph_meta):
    """把边 (src_idx, dst_idx) 和 node_type 翻译成可读 id."""
    readable = []
    for r in records:
        src_type, _, dst_type = r['edge_type']
        src_id = graph_meta['node_ids'][src_type][r['src']]
        dst_id = graph_meta['node_ids'][dst_type][r['dst']]
        readable.append(dict(
            relation='__'.join(r['edge_type']),
            src=f"{src_type}:{src_id}",
            dst=f"{dst_type}:{dst_id}",
            importance=round(r['importance'], 4),
        ))
    return readable


# ---------- L3: 反事实 ----------
@torch.no_grad()
def counterfactual_clear_berth(model, batch, static_x, edge_index_dict,
                                 num_nodes_per_type, berth_idx_to_clear):
    """把某个 berth 的占用清空, 看 top-3 预测的变化."""
    out_orig = model(batch, static_x, edge_index_dict, num_nodes_per_type)
    top3_orig = out_orig['logits'][0].topk(3).indices.tolist()

    batch_cf = {k: v.clone() for k, v in batch.items()}
    # dyn_berth 前 4 维是 train_class onehot, 第 5 维是 dwell; 全清零
    batch_cf['dyn_berth'][0, berth_idx_to_clear, :] = 0.0
    out_cf = model(batch_cf, static_x, edge_index_dict, num_nodes_per_type)
    top3_cf = out_cf['logits'][0].topk(3).indices.tolist()

    return dict(orig_top3=top3_orig, cf_top3=top3_cf,
                changed=top3_orig != top3_cf)


# ---------- 时间分支 attention 可视化 ----------
@torch.no_grad()
def temporal_attention(model, batch):
    """
    提取时间 Transformer 最后一层对最后位置的注意力分布.
    需要临时 hook 进 TransformerEncoderLayer.
    """
    attn_weights = []
    hooks = []
    def make_hook(storage):
        def hook(module, inp, out):
            # out: (attn_output, attn_weights)
            # 但 MultiheadAttention.need_weights=False 时 out[1] 是 None
            pass
        return hook

    # 简化做法: 手动调一次最后一层, 带 need_weights=True
    K = batch['events_cat'].shape[1]
    te = model.temporal
    e = batch['events_cat']
    t = batch['events_t']
    x = torch.cat([
        te.type_emb(e[..., 0]), te.id_emb(e[..., 1]),
        te.train_emb(e[..., 2]), te.time_emb(t),
    ], dim=-1)
    x = te.proj(x)

    # forward 逐层, 最后一层收集 attn
    for i, layer in enumerate(te.tf.layers):
        is_last = (i == len(te.tf.layers) - 1)
        if is_last:
            # 手动做: attention + feedforward
            q = k = v = layer.norm1(x) if layer.norm_first else x
            attn_out, attn_w = layer.self_attn(q, k, v, attn_mask=te.causal_mask,
                                                 need_weights=True, average_attn_weights=True)
            # attn_w: (B, K, K) — 最后位置 attend 到哪些位置
            last_pos_attn = attn_w[:, -1, :]   # (B, K)
            attn_weights.append(last_pos_attn.cpu())
        x = layer(x, src_mask=te.causal_mask)
    return attn_weights[0] if attn_weights else None


# ---------- 主 ----------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_hetero, meta = load_graph_hetero(args.graph_pkl)
    num_nodes_per_type = {nt: g_hetero[nt].num_nodes for nt in g_hetero.node_types}
    static_x = {nt: g_hetero[nt].x.to(device) for nt in g_hetero.node_types}
    edge_index_dict = {et: g_hetero[et].edge_index.to(device)
                       for et in g_hetero.edge_types}

    ds = DerbyDataset(args.samples_dir, args.split)
    model, _ = load_checkpoint(args.ckpt, g_hetero, ds.vocab, device)

    # 单样本 batch
    sub = Subset(ds, [args.sample_idx])
    dl = DataLoader(sub, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(dl))
    batch = {k: v.to(device) for k, v in batch.items()}

    # 前向
    out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
    logits = out['logits'][0]
    true_mark = int(batch['label_mark'][0])
    topk_5 = logits.topk(5).indices.tolist()

    true_name = meta['node_ids']['route'][true_mark]
    topk_names = [meta['node_ids']['route'][i] for i in topk_5]

    print("="*60)
    print(f"Sample index: {args.sample_idx}")
    print(f"  ground truth: {true_name} (idx {true_mark})")
    print(f"  model top-5: {topk_names}")
    print(f"  n_legal: {int(batch['n_legal'][0])}")
    print(f"  correct top1: {topk_5[0] == true_mark}")

    # ---- L1 规则 ----
    # 对 top-5 里没被选为 top-1 的 route 看有没有 TC 阻塞
    print("\n[L1 规则层]")
    rule_expl = rule_explanation(
        meta, batch['dyn_tc'][0].cpu(),
        current_berth=None,   # 简化: 不做当前 berth 反查
        candidate_routes=topk_5,
        model_topk_route=topk_5[0],
        route_node_id_map=None,
    )
    for r in rule_expl:
        print(f"  - {r}")

    # ---- L2 边重要性 ----
    print(f"\n[L2 边重要性 — 对 target route = {topk_names[0]}]")
    top_edges, base_logit = edge_importance(
        model, batch, static_x, edge_index_dict, num_nodes_per_type,
        target_route_idx=topk_5[0], top_k_edges=args.k_edges,
    )
    print(f"  base logit: {base_logit:.3f}")
    for e in edges_to_human(top_edges, meta):
        print(f"  {e['relation']:35s} {e['src']:25s} -> {e['dst']:25s}  imp={e['importance']}")

    # ---- L3 反事实 ----
    print("\n[L3 反事实]")
    # 找一个当前被占的 berth
    occ = (batch['dyn_berth'][0, :, :4].sum(dim=-1) > 0).nonzero(as_tuple=True)[0]
    if len(occ) > 0:
        b_to_clear = int(occ[0])
        b_name = meta['node_ids']['berth'][b_to_clear]
        cf = counterfactual_clear_berth(model, batch, static_x, edge_index_dict,
                                          num_nodes_per_type, b_to_clear)
        print(f"  clearing berth {b_name} (idx {b_to_clear}):")
        print(f"    orig top3: {[meta['node_ids']['route'][i] for i in cf['orig_top3']]}")
        print(f"    cf   top3: {[meta['node_ids']['route'][i] for i in cf['cf_top3']]}")
        print(f"    changed: {cf['changed']}")
    else:
        print("  no occupied berth to remove — skipped")

    # ---- 时间 attention ----
    print("\n[时间 attention — 最后位置 attend 到 K=64 历史 token 的哪几个]")
    attn = temporal_attention(model, batch)
    if attn is not None:
        a = attn[0].numpy()
        top5_pos = np.argsort(-a)[:5]
        ev = batch['events_cat'][0].cpu().numpy()
        for pos in top5_pos:
            typ = ds.vocab['type'][int(ev[pos, 0])]
            id_tok = ds.vocab['id'][int(ev[pos, 1])]
            trn = ds.vocab['train'][int(ev[pos, 2])]
            print(f"  pos={pos} attn={a[pos]:.3f}  "
                  f"type={typ}  id={id_tok}  train={trn}")

    print("\nDone.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--samples-dir', required=True)
    ap.add_argument('--graph-pkl', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--sample-idx', type=int, default=0)
    ap.add_argument('--k-edges', type=int, default=8)
    a = ap.parse_args()
    main(a)
