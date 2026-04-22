"""
Module 4.3: 训练 + 评估 (GPU 优化版)
==========================================
GPU 优化:
  - 自动混合精度 (AMP / bfloat16)
  - pin_memory + non_blocking 数据传输
  - cudnn benchmark (模型形状固定)
  - TF32 (Ampere+ 卡)
  - 显存监控
  - smoke test 模式 (快速验证代码能跑通)

用法:
    # 正式训练 (GPU)
    python m4_train.py \
      --samples-dir out_samples \
      --graph-pkl   out_graph/graph.pkl \
      --out         m4_runs/exp1 \
      --batch-size  64 --epochs 5

    # 冒烟测试 (10 min 内确认能跑)
    python m4_train.py ... --smoke-test

    # 半精度 (bfloat16 on Ampere+)
    python m4_train.py ... --amp

默认参数:
  d=128, hgt_layers=2, tf_layers=3, heads=4, K=64, n_mix=4
  lr=3e-4, weight_decay=0.01, warmup=500 steps, cosine after warmup
  loss = mark_CE + 0.1 * time_NLL
"""
import argparse, os, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from m4_dataset import DerbyDataset, load_graph_hetero, collate_fn
from m4_model import DerbyModel


def setup_gpu_perf(enable_tf32=True):
    """开启 GPU 性能相关的全局选项."""
    if not torch.cuda.is_available():
        return
    # 模型结构固定, 开 benchmark 让 cudnn 选最快的 kernel
    torch.backends.cudnn.benchmark = True
    if enable_tf32:
        # Ampere (RTX 30xx/40xx, A100) 之后的卡上 TF32 是无感加速
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[gpu] CUDA 不可用, 将用 CPU")
        return
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[gpu] {props.name}  |  {props.total_memory/1e9:.1f} GB  |  "
          f"cap={props.major}.{props.minor}  |  SMs={props.multi_processor_count}")


def print_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[gpu-mem {tag}] alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, static_x, edge_index_dict, num_nodes_per_type,
             device, max_batches=None, use_amp=False, amp_dtype=torch.bfloat16):
    model.eval()
    all_pred_top1, all_pred_top3, all_pred_top5 = [], [], []
    all_true = []
    all_n_legal = []
    all_dt_pred_loglik = []
    all_has_legal = []

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=use_amp and device.type=='cuda',
                                 dtype=amp_dtype):
            out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
            logits = out['logits'].float()   # amp 出来可能是 bf16, 评估用 fp32 更稳
        has_legal = torch.isfinite(logits).any(dim=-1)
        topk_5 = logits.topk(5, dim=-1).indices
        all_pred_top1.append(topk_5[:, 0].cpu())
        all_pred_top3.append(topk_5[:, :3].cpu())
        all_pred_top5.append(topk_5.cpu())
        all_true.append(batch['label_mark'].cpu())
        all_n_legal.append(batch['n_legal'].cpu())
        all_has_legal.append(has_legal.cpu())

        with torch.amp.autocast('cuda', enabled=use_amp and device.type=='cuda',
                                 dtype=amp_dtype):
            tau = torch.expm1(batch['label_dt']).clamp(min=1e-3)
            log_prob = model.time_head.log_prob(tau, out['h_H'])
        all_dt_pred_loglik.append(log_prob.float().cpu())

    top1 = torch.cat(all_pred_top1)
    top3 = torch.cat(all_pred_top3)
    top5 = torch.cat(all_pred_top5)
    y = torch.cat(all_true)
    nl = torch.cat(all_n_legal)
    hl = torch.cat(all_has_legal)

    valid = hl
    denom = valid.float().sum().clamp(min=1)
    acc1 = ((top1 == y) & valid).float().sum() / denom
    acc3 = ((top3 == y.unsqueeze(1)).any(dim=1) & valid).float().sum() / denom
    acc5 = ((top5 == y.unsqueeze(1)).any(dim=1) & valid).float().sum() / denom

    bins = [(1, 1), (2, 3), (4, 6), (7, 15), (16, 300)]
    layered = []
    for lo, hi in bins:
        m = ((nl >= lo) & (nl <= hi)) & valid
        if m.sum() == 0:
            layered.append({'bin': f'{lo}-{hi}', 'n': 0, 'top1': None})
        else:
            a = ((top1 == y) & m).float().sum() / m.float().sum()
            layered.append({'bin': f'{lo}-{hi}', 'n': int(m.sum()), 'top1': float(a)})

    time_nll = -torch.cat(all_dt_pred_loglik).mean()

    model.train()
    return dict(top1=float(acc1), top3=float(acc3), top5=float(acc5),
                time_nll=float(time_nll), layered=layered,
                n_valid=int(valid.sum()), n_total=int(len(y)))


def main(args):
    os.makedirs(args.out, exist_ok=True)
    setup_gpu_perf(enable_tf32=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    print_gpu_info()

    # ---- smoke test 覆盖默认 ----
    if args.smoke_test:
        print("[smoke-test] running with tiny config for quick verification")
        args.batch_size = min(args.batch_size, 16)
        args.epochs = 1
        args.val_max_batches = 20
        args.warmup = 5       # smoke 总共 31 步, warmup=5 确保 lr 快速爬到目标值
        args.lr = 1e-3        # smoke 要看到明显下降, 用较大 lr

    # ---- load graph ----
    g_hetero, meta = load_graph_hetero(args.graph_pkl)
    print(g_hetero)
    num_nodes_per_type = {nt: g_hetero[nt].num_nodes for nt in g_hetero.node_types}
    static_x = {nt: g_hetero[nt].x.to(device) for nt in g_hetero.node_types}
    edge_index_dict = {et: g_hetero[et].edge_index.to(device)
                       for et in g_hetero.edge_types}

    # ---- data ----
    ds_tr = DerbyDataset(args.samples_dir, 'train', relax_mask_in_train=True)
    ds_va = DerbyDataset(args.samples_dir, 'val')
    ds_te = DerbyDataset(args.samples_dir, 'test')

    if args.smoke_test:
        # 只取前 500 / 200 / 200 做冒烟
        from torch.utils.data import Subset
        ds_tr = Subset(ds_tr, list(range(min(500, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(200, len(ds_va)))))
        ds_te = Subset(ds_te, list(range(min(200, len(ds_te)))))

    pin = device.type == 'cuda'
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn,
                       drop_last=True, pin_memory=pin,
                       persistent_workers=args.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_fn,
                       pin_memory=pin,
                       persistent_workers=args.num_workers > 0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate_fn,
                       pin_memory=pin,
                       persistent_workers=args.num_workers > 0)

    # ---- model ----
    node_feat_dims = {
        'berth':  g_hetero['berth'].x.shape[1] + 5,
        'tc':     g_hetero['tc'].x.shape[1] + 2,
        'route':  g_hetero['route'].x.shape[1] + 2,
        'signal': g_hetero['signal'].x.shape[1] + 2,
    }
    print(f"[model] node_feat_dims = {node_feat_dims}")

    # vocab 长度: smoke-test 模式下 ds_tr 是 Subset, 要从原对象取
    base_tr = ds_tr.dataset if hasattr(ds_tr, 'dataset') else ds_tr
    n_type = len(base_tr.vocab['type'])
    n_id = len(base_tr.vocab['id'])
    n_train_tok = len(base_tr.vocab['train'])
    print(f"[model] vocabs: type={n_type} id={n_id} train={n_train_tok}")

    # 从一个样本推断 mov_dim 和 K — 自动兼容 v1 (23/64) 和 v2 (35/128)
    _probe = base_tr[0]
    mov_dim_inferred = _probe['mov'].shape[-1]
    K_inferred = _probe['events_cat'].shape[0]
    print(f"[model] inferred from data: mov_dim={mov_dim_inferred}  K={K_inferred}")

    model = DerbyModel(
        node_feat_dims=node_feat_dims,
        metadata=g_hetero.metadata(),
        n_type=n_type, n_id=n_id, n_train=n_train_tok,
        mov_dim=mov_dim_inferred,
        d=args.d, n_hgt_layers=args.hgt_layers,
        n_tf_layers=args.tf_layers,
        n_heads=args.heads, K=K_inferred, n_mix=args.n_mix,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params")

    # ---- 方案 A: 按 (area, headcode_class) 组合计算样本权重 ----
    sample_weights_tensor = None
    combo_names = None
    if args.class_weights:
        import numpy as _np
        base_tr_ds = ds_tr.dataset if hasattr(ds_tr, 'dataset') else ds_tr
        # 统计训练集内每个组合的样本数
        train_combos = base_tr_ds.combo_idx_all[base_tr_ds.idx]
        n_combo = len(base_tr_ds.combo_names)
        n_per = _np.bincount(train_combos, minlength=n_combo)
        median_n = float(_np.median(n_per[n_per > 0]))
        w = median_n / (n_per.astype(_np.float32) + 1e-6)
        w = _np.clip(w, args.class_weight_min, args.class_weight_max)
        sample_weights_tensor = torch.tensor(w, dtype=torch.float32)
        combo_names = base_tr_ds.combo_names
        print(f"[class-weights] median_n={median_n:.0f}  "
              f"clip=[{args.class_weight_min},{args.class_weight_max}]")
        print(f"[class-weights] per (area|hc) weights (train):")
        for name, nc, ww in sorted(zip(combo_names, n_per, w), key=lambda x: -x[1]):
            print(f"  {name:25s} n={nc:<6d} w={ww:.3f}")

    # ---- AMP setup ----
    # bfloat16 on Ampere+, fp16 + GradScaler otherwise (older cards)
    use_amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16
    use_scaler = False
    if use_amp:
        if torch.cuda.is_bf16_supported():
            print("[amp] using bfloat16 (no scaler needed)")
            amp_dtype = torch.bfloat16
        else:
            print("[amp] using fp16 + GradScaler")
            amp_dtype = torch.float16
            use_scaler = True
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # ---- optimizer + scheduler ----
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(dl_tr)
    total_steps = steps_per_epoch * args.epochs
    sched = LambdaLR(optim, lr_lambda=lambda s: warmup_cosine(s, args.warmup, total_steps))

    best_val_top1 = -1.0
    history = []
    log_f = open(os.path.join(args.out, 'train.log'), 'w')
    step = 0

    def _log(msg):
        print(msg); log_f.write(msg + '\n'); log_f.flush()

    _log(f"[train] {steps_per_epoch} steps/epoch, {total_steps} total, "
         f"amp={use_amp} amp_dtype={amp_dtype}")

    for epoch in range(args.epochs):
        t0 = time.time()
        running = {'loss': 0, 'mark_ce': 0, 'time_nll': 0, 'acc1': 0, 'n': 0}
        model.train()

        for bi, batch in enumerate(dl_tr):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
                logits = out['logits']

                has_legal = torch.isfinite(logits).any(dim=-1)
                if has_legal.sum() == 0:
                    continue

                # mark loss  (方案 A: class-weighted)
                log_p = F.log_softmax(logits, dim=-1)
                if sample_weights_tensor is not None:
                    w_batch = sample_weights_tensor.to(log_p.device)[batch['combo_idx']]
                    mark_ce_per = F.nll_loss(log_p[has_legal],
                                              batch['label_mark'][has_legal],
                                              reduction='none')
                    mark_ce = (mark_ce_per * w_batch[has_legal]).sum() / \
                               w_batch[has_legal].sum().clamp(min=1e-6)
                else:
                    mark_ce = F.nll_loss(log_p[has_legal],
                                          batch['label_mark'][has_legal])

                # time loss
                tau = torch.expm1(batch['label_dt']).clamp(min=1e-3)
                time_ll = model.time_head.log_prob(tau[has_legal], out['h_H'][has_legal])
                time_nll = -time_ll.mean()

                loss = mark_ce + args.time_weight * time_nll

            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            sched.step()
            step += 1

            # 统计
            n = int(has_legal.sum().item())
            running['loss'] += loss.item() * n
            running['mark_ce'] += mark_ce.item() * n
            running['time_nll'] += time_nll.item() * n
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                running['acc1'] += (pred == batch['label_mark'])[has_legal].float().sum().item()
            running['n'] += n

            if step % args.log_every == 0:
                _log(f"ep {epoch} step {step} lr={sched.get_last_lr()[0]:.2e} "
                     f"loss={running['loss']/running['n']:.4f} "
                     f"mark={running['mark_ce']/running['n']:.4f} "
                     f"time={running['time_nll']/running['n']:.4f} "
                     f"acc1={running['acc1']/running['n']:.4f}")
                if step % (args.log_every * 4) == 0:
                    print_gpu_mem(f"step {step}")

        # ---- validate ----
        val_rep = evaluate(model, dl_va, static_x, edge_index_dict,
                           num_nodes_per_type, device,
                           max_batches=args.val_max_batches,
                           use_amp=use_amp, amp_dtype=amp_dtype)
        _log(f"[ep {epoch}] val top1={val_rep['top1']:.4f} top3={val_rep['top3']:.4f} "
             f"top5={val_rep['top5']:.4f} time_nll={val_rep['time_nll']:.4f}  "
             f"({time.time()-t0:.0f}s)")
        history.append({'epoch': epoch, **val_rep,
                        'train_loss': running['loss']/max(1,running['n']),
                        'train_acc1': running['acc1']/max(1,running['n'])})
        if val_rep['top1'] > best_val_top1:
            best_val_top1 = val_rep['top1']
            torch.save({'model': model.state_dict(),
                         'epoch': epoch, 'val_rep': val_rep,
                         'args': vars(args),
                         'node_feat_dims': node_feat_dims,
                         'vocabs': base_tr.vocab,
                         'mov_dim': mov_dim_inferred,
                         'K': K_inferred,
                         'combo_names': combo_names},
                        os.path.join(args.out, 'best.pt'))
            _log(f"  -> saved best to {args.out}/best.pt")

    # ---- test with best checkpoint ----
    _log("\n[test] loading best checkpoint...")
    ckpt = torch.load(os.path.join(args.out, 'best.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_rep = evaluate(model, dl_te, static_x, edge_index_dict,
                         num_nodes_per_type, device,
                         use_amp=use_amp, amp_dtype=amp_dtype)
    _log(f"[test] top1={test_rep['top1']:.4f} top3={test_rep['top3']:.4f} "
         f"top5={test_rep['top5']:.4f} time_nll={test_rep['time_nll']:.4f}")
    for d in test_rep['layered']:
        _log(f"  n_legal in {d['bin']}: n={d['n']}  top1={d['top1']}")

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump({'history': history, 'test': test_rep,
                   'config': vars(args)}, f, indent=2)
    log_f.close()
    print_gpu_mem("final")
    print(f"\n[done] artifacts in {args.out}/")
    print(f"  best.pt        — trained model")
    print(f"  history.json   — metrics + config")
    print(f"  train.log      — training log")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-dir', required=True)
    ap.add_argument('--graph-pkl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--hgt-layers', type=int, default=2)
    ap.add_argument('--tf-layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--n-mix', type=int, default=4)
    ap.add_argument('--time-weight', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--val-max-batches', type=int, default=None)
    ap.add_argument('--amp', action='store_true', help='启用 AMP (bfloat16 on Ampere+)')
    ap.add_argument('--class-weights', action='store_true',
                    help='enable per-(area, headcode) weighted loss (方案 A)')
    ap.add_argument('--class-weight-min', type=float, default=0.5,
                    help='lower clip for class weights (default 0.5)')
    ap.add_argument('--class-weight-max', type=float, default=3.0,
                    help='upper clip for class weights (default 3.0)')
    ap.add_argument('--smoke-test', action='store_true', help='跑 500 样本快速验证')
    a = ap.parse_args()
    main(a)
