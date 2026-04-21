"""
profile_m4.py — 找出训练的瓶颈
=================================
在你电脑上跑:  python profile_m4.py

它会:
  1) 分别测量 数据加载 / CPU->GPU / forward / backward / 整体 的耗时
  2) 测试 num_workers = 0, 2, 4 对性能的影响
  3) 给出针对你硬件的建议
"""
import sys, time, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from m4_dataset import DerbyDataset, load_graph_hetero, collate_fn
from m4_model import DerbyModel


def check_gpu():
    if not torch.cuda.is_available():
        print("GPU 不可用"); sys.exit(1)
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  |  {p.total_memory/1e9:.1f}GB  |  cap={p.major}.{p.minor}")


def time_section(name, fn, n=3, sync=True):
    # 预热
    for _ in range(2):
        fn()
    if sync: torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if sync: torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  {name:35s} {dt*1000:8.1f} ms")
    return dt


def main():
    check_gpu()
    device = torch.device('cuda')

    print("\n[setup] loading graph + dataset...")
    t0 = time.time()
    g, _ = load_graph_hetero('./out_graph/graph.pkl')
    num_nodes_per_type = {nt: g[nt].num_nodes for nt in g.node_types}
    static_x = {nt: g[nt].x.to(device) for nt in g.node_types}
    edge_index_dict = {et: g[et].edge_index.to(device) for et in g.edge_types}
    ds = DerbyDataset('./out_samples', 'val')
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("\n[setup] building model...")
    model = DerbyModel(
        node_feat_dims={
            'berth': g['berth'].x.shape[1] + 5,
            'tc': g['tc'].x.shape[1] + 2,
            'route': g['route'].x.shape[1] + 2,
            'signal': g['signal'].x.shape[1] + 2,
        },
        metadata=g.metadata(),
        n_type=len(ds.vocab['type']),
        n_id=len(ds.vocab['id']),
        n_train=len(ds.vocab['train']),
        d=64, n_hgt_layers=2, n_tf_layers=2, n_heads=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.2f}M params")

    # ========== 测试 1: 把数据都搬到 GPU, 只测计算 ==========
    print("\n[test 1] Pure GPU compute speed (no data loading, no CPU->GPU)")
    print("-" * 60)
    BS = 16
    sub = Subset(ds, list(range(BS)))
    dl = DataLoader(sub, batch_size=BS, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))
    batch = {k: v.to(device) for k, v in batch.items()}

    def do_forward():
        out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
        return out

    def do_full_step():
        out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
        has = torch.isfinite(out['logits']).any(-1)
        loss = F.nll_loss(F.log_softmax(out['logits'], -1)[has],
                           batch['label_mark'][has])
        tau = torch.expm1(batch['label_dt']).clamp(min=1e-3)
        loss = loss + 0.1 * (-model.time_head.log_prob(tau[has], out['h_H'][has]).mean())
        return loss

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def do_full_step_backward():
        loss = do_full_step()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    t_fwd = time_section("forward only", do_forward)
    t_full = time_section("forward + loss", do_full_step)
    t_step = time_section("forward + loss + backward + step", do_full_step_backward)
    print(f"\n  pure GPU upper bound: {1/t_step:.1f} steps/sec")

    # ========== 测试 2: 不同 num_workers ==========
    print("\n[test 2] DataLoader with different num_workers")
    print("-" * 60)
    sub_big = Subset(ds, list(range(BS * 10)))

    for nw in [0, 2, 4]:
        try:
            dl = DataLoader(sub_big, batch_size=BS, collate_fn=collate_fn,
                            num_workers=nw, shuffle=False,
                            pin_memory=True,
                            persistent_workers=nw > 0)
            # 预热一个 epoch (启动 workers)
            it = iter(dl)
            next(it); next(it)

            t0 = time.time()
            count = 0
            for batch in dl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model(batch, static_x, edge_index_dict, num_nodes_per_type)
                has = torch.isfinite(out['logits']).any(-1)
                loss = F.nll_loss(F.log_softmax(out['logits'], -1)[has],
                                   batch['label_mark'][has])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                count += 1
                if count >= 5: break
            torch.cuda.synchronize()
            dt = (time.time() - t0) / count
            print(f"  num_workers={nw}: {dt*1000:.0f} ms/step  ({1/dt:.1f} steps/sec)")
            del dl
        except Exception as e:
            print(f"  num_workers={nw}: FAILED {e}")

    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pure GPU compute upper bound: {1/t_step:.1f} steps/sec")
    print(f"Your training should achieve >80% of this when num_workers is tuned.")
    print()
    print("如果 num_workers=0 比 num_workers>0 快: Windows 多进程 overhead 严重")
    print("  建议: full 训练用 --num-workers 0")
    print()
    print(f"预估 full 训练时间:")
    steps_per_ep = 138434 // 32
    best_throughput = 1 / t_step
    for nw_case in [0, 2, 4]:
        print(f"  (假设 {best_throughput:.1f} steps/sec): "
              f"1 epoch = {steps_per_ep/best_throughput/60:.0f} min, "
              f"5 epochs = {steps_per_ep*5/best_throughput/60:.0f} min")
        break


if __name__ == '__main__':
    main()
