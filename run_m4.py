"""
run_m4.py — Windows/Linux/Mac 跨平台训练启动脚本
==================================================
替代 run_m4.sh, 不依赖 bash, Windows PowerShell 能直接跑.

用法:
    python run_m4.py smoke       # 10 分钟冒烟测试
    python run_m4.py small       # 快速实验 (d=64, 2 epochs)
    python run_m4.py full        # 默认 (d=128, 5 epochs, ~1-3h)
    python run_m4.py large       # 最终数字 (d=192, 10 epochs)
"""
import sys, os, subprocess, datetime, shutil

MODES = {
    'smoke': dict(
        smoke_test=True,
        batch_size=16, d=64, epochs=1,
        num_workers=0, log_every=5,   # Windows: num_workers=0 最稳
    ),
    'small': dict(
        batch_size=64, d=64, epochs=2,
        hgt_layers=2, tf_layers=2, heads=4,
        lr=5e-4, warmup=300,
        amp=True, num_workers=0, log_every=50,
    ),
    'full': dict(
        # 针对 8GB 显存优化: batch=32, d=128.
        # 如显存充裕 (>=12GB), 可手动改 batch_size=64.
        batch_size=32, d=128, epochs=5,
        hgt_layers=2, tf_layers=3, heads=4,
        lr=3e-4, warmup=500,
        weight_decay=0.01, time_weight=0.1,
        amp=True, num_workers=0, log_every=50,
    ),
    'full_tuned': dict(
        # 方案 A 之前的 "full+": 更长训练 + 更强正则 + time_weight 提高
        # 适用于 v1 数据 (out_samples, mov_dim=23, K=64)
        batch_size=32, d=128, epochs=8,
        hgt_layers=2, tf_layers=3, heads=4,
        lr=3e-4, warmup=500,
        weight_decay=0.05, time_weight=0.3,
        amp=True, num_workers=0, log_every=50,
    ),
    'full_ace': dict(
        # 方案 A+C+E 联合: 加权 loss + K=128 + mov_dim=35
        # 必须配合 out_samples_v2 (K=128 重跑的新数据)
        # 用法: python run_m4.py full_ace --samples-dir out_samples_v2
        batch_size=32, d=128, epochs=5,
        hgt_layers=2, tf_layers=3, heads=4,
        lr=3e-4, warmup=500,
        weight_decay=0.01, time_weight=0.1,
        class_weights=True,
        amp=True, num_workers=0, log_every=50,
    ),
    'large': dict(
        batch_size=128, d=192, epochs=10,
        hgt_layers=3, tf_layers=4, heads=6,
        lr=2e-4, warmup=1000,
        amp=True, num_workers=0,
    ),
}


def build_cmd(mode, out_dir, samples_dir='./out_samples',
               graph_pkl='./out_graph/graph.pkl'):
    cfg = MODES[mode]
    cmd = [
        sys.executable, 'm4_train.py',
        '--samples-dir', samples_dir,
        '--graph-pkl',   graph_pkl,
        '--out',         out_dir,
    ]
    for k, v in cfg.items():
        arg_name = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                cmd.append(arg_name)
        else:
            cmd.append(arg_name)
            cmd.append(str(v))
    return cmd


def main():
    import argparse
    ap = argparse.ArgumentParser(
        usage='python run_m4.py <mode> [--samples-dir DIR] [--graph-pkl PATH]',
        description='Shortcut launcher for m4_train.py. '
                    'Mode presets in MODES dict.')
    ap.add_argument('mode', choices=list(MODES.keys()),
                    help='preset mode: ' + ' | '.join(MODES.keys()))
    ap.add_argument('--samples-dir', default='./out_samples',
                    help='data dir (default ./out_samples; 用方案 ACE 时改为 ./out_samples_v2)')
    ap.add_argument('--graph-pkl', default='./out_graph/graph.pkl')
    args = ap.parse_args()

    mode = args.mode
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('m4_runs', f'{ts}_{mode}')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Derby M4 training — mode={mode}")
    print(f"output:      {out_dir}")
    print(f"samples-dir: {args.samples_dir}")
    print(f"graph-pkl:   {args.graph_pkl}")
    print("=" * 60)

    # 提示: 先检查 nvidia-smi (如果有)
    if shutil.which('nvidia-smi'):
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
            )
            print(f"GPU: {r.stdout.strip()}")
        except Exception:
            pass
    print()

    # 检查数据目录
    for p in [os.path.join(args.samples_dir, 'samples.csv'), args.graph_pkl]:
        if not os.path.exists(p):
            print(f"ERROR: {p} 不存在.")
            print(f"  请确认 Module 2 产出在 {args.samples_dir}/, Module 1 在 {args.graph_pkl}")
            sys.exit(1)

    cmd = build_cmd(mode, out_dir, args.samples_dir, args.graph_pkl)
    print("Running:")
    print("  " + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
    print()

    # 实时打印输出 (不捕获, 直接给到终端)
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[interrupted] training stopped by user")
        sys.exit(130)

    print()
    print("=" * 60)
    if rc == 0:
        print("Training done. Artifacts:")
        for f in sorted(os.listdir(out_dir)):
            fp = os.path.join(out_dir, f)
            sz = os.path.getsize(fp) / 1e6
            print(f"  {f}  ({sz:.2f} MB)")
        print()
        print("To package results for sharing:")
        print(f"  python package_results.py {out_dir}")
    else:
        print(f"Training exited with code {rc}")
        print(f"Check logs: {os.path.join(out_dir, 'train.log')}")
    sys.exit(rc)


if __name__ == '__main__':
    main()
