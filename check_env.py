"""
check_env.py — 训练前环境检查
跑这个确保你的 PyTorch 能识别 RTX 5070 (Blackwell, cap 12.0).

用法:
    python check_env.py
"""
import sys
import platform

print("=" * 60)
print("Environment check")
print("=" * 60)
print(f"Python: {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

# --- torch ---
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  CUDA built: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
except ImportError:
    print("ERROR: PyTorch 未安装")
    print("  请装 CUDA 12.8 版本:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

# --- GPU 检测 ---
if not torch.cuda.is_available():
    print("\nERROR: CUDA 不可用")
    print("  可能原因:")
    print("    1. 装的是 CPU 版 PyTorch")
    print("    2. CUDA 驱动版本过低")
    print("  查当前驱动支持的 CUDA 版本:  nvidia-smi")
    sys.exit(1)

print(f"\nCUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    cap = f"{p.major}.{p.minor}"
    print(f"  [{i}] {p.name}")
    print(f"      Compute capability: {cap}")
    print(f"      Memory: {p.total_memory/1e9:.1f} GB")
    print(f"      SMs: {p.multi_processor_count}")

# --- Blackwell (RTX 50xx) 兼容性检查 ---
maj = torch.cuda.get_device_properties(0).major
if maj >= 12:
    print(f"\n[注意] 检测到 Blackwell 架构 (cap >= 12)")
    tv = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    cv = torch.version.cuda or ''
    cv_ok = cv.startswith('12.8') or cv.startswith('12.9') or cv.startswith('13')
    if tv < (2, 7) or not cv_ok:
        print(f"  ⚠ 当前 PyTorch={torch.__version__}, CUDA={cv}")
        print(f"  ⚠ Blackwell (5070/5080/5090) 需要 PyTorch >= 2.7 + CUDA 12.8+")
        print(f"    卸载重装:")
        print(f"      pip uninstall torch torchvision torchaudio -y")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(2)
    else:
        print(f"  ✓ PyTorch 版本 OK")

# --- 在 GPU 上做一次真实计算, 确认内核可用 ---
print("\n[test] running real GPU computation...")
try:
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    z = x @ y
    torch.cuda.synchronize()
    print(f"  ✓ matmul (1024x1024) OK, result mean={z.mean().item():.4f}")
except Exception as e:
    print(f"  ✗ GPU computation FAILED: {e}")
    print("  这通常意味着 PyTorch 版本和 GPU 架构不匹配.")
    sys.exit(3)

# --- bfloat16 / AMP ---
if torch.cuda.is_bf16_supported():
    print("  ✓ bfloat16 supported (会自动使用 AMP)")
else:
    print("  ○ bfloat16 NOT supported (会走 fp16+scaler 或 fp32)")

# --- PyTorch Geometric ---
try:
    import torch_geometric
    from torch_geometric.nn import HGTConv
    from torch_geometric.data import HeteroData
    print(f"\ntorch_geometric: {torch_geometric.__version__}  ✓")
except ImportError as e:
    print(f"\nERROR: torch_geometric 未装: {e}")
    print("  pip install torch_geometric")
    sys.exit(4)

# --- 其他必要包 ---
missing = []
for pkg in ['pandas', 'numpy']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"\n缺少: {missing}")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(5)
else:
    print("pandas/numpy: ✓")

print("\n" + "=" * 60)
print("✓ 环境检查通过, 可以开始训练:")
print("    python run_m4.py full")
print("=" * 60)
