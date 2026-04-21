# Windows + RTX 5070 Laptop 训练指南

你的 GPU: **RTX 5070 Laptop, 8GB VRAM, Blackwell 架构 (cap 12.x)**

## 重要: Blackwell 架构需要新版 PyTorch

RTX 50xx 系列（5070/5080/5090）是 **Blackwell (SM_120)** 架构。旧版 PyTorch 跑不了——会直接报 `no kernel image is available for execution on the device`。

**必须**安装 **PyTorch ≥ 2.7 + CUDA 12.8**：

```powershell
# 在 PyCharm 的 .venv 里卸掉旧版
pip uninstall torch torchvision torchaudio -y

# 装新版 (CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 其他依赖
pip install torch_geometric pandas numpy lightgbm
```

## 第一步：环境检查（1 分钟）

**在 PyCharm Terminal 里跑**（PowerShell 也行）：

```powershell
python check_env.py
```

预期看到：
```
PyTorch: 2.7.x+cu128
CUDA devices: 1
  [0] NVIDIA GeForce RTX 5070 Laptop GPU
      Compute capability: 12.0
      Memory: 8.0 GB
[注意] 检测到 Blackwell 架构 (cap >= 12)
  ✓ PyTorch 版本 OK
[test] running real GPU computation...
  ✓ matmul (1024x1024) OK
  ✓ bfloat16 supported
torch_geometric: 2.x.x  ✓
============================================================
✓ 环境检查通过, 可以开始训练:
    python run_m4.py full
============================================================
```

**如果报错**，按照脚本给的提示修，别直接跑训练。

## 第二步：冒烟测试（10 分钟）

```powershell
python run_m4.py smoke
```

这会跑 500 样本 1 epoch，确认整条 pipeline 在你的 GPU 上能跑通。**数字本身不重要**，出结果就行。

期望日志长这样：
```
[gpu] NVIDIA GeForce RTX 5070 Laptop GPU  |  8.0 GB  |  cap=12.0
[model] 0.58M params
[train] 31 steps/epoch, 31 total, amp=False
ep 0 step 10 ... loss=5.43 acc1=0.05
ep 0 step 20 ... loss=5.12 acc1=0.09
[ep 0] val top1=0.12 top3=0.31 top5=0.48 ... (35s)
[test] top1=0.11 top3=0.30 top5=0.47
```

## 第三步：正式训练

```powershell
python run_m4.py full
```

**8GB 显存下的默认配置**: `batch=32, d=128, 5 epochs`，预计 **2-4 小时**。

### 如果 OOM（显存不够）

改 `run_m4.py` 里的 `full` 配置:
```python
'full': dict(
    batch_size=16, d=96, ...   # 进一步缩小
)
```
或者直接调用 `m4_train.py`:
```powershell
python m4_train.py `
    --samples-dir .\out_samples `
    --graph-pkl   .\out_graph\graph.pkl `
    --out         m4_runs\exp1 `
    --batch-size 16 --d 96 --epochs 5 --amp --num-workers 4
```

注意 PowerShell 的续行符是**反引号** `` ` ``，不是 bash 的 `\`。

### 如果显存充裕（没 OOM）

可以把 `batch_size` 改回 64 来加速：
```python
'full': dict(
    batch_size=64, d=128, ...
)
```

## 第四步：查看 GPU 使用情况

另开一个 PowerShell 窗口，跑：
```powershell
nvidia-smi -l 1
```
（每秒刷新一次 GPU 状态）

训练跑起来后应该看到：
- GPU util 60-95%（低于 50% 说明数据加载是瓶颈，加 `--num-workers 6`）
- Memory used 6-7 GB（8GB 卡上合理，>7.5GB 可能 OOM）
- Temp 70-80℃（正常；>85℃ 长时间工作注意散热）

## 第五步：打包结果回传

训练结束后（屏幕会打印 `Training done. Artifacts: ...`）：

```powershell
python package_results.py m4_runs\20260421_143022_full
```

会生成 `20260421_143022_full_results.tar.gz`（10-30 MB），把这个文件传给我。

## 常见问题

### Q1: `bash run_m4.sh` 报 "/bin/bash not found"
就是你截图看到的问题。Windows 没 bash。**用 `python run_m4.py` 代替**。

### Q2: `no kernel image is available`
PyTorch 版本太旧，不支持 Blackwell。按最上面的步骤重装。

### Q3: DataLoader `num_workers` 在 Windows 上报错
Windows 的 multiprocessing 行为和 Linux 不同，极少数情况下 `num_workers>0` 会挂。如果遇到，改成 `--num-workers 0`（会慢点但稳）。

### Q4: PyCharm Terminal 执行 Python 脚本时用的不是 .venv
PyCharm Terminal 右下角会显示当前解释器。确保你看到 `Python 3.11 (GNN)` 或类似标记的 venv 环境（从截图看你已经是这个状态）。

### Q5: `nvidia-smi -l 1` 没反应
把 `-l 1` 换成 `--loop=1`，或者就直接 `nvidia-smi` 不带参数，看一次快照。

## 一个重要提醒

**先跑 `check_env.py`，再跑 `smoke` 模式，最后才跑 `full`。** 每一步都是在排除下一步可能出的问题。8GB 显存 + Blackwell 架构的组合要求比较严格，按顺序来不会浪费时间。

## 预期最终数字

在你的 RTX 5070 上跑 `full` 模式，预期 test top-1 在 **76-80%** 之间（基线 71.5%）。低于 73% 说明训练有问题，看日志里的 val 曲线走势给我看。
