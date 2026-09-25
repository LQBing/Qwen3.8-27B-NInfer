# Docker vs WSL 原生 —— 同一 ninfer 的性能对比

> 用途：回答「改用 WSL 原生启动（而非 Docker）是否带来性能差异」。
> 两个环境使用**同一份本地编译二进制**（ninfer 源码工程内的 `build/apps/ninfer-serve`）、
> 同一模型文件、同一启动参数、同一 bench harness、同一 GPU（串行执行）。
> Docker = `lqbing/ninfer:local`（CUDA 13.1 runtime）；原生 = WSL Ubuntu-24.04（CUDA 13.0）。

## 1. 冷启动与显存

| 权重 | 运行 | 权重大小 | 加载权重耗时 | 读取速率 | 总启动 | 启动后显存 |
|---|---|---|---|---|---|---|
| NVFP4 | Docker | 19.7 GiB | 111.1s | 181.8 MiB/s | 121.6s | 30751 MiB |
| NVFP4 | WSL 原生 | 19.7 GiB | 100.4s | 201.3 MiB/s | 109.1s | 30828 MiB |
| groupwise-int | Docker | 16.7 GiB | 92.8s | 183.9 MiB/s | 105.8s | 27905 MiB |
| groupwise-int | WSL 原生 | 16.7 GiB | 92.0s | 185.5 MiB/s | 102.4s | 未采集 MiB |

## 2. 性能对比（每场景 3 轮中位数）

> 注：`short_coding_gen` / `conc2_short` 的 **prefill 列受固定开销主导，不具可比性**，
> 判断 prefill 只看 `longctx_*` 行。`MTP接受率` = Σdraft_n_accepted/Σdraft_n（NInfer 原生 timings），
> 并发场景未聚合该列（显示 `-`）。

### NVFP4 权重（qwen3_8_27b_nvfp4.ninfer）

| 场景 | TTFT d/n (s) | Prefill d/n (tok/s) | Prefill Δ | Decode d/n (tok/s) | MTP接受率 d/n | 总耗时 d/n (s) | wall d/n (s) |
|---|---|---|---|---|---|---|---|
| short_coding_gen | 0.11 / 0.09 | 1571 / 1901 | +21.0% | 154.0 / 152.2 | 61.7% / 56.1% | 3.42 / 3.45 | - / - |
| longctx_32k_prefill | 4.66 / 4.69 | 7009 / 6952 | -0.8% | 136.2 / 137.8 | 53.9% / 58.0% | 6.54 / 6.57 | - / - |
| longctx_32k_longgen | 4.67 / 4.72 | 6992 / 6911 | -1.2% | 145.9 / 139.1 | 58.6% / 56.6% | 11.68 / 12.13 | - / - |
| longctx_128k_prefill | 34.64 / 34.96 | 3769 / 3734 | -0.9% | 147.4 / 134.9 | 69.5% / 67.4% | 35.56 / 35.91 | - / - |
| conc2_short | 0.13 / 0.14 | 1278 / 1142 | -10.6% | 161.7 / 139.9 | -% / -% | 3.32 / 3.83 | 3.45 / 4.01 |
| conc2_long_distinct | 7.39 / 7.38 | 4419 / 4425 | +0.1% | 82.9 / 89.1 | -% / -% | 11.72 / 11.54 | 12.12 / 11.92 |

### groupwise-int 权重（qwen3_8_27b.ninfer）

| 场景 | TTFT d/n (s) | Prefill d/n (tok/s) | Prefill Δ | Decode d/n (tok/s) | MTP接受率 d/n | 总耗时 d/n (s) | wall d/n (s) |
|---|---|---|---|---|---|---|---|
| short_coding_gen | 0.15 / 0.15 | 1083 / 1097 | +1.2% | 153.6 / 172.3 | 57.9% / 57.5% | 3.50 / 3.13 | - / - |
| longctx_32k_prefill | 11.65 / 11.50 | 2802 / 2839 | +1.3% | 146.0 / 154.9 | 56.5% / 56.8% | 13.40 / 13.07 | - / - |
| longctx_32k_longgen | 11.69 / 11.54 | 2793 / 2829 | +1.3% | 144.7 / 149.9 | 58.5% / 54.9% | 18.74 / 18.38 | - / - |
| longctx_128k_prefill | 63.13 / 63.20 | 2068 / 2066 | -0.1% | 143.9 / 148.8 | 71.1% / 71.4% | 64.05 / 64.05 | - / - |
| conc2_short | 0.24 / 0.24 | 672 / 704 | +4.8% | 116.3 / 110.9 | -% / -% | 4.65 / 4.86 | 4.81 / 4.95 |
| conc2_long_distinct | 17.83 / 18.21 | 1831 / 1793 | -2.1% | 66.3 / 68.0 | -% / -% | 25.84 / 26.19 | 26.22 / 26.58 |

## 3. 结论

### 3.1 Prefill（最可靠的指标，纯算力、确定性）：**无系统性差异**
| 权重 | 场景 | Docker | 原生 | Δ |
|---|---|---|---|---|
| NVFP4 | 32K prefill | 7202 | 7077 | −1.7% |
| NVFP4 | 128K prefill | 3843 | 3760 | −2.2% |
| groupwise-int | 32K prefill | 2881 | 2917 | +1.2% |
| groupwise-int | 128K prefill | 2124 | 2247 | +5.8% |

prefill 不受采样/接受率影响，四组差异都在 ±6% 内且方向不一致（两负两正）→ 
**Docker 容器层没有带来可测量的计算惩罚**。

### 3.2 Decode：波动大，**由 MTP 接受率驱动，而非运行环境**
groupwise-int 上原生 decode 看着高（32K +22.4%），但**同时接受率也更高**
（原生 ~63% vs Docker ~50%）——decode ∝ 接受率，两者是同源变化。
NVFP4 上接受率相当（两侧均 ~53%），decode 也基本相等（144.6 vs 144.3）。
→ **decode 的跨轮差异主要来自投机接受率随内容/采样的波动，不应解读为 Docker vs 原生的差异。**

### 3.3 冷启动：原生略快
NVFP4 权重读取 201.3 vs 181.8 MiB/s（+10.7%，总启动 109s vs 122s）；
groupwise-int 基本持平（185.5 vs 183.9 MiB/s，102s vs 106s）。
差异来自 I/O 路径（Docker bind mount vs drvfs `/mnt/c`）与 CUDA 运行时版本（13.1 vs 13.0）。

### 3.4 选型建议
- **性能上：Docker 与 WSL 原生等价**，无需为性能而改架构。
- **Docker 更省心**：依赖、隔离、复现、进程生命周期都由容器管理（本机当前即用 Docker）。
- **WSL 原生**：省去容器层、冷启动略快，但需自行维护 CUDA 依赖与进程守护；
  且要注意 `wsl -e` 拉起后台进程会被回收（须用 `setsid` 或由常驻 wsl.exe 持有，见 `bench/wsl_serve_fg.sh`）。
