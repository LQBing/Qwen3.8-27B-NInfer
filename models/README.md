# models/ — NInfer 工件清单

本目录存放 4 份 `.ninfer` 工件。`.ninfer` 是 NInfer 自研容器格式（magic `NINFER\0` + 版本字节 + JSON 目录 + 量化张量载荷）。

> ⚠️ 本目录被 `.gitignore` 忽略（`models`），故本 README 仅在本地可见。

## 1. 工件总表

| 文件 | 大小 (GiB) | 精确字节 | 容器版本 | 能否直接加载 | 血统 / 来源 |
|---|---|---|---|---|---|
| `qwen3_8_27b_nvfp4.ninfer` | 22.09 | 23,719,715,844 | **v3** | ✅ | 官方 `neroued/Qwen3.8-27B-nvfp4-NInfer`，源自 `unsloth/Qwen3.8-27B-NVFP4` |
| `qwen3_8_27b.ninfer` | 19.03 | 20,437,521,664 | **v3** | ✅ | 官方 `neroued/Qwen3.8-27B-NInfer`（`groupwise-int`） |
| `qwen3_8_27b_abliterated_nvfp4.ninfer` | 20.02 | 21,492,938,224 | **v3** | ✅ | 无护栏版（见下），由 `_original` 升级而来 |
| `qwen3_8_27b_abliterated_original.ninfer` | 20.02 | 21,492,695,040 | **v2** | ❌ | 原始下载件，`lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4` |

## 2. 校验和（SHA256）

| 文件 | SHA256 |
|---|---|
| `qwen3_8_27b_nvfp4.ninfer` | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| `qwen3_8_27b.ninfer` | `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da` |
| `qwen3_8_27b_abliterated_original.ninfer` | `f21f308d3b23ccd627071cd015e413db08deee4356643900518e2b251750fdc2` |
| `qwen3_8_27b_abliterated_nvfp4.ninfer` | `43bb451098b2053a2597af08e7c3484636a81724da0231f299b8e215c3ab4daf` |

校核：`Get-FileHash .\models\<文件> -Algorithm SHA256`（PowerShell）

## 3. 无护栏（abliterated）版：两个文件的关系

`_original` → `_abliterated_nvfp4` 是**同一份权重的两种容器框架**：

- 原始件是 **container v2**（2026-08 用旧版 NInfer 产出）；当前运行时（`lqbing/ninfer:latest`）会直接拒绝：
  `NInfer v2 artifact is not supported. Upgrade to v3 with: python3 tools/upgrade_ninfer_v2_to_v3.py INPUT OUTPUT`
- 用官方升级脚本重打包为 **container v3** 后即可加载。
- **权重逐字节相同**（载荷 sha256 均为 `5a1118ef…ecc5cb`），张量集合 1118/1118 完全一致；差别仅：
  1. 容器框架 v2 → v3（新增 `components` / `bindings` / `uses` / `metadata` / `provenance` / `files` 语义层）
  2. 末尾追加官方维护版 `qwen3_8.jinja` chat template（9,712 B）
  3. 体积 +243,184 B

验证脚本：`tools/compare_abliterated_artifacts.py`（只读，逐字节比对）。

升级复现：`bash tools/do_upgrade.sh`（WSL；依赖 `tools/upgrade_ninfer_v2_to_v3.py` + `tools/chat_templates/qwen3_8.jinja`）。

## 4. 启动配置对应关系

| 工件 | compose | 端口 |
|---|---|---|
| `qwen3_8_27b_nvfp4.ninfer` | `docker-compose.yaml` | 30000 |
| `qwen3_8_27b.ninfer` | `docker-compose-no-nvfp4.yaml` | 30000 |
| `qwen3_8_27b_abliterated_nvfp4.ninfer` | `docker-compose-abliterated.yaml` | 30000 |

> 所有 compose 都占 30000 且共用同一张 GPU，**互斥**，切换需先 `down` 再 `up`。
> 三者服务名均为 `qwen3.8-27b`（无护栏版未覆盖 `--model-id`），故对上层客户端可无感互换。

## 5. 实测结论（详见 data/ 与 docs/）

- **性能**：无护栏版与官方版在相同配置下**无差异**（各场景 ±1~2%）。见 `data/abliterated-vs-official.md`。
- **质量**：官方 36/40 vs 无护栏 30/40，差距集中在"更啰嗦、token 预算效率低"导致的长任务截断，而非能力退化。见 `data/quality-gap-abliterated-vs-official.md`。
- **坑**：`--vision` 会额外占 ~0.6 GiB 固定显存，使 prefill/decode 约腰斩；纯文本场景**不要开**。

## 6. 磁盘占用

4 份工件合计 **≈ 81.16 GiB**。其中 `_original`（20.02 GiB）为原始下载件，**按需保留**：升级结果已验证且可复现（见 §3），若要腾空间可删。
