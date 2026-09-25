#!/usr/bin/env bash
# 前台启动 ninfer-serve（由 Windows 侧 Start-Process wsl.exe 持有，保证不被回收）
# 用法: wsl_serve_fg.sh <model.ninfer> <tag>
#
# 路径约定（无绝对路径）：
#   本脚本位于 <项目>/bench/，ninfer 源码工程与 <项目> 同级
#   PROJECT_DIR = <项目>（＝脚本上级目录）
#   NINFER_BIN  = <项目>/../ninfer/build/apps/ninfer-serve
# 可用环境变量覆盖：PROJECT_DIR / NINFER_BIN / LOGDIR
#
# 注意：下面的启动参数对应“native vs docker”对比时的基线配置；
#       要复现当前 compose 配置，请让参数与其保持一致。
set -u
MODEL="$1"
TAG="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
BIN="${NINFER_BIN:-$PROJECT_DIR/../ninfer/build/apps/ninfer-serve}"
LOGDIR="${LOGDIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOGDIR"

if [ ! -x "$BIN" ]; then
  echo "[error] 找不到 ninfer-serve: $BIN（可用 NINFER_BIN 指定）" >&2
  exit 1
fi

pkill -f "ninfer-serve" 2>/dev/null
sleep 2

cd "$PROJECT_DIR" || exit 1
exec "$BIN" "$MODEL" \
  --host 0.0.0.0 --port 30000 \
  --max-context 240000 --kv-capacity 240000 --max-concurrency 2 \
  --kv-dtype fp8 --device-state-slots 2 --host-state-slots 8 \
  --host-kv-mib 8192 --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking \
  > "$LOGDIR/$TAG.log" 2>&1
