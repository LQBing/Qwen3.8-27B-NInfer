#!/usr/bin/env bash
# 检查本地编译的 ninfer-serve 在 WSL 原生环境能否运行
# 无绝对路径：二进制路径由脚本位置推导（PROJECT_DIR/../ninfer/build/apps/ninfer-serve），
# 可用环境变量 NINFER_BIN 覆盖。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
B="${NINFER_BIN:-$PROJECT_DIR/../ninfer/build/apps/ninfer-serve}"

if [ ! -x "$B" ]; then
  echo "找不到可执行文件: $B（可用 NINFER_BIN 指定）"
  exit 1
fi

echo "== file =="
file "$B" 2>&1

echo "== ldd missing =="
ldd "$B" 2>&1 | grep -i "not found" || echo "none-missing"

echo "== ldd all =="
ldd "$B" 2>&1 | sort | head -40

echo "== ldd libcuda resolution =="
ldd "$B" 2>&1 | grep -i cuda || echo "no-cuda-entry"

echo "== run --help (exit code at end) =="
"$B" --help 2>&1 | head -50
echo "== exit=$? =="

echo "== nvidia lib path =="
ls /usr/lib/wsl/lib/libcuda.so.1 2>&1
