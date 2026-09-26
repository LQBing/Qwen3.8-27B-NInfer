#!/usr/bin/env bash
# 复现 v2 -> v3 升级：从原始下载件重建可运行工件，并校验与现有件是否逐字节一致。
# 安全设计：不覆盖线上正在使用的模型文件，输出到 *.rebuilt.ninfer 后再比对。
set -u
cd /mnt/c/Tools/Qwen3.8-27B-NInfer || exit 1

IN=models/qwen3_8_27b_abliterated_original.ninfer
LIVE=models/qwen3_8_27b_abliterated_nvfp4.ninfer
OUT=models/qwen3_8_27b_abliterated_nvfp4.rebuilt.ninfer

[ -f "$IN" ] || { echo "FATAL: missing input $IN"; exit 1; }

echo "--- upgrade: $IN -> $OUT"
rm -f "$OUT"
start=$(date +%s)
python3 -u tools/upgrade_ninfer_v2_to_v3.py "$IN" "$OUT"
code=$?
end=$(date +%s)
echo "EXIT_CODE=$code  ELAPSED_SEC=$((end-start))"
[ "$code" -eq 0 ] || exit "$code"

echo "--- sha256 compare ---"
a=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "rebuilt : $a"
if [ -f "$LIVE" ]; then
  b=$(sha256sum "$LIVE" | cut -d' ' -f1)
  echo "in-place: $b"
  if [ "$a" = "$b" ]; then
    echo "MATCH: 升级可复现，重建件与现有件逐字节一致"
  else
    echo "DIFF: 与现有件不一致（检查 ninfer 版本 / chat template 版本）"
  fi
else
  echo "in-place: (absent) — 未找到 $LIVE"
fi
echo "提示：确认后清理临时件 -> rm -f \"$OUT\""
