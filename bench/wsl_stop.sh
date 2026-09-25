#!/usr/bin/env bash
# 停止原生 ninfer-serve
set -u
if pkill -f "ninfer-serve"; then
  echo "killed ninfer-serve"
else
  echo "no ninfer-serve running"
fi
sleep 2
pgrep -af ninfer-serve || echo "confirmed: none running"
