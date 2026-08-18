#!/bin/bash
# Install llama-cpp-python and verify CUDA offload support.
set -e
PY=~/fip-venv/bin/python
$PY -m pip install llama-cpp-python -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -6
echo "=== import test ==="
$PY - <<'PYEOF'
from llama_cpp import Llama
import llama_cpp
print("llama_cpp import OK")
try:
    print("cuda offload:", llama_cpp.llama_supports_gpu_offload())
except Exception as e:
    print("cuda check err:", e)
PYEOF
