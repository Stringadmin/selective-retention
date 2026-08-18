#!/bin/bash
# Install sentence-transformers and pre-download a Chinese BGE embedding model
# from ModelScope (HuggingFace is network-blocked in this environment).
set -e
PY=~/fip-venv/bin/python
$PY -m pip install -q sentence-transformers -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -4
echo "=== download BGE via modelscope ==="
$PY - <<'PYEOF'
import os
os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
try:
    from modelscope.hub.snapshot_download import snapshot_download
    p = snapshot_download("BAAI/bge-small-zh-v1.5")
    print("BGE downloaded to:", p)
except Exception as e:
    print("modelscope download failed:", e)
PYEOF
echo "=== verify sentence-transformers ==="
$PY -c "import sentence_transformers; print('st', sentence_transformers.__version__)"
