#!/usr/bin/env bash
export HOME=/home/omnichat
source /home/omnichat/fip-venv/bin/activate
echo "=== install torch cu128 via find-links ==="
pip install --no-input "torch==2.10.0+cu128" \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ 2>&1 | tail -20
echo "EXIT=${PIPESTATUS[0]}"
python -c "import torch; print('OK', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 || echo "torch import failed"
