#!/bin/bash
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
for L in 512 1024 2048; do
  echo "==================== lambda=$L  start: $(date '+%F %T') ===================="
  RESUME=""
  [ -f "checkpoints/psnr_l$L/latest.pth" ] && RESUME="--resume checkpoints/psnr_l$L/latest.pth" && echo "(resume checkpoints/psnr_l$L/latest.pth)"
  ./venv/bin/python -u train.py --lmbda $L --metric psnr --epochs 50 --batch_size 6 \
      --data_root data/real --height 256 --width 448 \
      --checkpoint checkpoints/psnr_l$L $RESUME
  echo "==================== lambda=$L  done:  $(date '+%F %T') ===================="
done
echo "ALL TRAINING DONE: $(date '+%F %T')"
