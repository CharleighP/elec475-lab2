set -eo pipefail

for NORM in none lrn bn; do
  python train.py --model alexnet --norm $NORM --dataset cifar10 --epochs 10 \
    --no-augment --seed 42 \
    --save-model checkpoints/alexnet_${NORM}.pth \
    --plot-file plots/loss_alexnet_${NORM}.png \
    2>&1 | tee logs/alexnet_${NORM}.txt
done