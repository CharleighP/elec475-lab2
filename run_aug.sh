set -eo pipefail

python train.py --model alexnet --norm bn --dataset cifar10 --epochs 15 --no-augment --seed 42 \
  --save-model checkpoints/alexnet_no_aug.pth --plot-file plots/loss_no_aug.png \
  2>&1 | tee logs/alexnet_no_aug.txt

python train.py --model alexnet --norm bn --dataset cifar10 --epochs 15 --seed 42 \
  --save-model checkpoints/alexnet_with_aug.pth --plot-file plots/loss_with_aug.png \
  2>&1 | tee logs/alexnet_with_aug.txt