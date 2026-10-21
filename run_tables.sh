set -eo pipefail

# Rows 1-2: LeNets on CIFAR-10 (T4 epoch times) + Tiny ImageNet
for M in lenet lenet_modern; do
  NAME=$M; [ "$M" = "lenet" ] && NAME=lenet_baseline
  python train.py --model $M --dataset cifar10 --epochs 10 --no-augment --seed 42 \
    --save-model checkpoints/${NAME}.pth --plot-file plots/loss_${NAME}.png 2>&1 | tee logs/${NAME}.txt
  python train.py --model $M --dataset tiny_imagenet --epochs 10 --no-augment --seed 42 \
    --save-model checkpoints/${NAME}_tiny.pth --plot-file plots/loss_${NAME}_tiny.png 2>&1 | tee logs/${NAME}_tiny.txt
done

# Rows 3-5: AlexNet variants on Tiny ImageNet (no aug)
for NORM in none lrn bn; do
  python train.py --model alexnet --norm $NORM --dataset tiny_imagenet --epochs 10 --no-augment --seed 42 \
    --save-model checkpoints/alexnet_${NORM}_tiny.pth --plot-file plots/loss_alexnet_${NORM}_tiny.png 2>&1 | tee logs/alexnet_${NORM}_tiny.txt
done

# Row 7: Zoo AlexNet on both datasets
for DS in cifar10 tiny_imagenet; do
  python train.py --model zoo_alexnet --dataset $DS --epochs 10 --no-augment --seed 42 \
    --save-model checkpoints/zoo_alexnet_${DS}.pth --plot-file plots/loss_zoo_alexnet_${DS}.png 2>&1 | tee logs/zoo_alexnet_${DS}.txt
done