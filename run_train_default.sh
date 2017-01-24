MODEL_DESCRIPTION=default model=wide-resnet widen_factor=10 depth=28 dropout=0.3 batchSize=128 dataset=./datasets/cifar100_whitened.t7 ./scripts/train_cifar.sh
