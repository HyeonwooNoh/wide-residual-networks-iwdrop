if [ "${GPU_ID}" != "0" ] && [ "${GPU_ID}" != "1" ] && [ "${GPU_ID}" != "2" ] && [ "${GPU_ID}" != "3" ]; then
	echo "GPU_ID: ${GPU_ID} is not a valid gpu_id, use [0, 1, 2, 3]"
fi

MODEL_DESCRIPTION=cifar100_original_reproduced_dropout_0.3_batch_size_128_depth_28_widen_10_seed_333 model=wide-resnet widen_factor=10 depth=28 dropout=0.3 batchSize=128 seed=333 dataset=./datasets/cifar100_original.t7 gpuid=${GPU_ID} ./scripts/train_cifar.sh
