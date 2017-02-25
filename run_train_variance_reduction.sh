if [ "${GPU_ID}" != "0" ] && [ "${GPU_ID}" != "1" ] && [ "${GPU_ID}" != "2" ]; then
	echo "GPU_ID: ${GPU_ID} is not a valid gpu_id, use [0, 1, 2]"
fi

MODEL_DESCRIPTION=cifar100_original_variance_reduction_dropout_0.5_num_samples_2_batch_size_128_iter_4_depth_28_widen_10 model=wide-resnet widen_factor=10 depth=28 dropout=0.5 num_samples=2 batchSize=128 num_iter=4 dataset=./datasets/cifar100_original.t7 gpuid=${GPU_ID} ./scripts/train_cifar.sh
