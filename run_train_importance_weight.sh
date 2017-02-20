if [ "${GPU_ID}" != "0" ] && [ "${GPU_ID}" != "1" ] && [ "${GPU_ID}" != "2" ]; then
	echo "GPU_ID: ${GPU_ID} is not a valid gpu_id, use [0, 1, 2]"
fi

MODEL_DESCRIPTION=cifar10_original_dropout_0.4_num_samples_8_batch_size_32_depth_28_widen_10 model=wide-resnet widen_factor=10 depth=28 dropout=0.4 importance_weighted_training=true num_samples=8 batchSize=32 num_iter=4 dataset=./datasets/cifar10_original.t7 gpuid=${GPU_ID} ./scripts/train_cifar.sh
