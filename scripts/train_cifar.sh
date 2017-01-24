#!/usr/bin/env bash

if [ "${MODEL_DESCRIPTION}" == "" ]; then
	echo "Error: Provide brief MODEL_DESCRIPTION"
	exit 1
fi

export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${MODEL_DESCRIPTION}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt
