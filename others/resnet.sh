#!/bin/bash

ensemble_size=4
dataset=cifar100

# gr and depth
# wps=(16 24 32 48)
# depths=(26 38 50 62)
# left: 48 x (38 50 62) and (16 24 32) x 26

wps=(48)
depths=(38 50 62)


for wp in ${wps[@]}
do
    for depth in ${depths[@]}
    do
    python3.8 resnet_exp_vary_param.py --dataset "${dataset}" --depth "${depth}" --width_param "${wp}" --ens_size "${ensemble_size}" --type horizontal --exp_suffix _e4_rand_augment
    python3.8 resnet_exp_vary_param.py --dataset "${dataset}" --depth "${depth}" --width_param "${wp}" --ens_size "${ensemble_size}" --type vertical --exp_suffix _e4_rand_augment
    done
clear

done