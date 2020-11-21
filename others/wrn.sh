#!/bin/bash

ensemble_size=4
dataset=cifar10

# gr and depth
# full (4 8 12) (28 52 58) done (4 8) (28 52, 58)
wps=(12)
depths=(52 58)
# depths=(52)

for wp in ${wps[@]}
do
    for depth in ${depths[@]}
    do
    python wrn_exp_vary_param.py --dataset "${dataset}" --depth "${depth}" --width_param "${wp}" --ens_size "${ensemble_size}" --type horizontal --exp_suffix _e4_rand_augment
    python wrn_exp_vary_param.py --dataset "${dataset}" --depth "${depth}" --width_param "${wp}" --ens_size "${ensemble_size}" --type vertical --exp_suffix _e4_rand_augment
    done
clear

done