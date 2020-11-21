#!/bin/bash

ensemble_size=4
dataset=imagenet32
dir=vary_both_e_"${ensemble_size}"_"${dataset}"

# gr and depth for imagenet
grs=(28 36)
# depths=(40 64 88) # need to run for depth 64
depths=(64 88) # need to run for depth 64


for gr in ${grs[@]}
do
    for depth in ${depths[@]}
    do
    subdir="${dir}"/gr_"${gr}"_d_"${depth}"_e_"${ensemble_size}"_"${dataset}"
    python3 main.py --exp_dir="${subdir}" --run_ensemble=True --depth="${depth}" --growth_rate="${gr}" --ensemble_size="${ensemble_size}" --n_epochs=90 --batch_size=128 --vary_growth_rate=True --vary_depth=False --efficient=False --dataset="${dataset}"
    done

done