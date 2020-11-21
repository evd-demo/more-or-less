#!/bin/bash

ensemble_size=4
dataset=svhn
dir=vary_both_e_"${ensemble_size}"_"${dataset}"

# gr and depth
grs=(12 20 28 36 48)
depths=(40 64 88 112)

for gr in ${grs[@]}
do
    for depth in ${depths[@]}
    do
    subdir="${dir}"/gr_"${gr}"_d_"${depth}"_e_"${ensemble_size}"_"${dataset}"
    python3 main.py --exp_dir="${subdir}" --run_ensemble=True --depth="${depth}" --growth_rate="${gr}" --ensemble_size="${ensemble_size}" --n_epochs=100 --batch_size=64 --vary_growth_rate=True --vary_depth=True --efficient=True --dataset="${dataset}"
    done

done