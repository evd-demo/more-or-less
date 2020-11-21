#!/bin/bash

ensemble_size=4
dataset=cifar10
dir=vary_both_e_"${ensemble_size}"_"${dataset}"_testing_equalization

# gr and depth
grs=(12)
depths=(40)

for gr in ${grs[@]}
do
    for depth in ${depths[@]}
    do
    subdir="${dir}"/gr_"${gr}"_d_"${depth}"_e_"${ensemble_size}"_"${dataset}"
    python3 main.py --exp_dir="${subdir}" --run_ensemble=True --depth="${depth}" --growth_rate="${gr}" --ensemble_size="${ensemble_size}" --n_epochs=120 --batch_size=128 --vary_growth_rate=True --vary_depth=True --dataset="${dataset}"
    done

done