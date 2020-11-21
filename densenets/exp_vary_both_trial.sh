#!/bin/bash

ensemble_size=4
dataset=cifar10
num_trial=2
dir=vary_both_e_"${ensemble_size}"_"${dataset}"_trials_2

# gr and depth
grs=(12 20 28 36 48)
depths=(40 64 88 112)

# For now, use a subset of parameters to iterate quickly
# grs=(12 20)
# depths=(40 64)


for gr in ${grs[@]}
do
    for depth in ${depths[@]}
    do
    subdir="${dir}"/gr_"${gr}"_d_"${depth}"_e_"${ensemble_size}"_"${dataset}"
    python run.py --command \""python3 main.py --exp_dir="${subdir}" --run_ensemble=True --depth="${depth}" --growth_rate="${gr}" --ensemble_size="${ensemble_size}" --num_trial="${num_trial}" --n_epochs=120 --batch_size=128 --vary_growth_rate=True --vary_depth=True --dataset="${dataset}""\" --job_name gr_"${gr}"_d_"${depth}"_e_"${ensemble_size}"_"${dataset}" --time 48
    done
clear

done