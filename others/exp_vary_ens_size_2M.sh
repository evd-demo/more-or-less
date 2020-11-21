#!/bin/bash
# 0.5 M to 5 M
python exp_vary_param.py --num_param 2000000 --ens_size 4
python exp_vary_param.py --num_param 2000000 --ens_size 6
python exp_vary_param.py --num_param 2000000 --ens_size 8
python exp_vary_param.py --num_param 2000000 --ens_size 12
python exp_vary_param.py --num_param 2000000 --ens_size 16