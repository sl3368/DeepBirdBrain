#!/bin/sh

# Directives
#PBS -N train_long_convae_5layer_nopad_20_1st
#PBS -W group_list=yetistats
#PBS -l walltime=70:00:00,mem=10gb,other=gpu
#PBS -q gpu
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/encoder/logs
#PBS -e localhost:/vega/stats/users/sl3368/encoder/logs

module load anaconda/2.7.8
module load cuda/6.5

python long_encoder.py

#END OF SCRIPT
