#!/bin/sh

# Directives
#PBS -N train_convae_5layer_nopad_10_3rd
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

python encoder.py

#END OF SCRIPT
