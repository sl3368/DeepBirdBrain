#!/bin/sh

# Directives
#PBS -N train_mnist_deepnet_50
#PBS -W group_list=yetistats
#PBS -l walltime=72:00:00,mem=10gb,other=gpu
#PBS -q gpu
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/TIMIT/logs
#PBS -e localhost:/vega/stats/users/sl3368/TIMIT/logs

module load anaconda/2.7.8
module load cuda/6.5

python mnist_dual_encoder.py

#END OF SCRIPT
