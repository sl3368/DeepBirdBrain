#!/bin/sh

# Directives
#PBS -N 2_layer_analysis
#PBS -W group_list=yetistats
#PBS -l walltime=20:00:00,mem=10gb,other=gpu
#PBS -q gpu
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/2_layer/400/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/2_layer/400/

module load anaconda/2.7.8
module load cuda/6.5

python 2_layer_analysis.py


#END OF SCRIPT
