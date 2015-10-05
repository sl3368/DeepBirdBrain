#!/bin/sh

# Directives
#PBS -N 1_layer_zebra_400_1000_4th
#PBS -W group_list=yetistats
#PBS -l walltime=72:00:00,mem=10gb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/1_layer/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/1_layer/

module load anaconda/2.7.8
module load cuda/6.5

python zebra_1_layer_script.py


#END OF SCRIPT
