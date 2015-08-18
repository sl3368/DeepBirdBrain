#!/bin/sh

# Directives
#PBS -N 1_layer_gen_song_zebra
#PBS -W group_list=yetistats
#PBS -l walltime=20:00:00,mem=10gb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/1_layer/1000/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/lstm/1_layer/1000/

module load anaconda/2.7.8
module load cuda/6.5

python zebra_1_layer_ahead_analysis.py


#END OF SCRIPT
