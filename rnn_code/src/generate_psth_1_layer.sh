#!/bin/sh

# Directives
#PBS -N PSTH_generate
#PBS -W group_list=yetistats
#PBS -l walltime=2:00:00,mem=5gb
#PBS -M sl3368@columbia.edu
#PBS -m a
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/

module load anaconda/2.7.8
module load cuda/6.5

python psth_neurons_1_layer_script.py $REGION $HELDOUT


#END OF SCRIPT
