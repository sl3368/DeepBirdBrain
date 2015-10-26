#!/bin/sh

# Directives
#PBS -N LSTM_PR_single
#PBS -W group_list=yetistats
#PBS -l walltime=1:45:00,mem=7gb
#PBS -M sl3368@columbia.edu
#PBS -m a
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/

module load anaconda/2.7.8
module load cuda/6.5

python neural_dual_single_neuron_script.py $REGION $HELDOUT $NEURON


#END OF SCRIPT
