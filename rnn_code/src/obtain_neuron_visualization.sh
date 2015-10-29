#!/bin/sh

# Directives
#PBS -N LSTM_VISUALIZE_NEURON
#PBS -W group_list=yetistats
#PBS -l walltime=12:00:00,mem=7gb
#PBS -M sl3368@columbia.edu
#PBS -m a
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/neural/

module load anaconda/2.7.8
module load cuda/6.5

python visualize_receptive_neuron.py $REGION $HELDOUT $NEURON


#END OF SCRIPT
