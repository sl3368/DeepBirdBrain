#!/bin/sh

# Directives
#PBS -N LSTM_VIS_GEN_INCREMENTAL
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

python visualization_tools.py $REGION $HELDOUT $NEURON $ITERATIONS $HIDDEN $STEPSIZE $SONGSIZE


#END OF SCRIPT
