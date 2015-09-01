#!/bin/sh

# Directives
#PBS -N LSTM_Testing
#PBS -W group_list=yetistats
#PBS -l walltime=4:00:00,mem=4gb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/LSTM/logs/
#PBS -e localhost:/vega/stats/users/sl3368/LSTM/logs/

module load anaconda/2.7.8
module load cuda/6.5

python encoding_test.py $HIDDEN $SONG $SIGNAL $TRIAL


#END OF SCRIPT
