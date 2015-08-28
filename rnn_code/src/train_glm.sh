#!/bin/sh

# Directives
#PBS -N GLM
#PBS -W group_list=yetistats
#PBS -l walltime=4:00:00,mem=6gb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/rnn_code/logs/glm/
#PBS -e localhost:/vega/stats/users/sl3368/rnn_code/logs/glm/

module load anaconda/2.7.8
module load cuda/6.5

python glm_script.py $REGION $HELDOUT


#END OF SCRIPT
