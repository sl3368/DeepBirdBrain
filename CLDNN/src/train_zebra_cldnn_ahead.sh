#!/bin/sh

# Directives
#PBS -N cldnn_zebra_ahead_6
#PBS -W group_list=yetistats
#PBS -l walltime=72:00:00,mem=10gb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/CLDNN/logs/
#PBS -e localhost:/vega/stats/users/sl3368/CLDNN/logs/

module load anaconda/2.7.8
module load cuda/6.5

python zebra_cldnn_ahead.py


#END OF SCRIPT
