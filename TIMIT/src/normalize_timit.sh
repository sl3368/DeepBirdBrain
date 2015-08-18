#!/bin/sh

# Directives
#PBS -N NormalizeTimitTest
#PBS -W group_list=yetistats
#PBS -l nodes=1:ppn=1:v1,walltime=60:03:30,mem=35000mb
#PBS -M sl3368@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/stats/users/sl3368/Data_LC/logs
#PBS -e localhost:/vega/stats/users/sl3368/Data_LC/logs

matlab -r process_timit
