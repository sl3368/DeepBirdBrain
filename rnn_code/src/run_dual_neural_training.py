import sys
import os
import subprocess

#regions = ['L1','L2','L3','NC','MLd']
regions = ['L1','L2','L3','MLd']

for region in regions:
    for i in range(20):
	arguments = '-vREGION='+region+',HELDOUT='+str(i)
	subprocess.call(['qsub',arguments,'train_neural_dual_1_layer.sh'])
