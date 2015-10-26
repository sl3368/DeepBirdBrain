import sys
import os
import subprocess

#regions = ['L1','L2','L3','NC','MLd']
regions = ['L1']
neurons = {'L1':219,'L2':377,'L3':237,'NC':273,'MLd':107}

for region in regions:
    for i in range(20):
        for neuron in range(neurons[region]):
	    arguments = '-vREGION='+region+',HELDOUT='+str(i)+',NEURON='+str(neuron)
	    subprocess.call(['qsub',arguments,'train_neural_dual_single.sh'])
