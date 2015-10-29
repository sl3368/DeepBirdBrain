import sys
import os
import subprocess

#regions = ['L1','L2','L3','NC','MLd']
#regions = ['L1']
#neurons = {'L1':219,'L2':377,'L3':237,'NC':273,'MLd':107}


#for region in regions:
#    for i in range(20):
#        for neuron in range(neurons[region]):
#	    arguments = '-vREGION='+region+',HELDOUT='+str(i)+',NEURON='+str(neuron)
#	    subprocess.call(['qsub',arguments,'obtain_neuron_visualization.sh'])

region = ['L1','L1','L1','L1','L1','L1']
heldout = [2,6,7,9,11,14]
neuron = [135,135,135,135,135,135]

for i in range(len(region)):
    arguments = '-vREGION='+region[i]+',HELDOUT='+str(heldout[i])+',NEURON='+str(neuron[i])
    subprocess.call(['qsub',arguments,'obtain_neuron_visualization.sh'])
