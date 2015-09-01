import os
import sys
import cPickle
import numpy

regions=['L1','L2','L3','NC','MLd']

neurons = []
for region in regions:
   sample = cPickle.load(open(region+'_0.save_neurons'))  
   region_mat = numpy.ones((20,sample.shape[0]),dtype=numpy.float32)
   for i in range(20):
	row = cPickle.load(open(region+'_'+str(i)+'.save_neurons'))
	region_mat[i]=row
   neurons.append(region_mat)

numpy.savez('LSTM_pr_results.npz',L1=neurons[0],L2=neurons[1],L3=neurons[2],NC=neurons[3],MLd=neurons[4])
