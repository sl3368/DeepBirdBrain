import os
from os import path
import sys
import cPickle
import numpy

results = []
for signal_length in range(20,200):
    for trial_no in range(20):
	lstm_pr = 'lstm_100_25_'+str(signal_length)+'_'+str(trial_no)+'.save'	
        if path.isfile(lstm_pr):	
	   f = open(lstm_pr,'rb')
	   try:
		r = cPickle.load(f)
	   except EOFError:
		print lstm_pr
           f.close()
	   results.append([r,signal_length])

all_trials = numpy.zeros((len(results),2),dtype=numpy.float32)
for i in range(len(results)):
    all_trials[i][0]=results[i][1]
    all_trials[i][1]=results[i][0]

numpy.savez('LSTM_test_results.npz',results=all_trials) 
