import sys
import os
import subprocess

#parameters are: n_hidden, song_size, signal length, trials
n_hidden = 100
song_size = 25

for signal_length in range(200,2001,100):
    for i in range(10):
	arguments = '-vHIDDEN='+str(n_hidden)+',SONG='+str(song_size)+',SIGNAL='+str(signal_length)+',TRIAL='+str(i)
	subprocess.call(['qsub',arguments,'train_lstm_test.sh'])
