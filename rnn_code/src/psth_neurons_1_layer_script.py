
################################################
# Import statements
################################################

import cPickle
import os
from os import path
import sys
import time
import numpy
import theano
import theano.tensor as T
from loading_functions import load_all_data, load_class_data_batch, load_class_data_vt, load_neural_data
from layer_classes import LinearRegression, Dropout, LSTM, RNN, hybridRNN , IRNN, PoissonRegression
from one_ahead import GradClip, clip_gradient
from misc import Adam

################################################
# Script Parameters
################################################

#arguments: script_name, region, held_out_song


region_dict = {'L1':0,'L2':2,'L3':4,'NC':6,'MLd':8}
held_out_song = int(sys.argv[2])
brain_region = sys.argv[1]
brain_region_index = region_dict[brain_region]

n_epochs= 1
n_hidden = 240

print 'Running CV for held out song '+str(held_out_song)+' for brain region '+brain_region+' index at '+str(brain_region_index)

#Filepath for printing results
results_filename='/vega/stats/users/sl3368/rnn_code/results/neural/'+brain_region+'_'+str(held_out_song)+'.out'

#Directive and path for loading previous parameters
load_params_lstm = True
load_params_lstm_filename = '/vega/stats/users/sl3368/rnn_code/saves/params/lstm/1_layer/1000/zebra_1st_20_5000.save'

#check if exists already, then load or not load 
load_params_pr_filename = '/vega/stats/users/sl3368/rnn_code/saves/params/neural/'+brain_region+'_'+str(held_out_song)+'.save'
if path.isfile(load_params_pr_filename):
    print 'Will load previous regression parameters...'
    load_params_pr = True
else:
    load_params_pr = False
	

song_size = 2459

#filepath for saving parameters

psth_savefilename ='/vega/stats/users/sl3368/rnn_code/saves/params/psth/'+brain_region+'_'+str(held_out_song)+'.psth'

################################################
# Load Data
################################################
dataset_info = load_all_data()
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)

n_batches = data_set_x.shape[0].eval()/song_size

n_train_batches = n_batches 

print 'Number of songs in single matlab chunk: '+str(n_train_batches)

print 'Getting neural data...'

neural_data = load_neural_data()

ntrials = theano.shared(neural_data[brain_region_index],borrow=True)
responses = theano.shared(neural_data[brain_region_index+1],borrow=True)

######################
# BUILD ACTUAL MODEL #
######################

print 'building the model...'

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
x = clip_gradient(x,1.0)     
y = T.matrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
trial_no = T.matrix('trial_no')

is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

rng = numpy.random.RandomState(1234)

# Architecture: input --> LSTM --> predict one-ahead

lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)

output = PoissonRegression(input=lstm_1.output, n_in=n_hidden, n_out=responses.get_value(borrow=True).shape[1])
pred = output.E_y_given_x * trial_no
nll = output.negative_log_likelihood(y,trial_no)

################################
# Objective function and GD
################################

print 'defining cost, parameters, and learning function...'

# the cost we minimize during training is the negative log likelihood of
# the model 
cost = T.mean(nll)

#Defining params
params = output.params

# updates from ADAM
updates = Adam(cost, params)

#######################
# Objective function
#######################

print 'compiling train....'

#train_model = theano.function(inputs=[index], outputs=cost,
#        updates=updates,
#        givens={
#            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
#	    trial_no: ntrials[index * song_size:((index + 1) * song_size - 1)],
#            y: responses[index * song_size:((index + 1) * song_size - 1)]})

#test_model = theano.function(inputs=[index],
#        outputs=[cost],        givens={
#            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
#            y: data_set_x[(index * song_size + 1):(index + 1) * song_size]})
#
#

validate_model = theano.function(inputs=[index],
        outputs=[cost,nll,pred],
        givens={
            x: data_set_x[index * song_size:((index + 1) * song_size)],
	    trial_no: ntrials[index * song_size:((index + 1) * song_size)],
            y: responses[index * song_size:((index + 1) * song_size)]})

#######################
# Parameters and gradients
#######################
print 'parameters and gradients...'

if load_params_lstm:
    print 'loading parameters from file...'
    f = open( load_params_lstm_filename)
    old_p = cPickle.load(f)
    lstm_1.W_i.set_value(old_p[0].get_value(), borrow=True)
    lstm_1.W_f.set_value(old_p[1].get_value(), borrow=True)
    lstm_1.W_c.set_value(old_p[2].get_value(), borrow=True)
    lstm_1.W_o.set_value(old_p[3].get_value(), borrow=True)
    lstm_1.U_i.set_value(old_p[4].get_value(), borrow=True)
    lstm_1.U_f.set_value(old_p[5].get_value(), borrow=True)
    lstm_1.U_c.set_value(old_p[6].get_value(), borrow=True)
    lstm_1.U_o.set_value(old_p[7].get_value(), borrow=True)
    lstm_1.V_o.set_value(old_p[8].get_value(), borrow=True)
    lstm_1.b_i.set_value(old_p[9].get_value(), borrow=True)
    lstm_1.b_f.set_value(old_p[10].get_value(), borrow=True)
    lstm_1.b_c.set_value(old_p[11].get_value(), borrow=True)
    lstm_1.b_o.set_value(old_p[12].get_value(), borrow=True)
    f.close()

if load_params_pr:
    print 'loading parameters from file...'
    f = open( load_params_pr_filename)
    old_p = cPickle.load(f)
    output.W.set_value(old_p[0].get_value(), borrow=True)
    output.b.set_value(old_p[1].get_value(), borrow=True)
    f.close()

###############
# TRAIN MODEL #
###############
epoch = 0
print 'computing psth...'

while (epoch < n_epochs):

    epoch = epoch + 1
    
    if held_out_song<14:
        heldoutsong=held_out_song
    else:
        heldoutsong=held_out_song+10
    
    validation_info = validate_model(heldoutsong)

    f = file(psth_savefilename, 'wb')
    for obj in [validation_info[2]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

print '...Finished...'
