
################################################
# Import statements
################################################

import cPickle
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from loading_functions import load_all_data, load_class_data_batch, load_class_data_vt
from layer_classes import LinearRegression, Dropout, LSTM, LogisticRegression
from one_ahead import GradClip, clip_gradient
from misc import Adam
from sklearn.preprocessing import OneHotEncoder
import numpy.random

################################################
# Script Parameters
################################################

#ADD COMMAND LINE ARGUMENTS AS PARAMETERS
# script name, n_hidden, n_numbers, signal_size, trial

n_epochs=5000

n_hidden = int(sys.argv[1])
song_size = int(sys.argv[2])
signal_size = int(sys.argv[3])
trial = int(sys.argv[4])

#Filepath for printing results
results_filename='/vega/stats/users/sl3368/LSTM/results/lstm_'+str(n_hidden)+'_'+str(song_size)+'_'+str(signal_size)+'_'+str(trial)+'.out'

#Directive and path for loading previous parameters
load_params = False
load_params_filename = '/vega/stats/users/sl3368/rnn_code/saves/params/lstm/1_layer/1000/zebra_4th_1_500.save'

#filepath for saving parameters
savefilename = '/vega/stats/users/sl3368/LSTM/saves/tests/lstm_'+str(n_hidden)+'_'+str(song_size)+'_'+str(signal_size)+'_'+str(trial)+'.save'

################################################
# Load Data
################################################
#dataset_info = load_all_data()
#stim = dataset_info[0]

encodings = numpy.random.randint(song_size,size=signal_size)
stim = numpy.reshape(encodings,(signal_size,1))
enc = OneHotEncoder(n_values=song_size,dtype=numpy.float32,sparse=False)
stim = enc.fit_transform(stim)
data_set_x = theano.shared(stim, borrow=True)

n_batches = data_set_x.shape[0].eval()/song_size

n_train_batches = n_batches 
print 'Number of songs in single matlab chunk: '+str(n_train_batches)

######################
# BUILD ACTUAL MODEL #
######################

print 'building the model...'

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
x = clip_gradient(x,1.0)     
y = T.matrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector

is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

rng = numpy.random.RandomState(1234)

# Architecture: input --> LSTM --> predict one-ahead

lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)

output = LogisticRegression(input=lstm_1.output, n_in=n_hidden, n_out=data_set_x.get_value(borrow=True).shape[1])


################################
# Objective function and GD
################################

print 'defining cost, parameters, and learning function...'

# the cost we minimize during training is the negative log likelihood of
# the model 
cost = T.mean(output.cross_entropy_binary(y))

#Defining params
params = lstm_1.params + output.params

# updates from ADAM
updates = Adam(cost, params)

#######################
# Objective function
#######################

print 'compiling train....'

train_model = theano.function(inputs=[index], outputs=cost,
        updates=updates,
        givens={
            x: data_set_x[index * signal_size:((index + 1) * signal_size - 1)],
            y: data_set_x[(index * signal_size + 1):(index + 1) * signal_size]})

test_model = theano.function(inputs=[index],
        outputs=[cost],        givens={
            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x[(index * song_size + 1):(index + 1) * song_size]})


validate_model = theano.function(inputs=[index],
        outputs=cost,
        givens={
            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x[(index * song_size + 1):(index + 1) * song_size]})

#######################
# Parameters and gradients
#######################
print 'parameters and gradients...'

if load_params:
    print 'loading parameters from file...'
    f = open( load_params_filename)
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
    output.W.set_value(old_p[13].get_value(), borrow=True)
    output.b.set_value(old_p[14].get_value(), borrow=True)


###############
# TRAIN MODEL #
###############
print 'training...'

best_validation_loss = numpy.inf
epoch = 0

last_e = time.time()

#r_log=open(results_filename,'w')
#r_log.write('Starting training...\n')
#r_log.close()

while (epoch < n_epochs):
    print str(epoch)+' epoch took: '+str(time.time()-last_e)
   
#    r_log=open(results_filename, 'a')
#    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
#    r_log.close()

    last_e = time.time()
    epoch = epoch + 1

    for minibatch_index in xrange(1):
        minibatch_avg_cost = train_model(minibatch_index)
        print minibatch_avg_cost

    # compute absolute error loss on validation set
#    validation_losses = [validate_model(i) for i in val_inds]
#    this_validation_loss = numpy.mean(validation_losses)
#    
#    print('epoch %i, minibatch %i, validation error %f' %  (epoch, minibatch_index + 1, this_validation_loss))
#    
#    r_log=open(results_filename, 'a')
#    r_log.write('epoch %i, minibatch %i, validation error %f\n' % (epoch, minibatch_index + 1, this_validation_loss))
#    r_log.close()


    # if we got the best validation score until now
    if minibatch_avg_cost < best_validation_loss:
	best_validation_loss = minibatch_avg_cost        
        #store data
        f = file(savefilename, 'wb')
        for obj in [best_validation_loss]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


end_time = time.clock()

print '...Finished...'
