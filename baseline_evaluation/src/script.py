


################################################
# Import statements
################################################

import cPickle
#import gzip
import os
import sys
import time

import numpy
#import h5py
#from pylab import *
#import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from loading_functions import load_all_data, load_class_data_batch, load_class_data_vt
from layer_classes import LinearRegression, Dropout, LSTM, RNN, hybridRNN , IRNN
from one_ahead import GradClip, clip_gradient


n_epochs=5

#dataset_info = load_all_data()
#data_set_x = dataset_info[0]
#maxBatchSize = numpy.int_(dataset_info[1])
#batch_size = maxBatchSize
#n_train_batches = 28
#n_valid_batches = 1
#n_test_batches = 1

datapathpre = '/vega/stats/users/sl3368/Data_LC/NormData/'
dataset_info = load_class_data_batch(datapathpre + 'LC_stim_1.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)

#validation and testing - for now, use last one
dataset_info_vt = load_class_data_vt(datapathpre + 'LC_stim_15.mat')
data_set_x_vt = dataset_info_vt[0]

batch_size = 3000
n_batches = data_set_x.shape[0].eval()/batch_size
print 'n_batches: '+str(n_batches)
n_val_batches = 10
n_test_batches = 10

n_train_batches = n_batches #data_set_x.shape[0].eval()/batch_size - n_val_batches - n_test_batches
print 'Number of batches for training: '+str(n_train_batches)
all_inds = numpy.arange(n_batches)
numpy.random.shuffle(all_inds);
train_inds = all_inds[0:n_train_batches]
val_inds = numpy.arange(n_val_batches)#all_inds[n_train_batches:n_train_batches+n_val_batches] #numpy.random.choice(n_batches,n_val_batches,replace=False)
test_inds = numpy.arange(n_val_batches)+n_val_batches


######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
x = clip_gradient(x,1.0)     
y = T.matrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector

is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

rng = numpy.random.RandomState(1234)

################################################
# Architecture: input --> LSTM --> predict one-ahead
################################################

# The poisson regression layer gets as input the hidden units
# of the hidden layer
#d_input = Dropout(rng, is_train, x)

#nn_lstm = 40
#nn_rnn = 60
#n_hidden = nn_lstm + nn_rnn

n_hidden = 200;

lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)
#lstm_1 = RNN(rng, d_input.output, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden) #vanilla rnn
#lstm_1 = hybridRNN(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_lstm=nn_lstm, n_rnn=nn_rnn) #each type must have at least 2
#lstm_2 = hybridRNN(rng, lstm_1.output, n_in=n_hidden, n_lstm=nn_lstm, n_rnn=nn_rnn) #each type must have at least 2

#lstm_1 = IRNN(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out = n_hidden) #each type must have at least 2
#lstm_2 = IRNN(rng, lstm_1.output, n_in=n_hidden, n_out=n_hidden) #each type must have at least 2
#d_lstm_1 = Dropout(rng, is_train, lstm_1.output)
output = LinearRegression(input=lstm_1.output, n_in=n_hidden, n_out=data_set_x.get_value(borrow=True).shape[1])

savefilename = '/vega/stats/users/sl3368/rnn_code/saves/params/lc_1_5_params_LSTM1_200_ada.save'


#######################
# Objective function
#######################
print '... defining objective and compiling test and validate'

# the cost we minimize during training is the negative log likelihood of
# the model 
cost = T.mean(output.negative_log_likelihood(y))

# compiling a Theano function that computes the mistakes that are made
# by the model on a minibatch
# use cost or errors(y,tc,md) as output?
test_model = theano.function(inputs=[index],
        outputs=[cost],#[cost, output.E_y_given_x],
        givens={
            x: data_set_x_vt[index * batch_size:((index + 1) * batch_size - 1)],
            y: data_set_x_vt[(index * batch_size + 1):(index + 1) * batch_size]})
            #is_train: numpy.cast['int32'](0)})

# wanted to use below indexes and have different sized batches, but this didn't work
#[int(batchBreaks[index]-1):int(batchBreaks[(index+1)]-1)]

validate_model = theano.function(inputs=[index],
        outputs=cost,
        givens={
            x: data_set_x_vt[index * batch_size:((index + 1) * batch_size - 1)],
            y: data_set_x_vt[(index * batch_size + 1):(index + 1) * batch_size]})
            #is_train: numpy.cast['int32'](0)})

#######################
# Parameters and gradients
#######################
print '... parameters and gradients'

# create a list (concatenated) of all model parameters to be fit by gradient descent
#order: [self.W, self.b]
#params = lstm_1.params + lstm_2.params + output.params
#params_helper = lstm_1.params_helper + lstm_2.params_helper + output.params_helper
#params_helper2 = lstm_1.params_helper2 + lstm_2.params_helper2 + output.params_helper2

params = lstm_1.params + output.params
params_helper = lstm_1.params_helper + output.params_helper
params_helper2 = lstm_1.params_helper2 + output.params_helper2

for p in params:
    print p.shape.eval()

# compute the gradient of cost with respect to theta (sotred in params)
# the resulting gradients will be stored in a list gparams
gparams = []
for param in params:
    gparam = T.grad(cost, param)
    gparams.append(gparam)

# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs
updates = []
# given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
# same length, zip generates a list C of same size, where each element
# is a pair formed from the two lists :
#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
#for param, gparam in zip(params, gparams):
#    updates.append((param, param - learning_rate * gparam))
#iter_count = theano.shared(1)
#L1_penalized = []
#larger_stepsize = []
#enforce_positive = [2, 3] #if recurrent
#enforce_positive = []
#zero_stepsize = []
param_index = 0
#rho = 1e-6
#for param, param_helper, param_helper2, gparam in zip(params, params_helper, params_helper2, gparams):
    #updates.append((param_helper, param_helper + gparam ** 2)) #need sum of squares for learning rate
    #updates.append((param_helper2, param_helper2 + gparam)) #need sum of gradients for L1 thresholding

#vanilla SGD
#learning_rate=1e-4
#for param, gparam in zip(params, gparams):
#    updates.append((param, param - learning_rate * gparam))
#    param_index += 1

#adadelta updates
rho = .95
eps_big = 1e-6
for param, param_helper, param_helper2, gparam in zip(params, params_helper, params_helper2, gparams):
    updates.append((param_helper,rho * param_helper + (1. - rho) * (gparam ** 2))) #update decaying sum of previous gradients
    dparam = - T.sqrt((param_helper2 + eps_big) / (rho * param_helper + (1. - rho) * (gparam ** 2) + eps_big)) *gparam # calculate step size
    updates.append((param_helper2, rho * param_helper2 + (1. - rho) * (dparam ** 2))) #update decaying sum of previous step sizes
    updates.append((param, param + dparam))

#updates.append((iter_count, iter_count + 1))

print '... compiling train'
# compiling a Theano function `train_model` that returns the cost, but
# in the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(inputs=[index], outputs=cost,
        updates=updates,
        givens={
            x: data_set_x[index * batch_size:((index + 1) * batch_size - 1)],
            y: data_set_x[(index * batch_size + 1):(index + 1) * batch_size]})
            #is_train: numpy.cast['int32'](0)})

###############
# TRAIN MODEL #
###############
print '... training'

# early-stopping parameters
patience = 5000  # look as this many examples regardless
add_patience = 5000
#patience = train_set_x.get_value(borrow=True).shape[0] * n_epochs #no early stopping
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.99  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

print 'Validation frequency: '+str(validation_frequency)

#best_params = None
best_validation_loss = numpy.inf
best_iter = 0
#test_score = 0.
start_time = time.clock()

epoch = 0
reset_epoch=0
done_looping = False
loop_vec = numpy.arange(n_train_batches)
data_part = 1
#dataparts = numpy.r_[range(1,28),29,range(30,61),range(73,89)] # [1:9 10:19 20:27 29  30:39 40:49 50:59 60 73:79 80:88]
dataparts = numpy.r_[range(1,6)]

track_train = list()
track_valid = list()
track_test = list()
last_m = time.time()
last_e = time.time()

results_filename='/vega/stats/users/sl3368/rnn_code/results/200_ada_rnn_1_5_batches.out'
r_log=open(results_filename,'w')
r_log.write('Starting training...\n')
r_log.close()

while (epoch < n_epochs) and (not done_looping):
    print 'Last epoch took: '+str(time.time()-last_e)
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()
    last_e = time.time()
    epoch = epoch + 1
    reset_epoch = reset_epoch + 1
    numpy.random.shuffle(loop_vec)

    if reset_epoch == 1:
        dataset_info = load_class_data_batch(datapathpre + 'LC_stim_%i.mat'%dataparts[data_part])
        print dataparts[data_part]
        stim = dataset_info[0]
        print stim.shape
        data_set_x.set_value(stim,borrow=True)
        reset_epoch = 0
        data_part = data_part+1
        if data_part > (len(dataparts)-1):
            data_part = 0

    for minibatch_index in xrange(n_train_batches):
        #print 'Last minibatch took: '+str(time.time()-last_m)
        last_m = time.time()
        mini_iter = loop_vec[minibatch_index]
        minibatch_avg_cost = train_model(mini_iter)
        #print minibatch_avg_cost
        track_train.append(minibatch_avg_cost)

        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute absolute error loss on validation set
            validation_losses = [validate_model(i) for i
                                 in val_inds]
            this_validation_loss = numpy.mean(validation_losses) #mean over batches
            print('epoch %i, minibatch %i, validation error %f' %
                 (epoch, minibatch_index + 1,
                  this_validation_loss))
            track_valid.append(this_validation_loss)
            r_log=open(results_filename, 'a')
            r_log.write('epoch %i, minibatch %i, validation error %f\n' %
                 (epoch, minibatch_index + 1,
                  this_validation_loss))
            r_log.close()


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                    patience = max(patience, iter + add_patience)
                    #patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i) for i
                               in test_inds]
                test_cost = numpy.mean(test_losses)
                #test_score = numpy.mean(test_losses)
                #test_cost, test_pred = test_model(29)
                #test_cost, test_costs_separate, test_pred_separate, test_actual_separate = test_model(29)

                print(('     epoch %i, minibatch %i, test error of '
                       'best model %f') %
                      (epoch, minibatch_index + 1,
                       numpy.mean(test_cost)))
                track_test.append(numpy.mean(test_cost))

        if (iter+1)%add_patience==0:
            #store data
            f = file(savefilename, 'wb')
            for obj in [params + [track_train] + [track_valid] + [track_test]]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        if patience <= iter:
                done_looping = True
                break

end_time = time.clock()
print(('Optimization complete. Best validation score of %f'
       'obtained at iteration %i, with test performance %f') %
      (best_validation_loss, best_iter + 1, numpy.sum(test_cost)))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

#store data
f = file(savefilename, 'wb')
for obj in [params + [track_train] + [track_valid] + [track_test]]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

