
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
from layer_classes import LinearRegression, Dropout, LSTM, LeNetConvPoolLayer, HiddenLayer
from one_ahead import clip_gradient
from misc import Adam

################################################
# Script Parameters
################################################

n_epochs= 350

n_hidden = 400

filter_number_1 = 10
filter_number_2 = 5

#Filepath for printing results
results_filename='/vega/stats/users/sl3368/CLDNN/results/deep_one_ahead_3rd.out'

#Directive and path for loading previous parameters
load_params = True
load_params_filename = '/vega/stats/users/sl3368/CLDNN/saves/deep_one_ahead_2nd.save'

song_size = 2459

#filepath for saving parameters
savefilename = '/vega/stats/users/sl3368/CLDNN/saves/deep_one_ahead_3rd.save'

################################################
# Load Data
################################################
dataset_info = load_all_data()
stim = dataset_info[0]
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
layer0_input = x.reshape((1, 1, song_size-1, 60))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(1, 1, song_size-1, 60),
    filter_shape=( filter_number_1, 1, 1, 3),
    poolsize=(1, 3),
    dim2 = 1
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(1, filter_number_1, song_size-1, 20),
    filter_shape=( filter_number_2, filter_number_1, 1, 2),
    poolsize=(1, 2),
    dim2 = 1
)

lstm_input = layer1.output.reshape((song_size-1,10 * filter_number_2))

#May be worth splitting to different LSTMs...would require smaller filter size
lstm_1 = LSTM(rng, lstm_input, n_in=10 * filter_number_2, n_out=n_hidden)

#output = LinearRegression(input=lstm_1.output, n_in=n_hidden, n_out=data_set_x.get_value(borrow=True).shape[1])
dnn = HiddenLayer(rng, lstm_1.output, n_in=n_hidden, n_out= 10 * filter_number_2)

dnn_output = dnn.output.reshape((1,filter_number_2,song_size-1,10))

layer1.reverseConv(dnn_output,(1,filter_number_2,song_size-1,20),(filter_number_1,filter_number_2,1,2)) #filter flipped on first two axes

layer0.reverseConv(layer1.reverseOutput,(1,filter_number_1,song_size-1,60),(1,filter_number_1,1,3)) #filter flipped on first two axes

reconstructed = layer0.reverseOutput.reshape((song_size-1,60))

################################
# Objective function and GD
################################

print 'defining cost, parameters, and learning function...'

# the cost we minimize during training is the negative log likelihood of
# the model 
cost = T.mean((y-reconstructed) **2)

#Defining params
params = lstm_1.params + layer0.params + layer1.params + dnn.params

# updates from ADAM
updates = Adam(cost, params)

#######################
# Objective function
#######################

print 'compiling train....'

train_model = theano.function(inputs=[index], outputs=cost,
        updates=updates,
        givens={
            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x[(index * song_size + 1):(index + 1) * song_size]})

test_model = theano.function(inputs=[index],outputs=[cost],        
	givens={
            x: data_set_x[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x[(index * song_size + 1):(index + 1) * song_size]})


validate_model = theano.function(inputs=[index], outputs=[layer1.reverseOutput.shape,layer1.reverseOutput],
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
    layer0.W.set_value(old_p[13].get_value(), borrow=True)
    layer0.b.set_value(old_p[14].get_value(), borrow=True)
    layer1.W.set_value(old_p[15].get_value(), borrow=True)
    layer1.b.set_value(old_p[16].get_value(), borrow=True)
    dnn.W.set_value(old_p[17].get_value(), borrow=True)
    dnn.b.set_value(old_p[18].get_value(), borrow=True)


###############
# TRAIN MODEL #
###############
print 'training...'

best_validation_loss = numpy.inf
epoch = 0

last_e = time.time()

r_log=open(results_filename,'w')
r_log.write('Starting training...\n')
r_log.close()

while (epoch < n_epochs):
    print str(epoch)+' epoch took: '+str(time.time()-last_e)
   
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()

    last_e = time.time()
    epoch = epoch + 1

    mb_costs = []

    for minibatch_index in xrange(14):
        minibatch_avg_cost = train_model(minibatch_index)
        print minibatch_avg_cost
	mb_costs.append(minibatch_avg_cost)

    for minibatch_index in xrange(24,30):
        minibatch_avg_cost = train_model(minibatch_index)
        print minibatch_avg_cost
	mb_costs.append(minibatch_avg_cost)

    # compute absolute error loss on validation set
#    validation_losses = [validate_model(i) for i in val_inds]
#    this_validation_loss = numpy.mean(validation_losses)
#    
#    print('epoch %i, minibatch %i, validation error %f' %  (epoch, minibatch_index + 1, this_validation_loss))
#    
#    r_log=open(results_filename, 'a')
#    r_log.write('epoch %i, minibatch %i, validation error %f\n' % (epoch, minibatch_index + 1, this_validation_loss))
#    r_log.close()

    avg_cost = numpy.mean(mb_costs)
    print 'Average training error: '+str(avg_cost)
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, training error %f\n' % (epoch, avg_cost))
    r_log.close()

    # if we got the best validation score until now
    if avg_cost < best_validation_loss:
	best_validation_loss = avg_cost
        #store data
        f = file(savefilename, 'wb')
        for obj in [params]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

print '...Finished...'
