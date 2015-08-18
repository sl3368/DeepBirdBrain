
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
from layer_classes import LinearRegression, Dropout, LSTM, RNN, hybridRNN , IRNN
from one_ahead import GradClip, clip_gradient
from misc import Adam

################################################
# Script Parameters
################################################

datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'
n_epochs=4

#Filepath for printing results
results_filename='/vega/stats/users/sl3368/rnn_code/results/lstm/2_layer/400/3rd_8_12.out'

#Directive and filepath for loading files
load_params = True
load_params_filename = '/vega/stats/users/sl3368/rnn_code/saves/params/lstm/2_layer/400/recent_2nd.save' 

#sizes
minibatch_size = 1
song_size = 2000

#Validation and Testing sizes
n_val_batches = 100
n_test_batches = 10

#Savefile path
savefilename = '/vega/stats/users/sl3368/rnn_code/saves/params/lstm/2_layer/400/3rd_8_12.save'

################################################
# Load Data
################################################
dataset_info = load_class_data_batch(datapathpre + 'LC_stim_8.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)

#validation and testing - for now, use last one
dataset_info_vt = load_class_data_vt(datapathpre + 'LC_stim_15.mat')
data_set_x_vt = dataset_info_vt[0]

n_batches = data_set_x.shape[0].eval()/song_size

n_train_batches = n_batches 
print 'Number of songs in single matlab chunk: '+str(n_train_batches)
all_inds = numpy.arange(n_batches)
numpy.random.shuffle(all_inds);
train_inds = all_inds[0:n_train_batches]
val_inds = numpy.arange(n_val_batches)
test_inds = numpy.arange(n_val_batches)+n_val_batches


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

n_hidden = 400;

lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)

lstm_2 = LSTM(rng, lstm_1.output, n_in=n_hidden, n_out=n_hidden-200)

output = LinearRegression(input=lstm_2.output, n_in=n_hidden-200, n_out=data_set_x.get_value(borrow=True).shape[1])


################################
# Objective function and GD
################################

print 'defining cost, parameters, and learning function...'

# the cost we minimize during training is the negative log likelihood of
# the model 
cost = T.mean(output.negative_log_likelihood(y))

#Defining params
params = lstm_1.params + lstm_2.params + output.params

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
            x: data_set_x_vt[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x_vt[(index * song_size + 1):(index + 1) * song_size]})


validate_model = theano.function(inputs=[index],
        outputs=cost,
        givens={
            x: data_set_x_vt[index * song_size:((index + 1) * song_size - 1)],
            y: data_set_x_vt[(index * song_size + 1):(index + 1) * song_size]})

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
    lstm_2.W_i.set_value(old_p[13].get_value(), borrow=True)
    lstm_2.W_f.set_value(old_p[14].get_value(), borrow=True)
    lstm_2.W_c.set_value(old_p[15].get_value(), borrow=True)
    lstm_2.W_o.set_value(old_p[16].get_value(), borrow=True)
    lstm_2.U_i.set_value(old_p[17].get_value(), borrow=True)
    lstm_2.U_f.set_value(old_p[18].get_value(), borrow=True)
    lstm_2.U_c.set_value(old_p[19].get_value(), borrow=True)
    lstm_2.U_o.set_value(old_p[20].get_value(), borrow=True)
    lstm_2.V_o.set_value(old_p[21].get_value(), borrow=True)
    lstm_2.b_i.set_value(old_p[22].get_value(), borrow=True)
    lstm_2.b_f.set_value(old_p[23].get_value(), borrow=True)
    lstm_2.b_c.set_value(old_p[24].get_value(), borrow=True)
    lstm_2.b_o.set_value(old_p[25].get_value(), borrow=True)
    output.W.set_value(old_p[26].get_value(), borrow=True)
    output.b.set_value(old_p[27].get_value(), borrow=True)


###############
# TRAIN MODEL #
###############
print 'training...'

# early-stopping parameters

# look as this many examples regardless 
patience = 5000
add_patience = 5000

# wait this much longer when a new best is
# found
patience_increase = 2  
                       
# a relative improvement of this much is
# considered significant
improvement_threshold = 0.99


# go through this many
# minibatche before checking the network
# on the validation set; in this case we
# check every epoch
validation_frequency = min(n_train_batches, patience / 2)

#print 'Validation frequency: '+str(validation_frequency)

best_validation_loss = numpy.inf
best_iter = 0
start_time = time.clock()

epoch = 0
reset_epoch=0
done_looping = False
loop_vec = numpy.arange(n_train_batches)

data_part = 7
dataparts = numpy.r_[range(1,13)]

track_train = list()
track_valid = list()
track_test = list()
last_m = time.time()
last_e = time.time()

r_log=open(results_filename,'w')
r_log.write('Starting training...\n')
r_log.close()

#while (epoch < n_epochs) and (not done_looping):
while (epoch < n_epochs):
    print str(epoch)+' epoch took: '+str(time.time()-last_e)
   
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()
    last_e = time.time()
    epoch = epoch + 1
    numpy.random.shuffle(loop_vec)

    if reset_epoch == 1:
        dataset_info = load_class_data_batch(datapathpre + 'LC_stim_%i.mat'%dataparts[data_part])
        stim = dataset_info[0]
        data_set_x.set_value(stim,borrow=True)
        reset_epoch = 0
        data_part = data_part+1
        if data_part > (len(dataparts)-1):
            data_part = 0

    reset_epoch = reset_epoch + 1

    for minibatch_index in xrange(n_train_batches):
        mini_iter = loop_vec[minibatch_index]
        minibatch_avg_cost = train_model(mini_iter)
        #print minibatch_avg_cost
        track_train.append(minibatch_avg_cost)

    # compute absolute error loss on validation set
    validation_losses = [validate_model(i) for i in val_inds]
    this_validation_loss = numpy.mean(validation_losses)
    
    print('epoch %i, minibatch %i, validation error %f' %  (epoch, minibatch_index + 1, this_validation_loss))
    track_valid.append(this_validation_loss)
    
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, minibatch %i, validation error %f\n' % (epoch, minibatch_index + 1, this_validation_loss))
    r_log.close()


    # if we got the best validation score until now
    if this_validation_loss < best_validation_loss:
        
	# test it on the test set
        #test_losses = [test_model(i) for i in test_inds]
        #test_cost = numpy.mean(test_losses)

        #print(('     epoch %i, minibatch %i, test error of best model %f') % (epoch, minibatch_index + 1, numpy.mean(test_cost)))
        #track_test.append(numpy.mean(test_cost))

        #store data
        f = file(savefilename, 'wb')
        for obj in [params + [track_train] + [track_valid] + [track_test]]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


print 'Savind and ending...'

#store data
f = file(savefilename, 'wb')
for obj in [params]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

