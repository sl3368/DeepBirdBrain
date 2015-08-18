import sys
import time
import numpy 
import theano
from theano import shared
import cPickle, gzip
from theano import tensor as T
from layer_classes import LeNetConvPoolLayer, LogisticRegression, LinearRegression, LSTM
from misc import Adam
from loading_functions import load_class_data_batch, load_class_data_vt, shared_dataset
from load_timit import get_timit_waveform

### Tunable parameters and variables ###
########################################

n_epochs=5

#Number hidden units per layer
lstm_1_hidden = 50
lstm_2_hidden = 100

minibatch_size = 60
song_size = 2000 #dependent on padding (regular,low,none)

savefilename = '/vega/stats/users/sl3368/TIMIT/saves/params/timit_lstm_wav_5.save'
results_filename='/vega/stats/users/sl3368/TIMIT/results/timit_lstm_wav_5.out'
datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'

train_x_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_waveform_train_x.data'
train_y_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_waveform_train_y.npz'
test_x_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_waveform_test_x.data'
test_y_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_waveform_test_y.npz'

#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

print 'Loading TIMIT Data..'
#start = time.time()
#
## Load the dataset
#train_x, train_y, test_x, test_y = get_timit_waveform()
#print 'Finished Loading, took: '+str(start-time.time())
#print 'Saving...'
#
#start = time.time()
#f = file(train_x_filename, 'wb')
#for obj in [train_x]:
#    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#print 'Saved train x, took: '+str(start-time.time())
#
#start = time.time()
##f = file(train_y_filename, 'wb')
##for obj in [train_y]:
##    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
#numpy.savez_compressed(train_y_filename,train_y=train_y)
##f.close()
#print 'Saved train y, took: '+str(start-time.time())
#
#start = time.time()
#f = file(test_x_filename, 'wb')
#for obj in [test_x]:
#    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#print 'Saved test x, took: '+str(start-time.time())
#
#start = time.time()
##f = file(test_y_filename, 'wb')
##for obj in [test_y]:
##    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
##f.close()
#numpy.savez_compressed(test_y_filename,test_y=test_y)
#print 'Saved test y, took: '+str(start-time.time())

start = time.time()
f = file(train_x_filename, 'rb')
train_x = cPickle.load(f)
f.close()
print 'Loaded train x...took: '+str(time.time()-start)

start = time.time()
#f = file(train_y_filename, 'rb')
#train_y = cPickle.load(f)
#f.close()
train_y = numpy.load(train_y_filename)['train_y']
print 'Loaded train y...'+str(time.time()-start)

start = time.time()
f = file(test_x_filename, 'rb')
test_x = cPickle.load(f)
f.close()
print 'Loaded test x...'+str(time.time()-start)

start = time.time()
#f = file(test_y_filename, 'rb')
#test_y = cPickle.load(f)
#f.close()
test_y = numpy.load(test_y_filename)['test_y']
print 'Loaded test y...'+str(time.time()-start)

print 'Loading complete...'

assert len(train_x) == len(train_y)
print 'Number of training sentences: '+str(len(train_x))

assert len(test_x) == len(test_y) 
print 'Number of testing sentences: '+str(len(test_x))

#remember clipping individual songs

###########################################################
###########################################################

############ CONSTRUCTING MODEL ARCHITECTURE ##############
###########################################################


print 'Building model...'

# allocate symbolic variables for the data

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
y = T.imatrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
ahead = T.matrix('ahead')	
sent = T.matrix('sentence')
phonemes = T.imatrix('phonemes')

rng = numpy.random.RandomState(1234)

init_reg = LinearRegression(x, 1, 30,True)

lstm_1 = LSTM(rng,init_reg.E_y_given_x,30,lstm_1_hidden)

lstm_2 = LSTM(rng,lstm_1.output,lstm_1_hidden,lstm_2_hidden)

reg_input = lstm_2.output

#need log_reg and cross covariate layers
log_reg = LogisticRegression(reg_input,lstm_2_hidden, 41)

#lin_reg = LinearRegression(reg_input,lstm_2_hidden,1,True)

log_reg.reconstruct(log_reg.p_y_given_x)
#lin_reg.reconstruct(lin_reg.E_y_given_x)

#reconstructed_regressions = T.concatenate([log_reg.reconstructed_x,lin_reg.reconstructed_x],axis=1)
#
#reverse_layer = LinearRegression(reconstructed_regressions, 2*lstm_2_hidden, lstm_2_hidden,False)

lstm_3 = LSTM(rng,log_reg.reconstructed_x,lstm_2_hidden,lstm_1_hidden)

lstm_4 = LSTM(rng,lstm_3.output,lstm_1_hidden,30)

init_reg.reconstruct(lstm_4.output)

difference = (ahead-init_reg.reconstructed_x) ** 2

encoder_cost = T.mean( difference )

cross_entropy_cost = T.mean(log_reg.cross_entropy_binary(y))

#y_hat_mean = T.mean(log_reg.p_y_given_x,axis=0)
#
#z_hat_mean = T.mean(lin_reg.E_y_given_x,axis=0)
#
#z_variance = lin_reg.E_y_given_x - z_hat_mean
#z_var = z_variance.reshape((60,2,1)) #must reshape for outer product
#
#y_variance = log_reg.p_y_given_x - y_hat_mean
#y_var = y_variance.reshape((60,1,10))
#
#product = T.batched_dot(z_var,y_var) #an outer product across batches
#
#product_mean_sqr = (T.mean(product,axis=0) **2)
#
#covariance_cost = T.sum( product_mean_sqr )/2

cost = encoder_cost - cross_entropy_cost# + 10*covariance_cost

###########################################################
###########################################################

############ ESTABLISHING PARAMETERS AND GD  ##############
###########################################################

print '...establishing parameters and gradients'

# create a list of all model parameters
params = init_reg.params + lstm_1.params + lstm_2.params + log_reg.params + lstm_3.params + lstm_4.params

# updates from ADAM
updates = Adam(cost, params)

###########################################################
###########################################################

############ THEANO FUNC. FOR TRAINING, VAL., ETC.  #######
###########################################################

print '....compiling training and testing functions'

train_model = theano.function([sent,phonemes], outputs=[cost,encoder_cost,cross_entropy_cost], updates=updates,
      givens={
        x: sent[:-1],
	ahead: sent[1:],
        y: phonemes[:-1]})

probe_model = theano.function([sent,phonemes], outputs=[cost,encoder_cost,cross_entropy_cost], 
      givens={
        x: sent[:-1],
	ahead: sent[1:],
        y: phonemes[:-1]})

validate_model = theano.function(inputs=[sent,phonemes],outputs=[cost,encoder_cost,cross_entropy_cost],
      givens={
        x: sent[:-1],
	ahead: sent[1:],
        y: phonemes[:-1]})

###########################################################
###########################################################

############ MAIN TRAINING LOOP ###########################
###########################################################

print 'Starting training...'

epoch = 0
best_validation_loss = numpy.inf

last_e = time.time()

r_log=open(results_filename,'w')
r_log.write('Starting training...\n')
r_log.close()

while (epoch < n_epochs) :
    
    print 'Last epoch took: '+str(time.time()-last_e)
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()
    last_e = time.time()
    epoch = epoch + 1

    for minibatch_index in xrange(len(train_x)):
	sentence = train_x[minibatch_index]
        phonemes = train_y[minibatch_index]
	if len(sentence) != len(phonemes):
            if len(sentence)>len(phonemes):
	        sentence = sentence[:len(phonemes)]
	    else:	
		phonemes = phonemes[:len(sentence)]
	sentence = numpy.reshape(sentence,(len(sentence),1))
	minibatch_avg_cost = train_model(sentence,phonemes)
        print str(minibatch_avg_cost[0])+','+str(minibatch_avg_cost[1])+','+str(minibatch_avg_cost[2])
    
    # calculate validation error
    validation_losses = []
    for minibatch_index in xrange(len(test_x)):
	sentence = train_x[minibatch_index]
        phonemes = train_y[minibatch_index]
	if len(sentence) != len(phonemes):
            if len(sentence)>len(phonemes):
	        sentence = sentence[:len(phonemes)]
	    else:	
		phonemes = phonemes[:len(sentence)]
	sentence = numpy.reshape(sentence,(len(sentence),1))
        validation_losses.append(validate_model(sentence,phonemes))
    loss_totals = [i[0] for i in validation_losses]
    enc = [i[1] for i in validation_losses]
    cross = [i[2] for i in validation_losses]
    #cov = [i[3] for i in validation_losses]

    this_validation_loss = numpy.mean(loss_totals)
    encoder_loss = numpy.mean(enc)
    cross_loss = numpy.mean(cross)
    #cov_loss = numpy.mean(cov)
    
    # Printing and logging validation error
    print('epoch %i, validation error %f reconstruction error %f cross entropy error %f covariance error %f' % (epoch,this_validation_loss,encoder_loss,cross_loss))
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, validation error %f reconstruction error %f cross entropy error %f covariance error %f\n' % (epoch,this_validation_loss,encoder_loss,cross_loss))
    r_log.close()

    # Saving parameters if best result
    if this_validation_loss < best_validation_loss:
        #store data
        f = file(savefilename, 'wb')
        for obj in [params]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
