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
from load_timit import get_timit_specs_images

### Tunable parameters and variables ###
########################################

n_epochs=15

layer0_filters = 20
layer1_filters = 20
layer2_filters = 20
layer3_filters = 20
layer4_filters = 20

minibatch_size = 10
window_size = 60

savefilename = '/vega/stats/users/sl3368/TIMIT/saves/params/timit_conv_'+str(window_size)+'_5.save'
results_filename='/vega/stats/users/sl3368/TIMIT/results/timit_conv_'+str(window_size)+'_5.out'
datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'

train_x_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_spec_train_x.npz'
train_y_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_spec_train_y.npz'
test_x_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_spec_test_x.npz'
test_y_filename = '/vega/stats/users/sl3368/TIMIT/saves/timit_spec_test_y.npz'

#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

print 'Loading TIMIT Data..'
#start = time.time()
#
## Load the dataset
#train_x, train_y, test_x, test_y = get_timit_specs_images(60)
#print 'Finished Loading, took: '+str(time.time()-start)
#print 'Saving...'
#
#start = time.time()
#numpy.savez_compressed(train_x_filename,train_x=train_x)
#print 'Saved train x, took: '+str(time.time()-start)
#
#start = time.time()
#numpy.savez_compressed(train_y_filename,train_y=train_y)
#print 'Saved train y, took: '+str(time.time()-start)
#
#start = time.time()
#numpy.savez_compressed(test_x_filename,test_x=test_x)
#print 'Saved test x, took: '+str(time.time()-start)
#
#start = time.time()
#numpy.savez_compressed(test_y_filename,test_y=test_y)
#print 'Saved test y, took: '+str(time.time()-start)

start = time.time()
train_x = numpy.load(train_x_filename)['train_x']
print 'Loaded train x...took: '+str(time.time()-start)

start = time.time()
train_y = numpy.load(train_y_filename)['train_y']
print 'Loaded train y...'+str(time.time()-start)

start = time.time()
test_x = numpy.load(test_x_filename)['test_x']
print 'Loaded test x...'+str(time.time()-start)

start = time.time()
test_y = numpy.load(test_y_filename)['test_y']
print 'Loaded test y...'+str(time.time()-start)

print 'Loading complete...'

assert len(train_x)/5000 == len(train_y)
print 'Number of training sentences: '+str(len(train_x))

assert len(test_x)/5000 == len(test_y) 
print 'Number of testing sentences: '+str(len(test_x))

#remember clipping individual songs

###########################################################
###########################################################

############ CONSTRUCTING MODEL ARCHITECTURE ##############
###########################################################
print 'Building model...'

# allocate symbolic variables for the data

index = T.lscalar()  # index to a [mini]batch
x = T.tensor3('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
y = T.imatrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
y_enc = T.imatrix('y_enc')
rng = numpy.random.RandomState(1234)
number = T.matrix('number')
variation = T.matrix('variation')

# Reshape matrix of rasterized images of shape (batch_size, 2000 * 60)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((minibatch_size, 1, 60, 60))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(minibatch_size, 1, 60, 60),
    filter_shape=( layer0_filters, 1, 3, 3),
    poolsize=(2, 2),
    dim2 = 1
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(minibatch_size, layer0_filters, 30, 30),
    filter_shape=( layer1_filters, layer0_filters, 2, 2),
    poolsize=(2, 2),
    dim2 = 1
)

layer2 = LeNetConvPoolLayer(
    rng,
    input=layer1.output,
    image_shape=(minibatch_size, layer1_filters, 15, 15),
    filter_shape=( layer2_filters, layer1_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

layer3 = LeNetConvPoolLayer(
    rng,
    input=layer2.output,
    image_shape=(minibatch_size, layer2_filters, 15, 15),
    filter_shape=( layer3_filters, layer2_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

reg_input = layer3.output.flatten(2)

log_reg = LogisticRegression(reg_input,15*15*layer3_filters, 41)

lin_reg = LinearRegression(reg_input,15*15*layer3_filters,2,True)

log_input = log_reg.p_y_given_x
lin_input = lin_reg.E_y_given_x

log_reg.reconstruct(log_input)
lin_reg.reconstruct(lin_input)

reconstructed_regressions = T.concatenate([log_reg.reconstructed_x,lin_reg.reconstructed_x],axis=1)

reverse_layer = LinearRegression(reconstructed_regressions, 2*15*15*layer3_filters, 15*15*layer3_filters,False)

reconstruct = reverse_layer.E_y_given_x.reshape((minibatch_size,layer3_filters,15,15))

layer3.reverseConv(reconstruct,(minibatch_size,layer3_filters,15,15),(layer2_filters,layer3_filters,2,2))

layer2.reverseConv(layer3.reverseOutput,(minibatch_size,layer2_filters,15,15),(layer1_filters,layer2_filters,2,2))

layer1.reverseConv(layer2.reverseOutput,(minibatch_size,layer1_filters,30,30),(layer0_filters,layer1_filters,2,2))

layer0.reverseConv(layer1.reverseOutput,(minibatch_size,layer0_filters,60,60),(1,layer0_filters,3,3,))

difference = (layer0_input-layer0.reverseOutput) ** 2

encoder_cost = T.mean( difference )

cross_entropy_cost = T.mean(log_reg.cross_entropy_binary(y))

y_hat_mean = T.mean(log_reg.p_y_given_x,axis=0)

z_hat_mean = T.mean(lin_reg.E_y_given_x,axis=0)

z_variance = lin_reg.E_y_given_x - z_hat_mean
z_var = z_variance.reshape((minibatch_size,2,1)) #must reshape for outer product

y_variance = log_reg.p_y_given_x - y_hat_mean
y_var = y_variance.reshape((minibatch_size,1,41))

product = T.batched_dot(z_var,y_var) #an outer product across batches

product_mean_sqr = (T.mean(product,axis=0) **2)

covariance_cost = T.sum( product_mean_sqr )/2

cost = 12*encoder_cost + cross_entropy_cost + 10*covariance_cost

###########################################################
###########################################################

############ ESTABLISHING PARAMETERS AND GD  ##############
###########################################################

print '...establishing parameters and gradients'

# create a list of all model parameters
params = layer0.params + layer1.params + layer2.params + layer3.params + log_reg.params + lin_reg.params + reverse_layer.params

# updates from ADAM
updates = Adam(cost, params)

###########################################################
###########################################################

############ THEANO FUNC. FOR TRAINING, VAL., ETC.  #######
###########################################################

print '....compiling training and testing functions'

train_model = theano.function([x,y], outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost], updates=updates)

probe_model = theano.function([x,y], outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost])

validate_model = theano.function(inputs=[x,y], outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost])

#test_model_accuracy = theano.function([imgs], outputs = [log_input],
#	givens={x: imgs})
#
#build_model_reconstructions = theano.function([number,variation],outputs=[layer0.reverseOutput],
#        givens={ log_input: number, lin_input: variation})
#

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

    for minibatch_index in xrange(len(train_x)/5000):
	sentence = train_x[(minibatch_index*5000):(minibatch_index+1)*5000]
        phonemes = train_y[minibatch_index]
	if len(sentence) != len(phonemes):
            if len(sentence)>len(phonemes):
	        sentence = sentence[:len(phonemes)]
	    else:	
		phonemes = phonemes[:len(sentence)]
	sentence = numpy.reshape(sentence,(len(sentence),60))
	sentence = numpy.asarray(sentence,dtype=numpy.float32)
	for i in range(0,sentence.shape[0]-minibatch_size,minibatch_size):
	     input = numpy.zeros((minibatch_size,window_size,sentence.shape[1]))
	     for k in range(minibatch_size):
	        input[k] = sentence[i+k:i+k+window_size]
	     minibatch_avg_cost = train_model(input.astype(numpy.float32),phonemes[i:i+minibatch_size])
	     #print minibatch_avg_cost
             print str(minibatch_avg_cost[0])+','+str(minibatch_avg_cost[1])+','+str(minibatch_avg_cost[2])
    
    # calculate validation error
    validation_losses = []
    for minibatch_index in xrange(len(test_x)/5000):
	sentence = train_x[minibatch_index*5000:(minibatch_index+1)*5000]
        phonemes = train_y[minibatch_index]
	if len(sentence) != len(phonemes):
            if len(sentence)>len(phonemes):
	        sentence = sentence[:len(phonemes)]
	    else:	
		phonemes = phonemes[:len(sentence)]
	sentence = numpy.reshape(sentence,(len(sentence),60))
	sentence = numpy.asarray(sentence,dtype=numpy.float32)
	for i in range(0,sentence.shape[0],minibatch_size):
	     input = numpy.zeros((minibatch_size,window_size,sentence.shape[1]))
	     for k in range(minibatch_size):
	        input[k] = sentence[i+k:i+k+window_size]
	     validation_losses.append(validate_model(input.astype(numpy.float32),phonemes[i:i+minibatch_size]))
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
