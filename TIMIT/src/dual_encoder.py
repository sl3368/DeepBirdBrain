import sys
import time
import numpy 
import theano
from theano import shared
import cPickle
from theano import tensor as T
from layer_classes import LeNetConvPoolLayer
from misc import Adam
from loading_functions import load_class_data_batch, load_class_data_vt

### Tunable parameters and variables ###
########################################

n_epochs=10

#Number of filters per layer
#must change filter sizes in layer instantiation
layer0_filters = 100
layer1_filters = 80
layer2_filters = 60
layer3_filters = 40
layer4_filters = 20

n_val_batches = 10
n_test_batches = 10
minibatch_size = 1 #should be 1, train single song at a time
song_size = 2000 #dependent on padding (regular,low,none)

savefilename = '/vega/stats/users/sl3368/encoder/saves/params/gpu_conv_ae_5_layer_10.save'
results_filename='/vega/stats/users/sl3368/encoder/results/gpu_conv_ae_5_layer_10.out'
datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'


#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

dataset_info = load_class_data_batch(datapathpre + 'LC_stim_1.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)

#validation and testing - for now, use last one
dataset_info_vt = load_class_data_vt(datapathpre + 'LC_stim_15.mat')
data_set_x_vt = dataset_info_vt[0]

n_batches = data_set_x.shape[0].eval()/song_size
n_train_batches = n_batches 

print 'Number of song for training in single chunk file: '+str(n_train_batches)

###########################################################
###########################################################

############ CONSTRUCTING MODEL ARCHITECTURE ##############
###########################################################


print 'Building model...'

# allocate symbolic variables for the data

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
rng = numpy.random.RandomState(1234)
	
# Reshape matrix of rasterized images of shape (batch_size, 2000 * 60)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((minibatch_size, 1, 2000, 60))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(minibatch_size, 1, 2000, 60),
    filter_shape=( layer0_filters, 1, 3, 3),
    poolsize=(2, 2),
    dim2 = 1
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(minibatch_size, layer0_filters, 1000, 30),
    filter_shape=( layer1_filters, layer0_filters, 2, 2),
    poolsize=(2, 2),
    dim2 = 1
)

layer2 = LeNetConvPoolLayer(
    rng,
    input=layer1.output,
    image_shape=(minibatch_size, layer1_filters, 500, 15),
    filter_shape=( layer2_filters, layer1_filters, 2, 2),
    poolsize=(2, 1),
    dim2 = 1
)

layer3 = LeNetConvPoolLayer(
    rng,
    input=layer2.output,
    image_shape=(minibatch_size, layer2_filters, 250, 15),
    filter_shape=( layer3_filters, layer2_filters, 2, 2),
    poolsize=(2, 1),
    dim2 = 1
)

layer4 = LeNetConvPoolLayer(
    rng,
    input=layer3.output,
    image_shape=(minibatch_size, layer3_filters, 125, 15),
    filter_shape=( layer4_filters, layer3_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

#need log_reg and decovariate layers

layer4.reverseConv(layer4.output,(1,layer4_filters,125,15),(layer3_filters,layer4_filters,2,2))

layer3.reverseConv(layer4.reverseOutput,(1,layer3_filters,250,15),(layer2_filters,layer3_filters,2,2))

layer2.reverseConv(layer3.reverseOutput,(1,layer2_filters,500,15),(layer1_filters,layer2_filters,2,2))

layer1.reverseConv(layer2.reverseOutput,(1,layer1_filters,1000,30),(layer0_filters,layer1_filters,2,2))

layer0.reverseConv(layer1.reverseOutput,(1,layer0_filters,2000,60),(1,layer0_filters,3,3,)) #filter flipped on first two axes

reconstructed = layer0.reverseOutput

cost = T.mean( (x-reconstructed) **2 )

###########################################################
###########################################################

############ ESTABLISHING PARAMETERS AND GD  ##############
###########################################################

print '...establishing parameters and gradients'

# create a list of all model parameters
params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params

# updates from ADAM
updates = Adam(cost, params)

###########################################################
###########################################################

############ THEANO FUNC. FOR TRAINING, VAL., ETC.  #######
###########################################################

print '....compiling training and testing functions'

train_model = theano.function([index], cost, updates=updates,
      givens={
        x: data_set_x[index * song_size: (index + 1) * song_size]})

probe_model = theano.function([index], outputs=[layer0.reverseOutput], givens={x:data_set_x[index * song_size: (index + 1) * song_size]})

validate_model = theano.function(inputs=[index],
        outputs=cost,
        givens={
            x: data_set_x_vt[index * song_size:((index + 1) * song_size)]})

###########################################################
###########################################################

############ MAIN TRAINING LOOP ###########################
###########################################################

print 'Starting training...'

epoch = 0
reset_epoch=0
data_part = 1
dataparts = numpy.r_[range(1,11)]

best_validation_loss = numpy.inf

track_train = list()
track_valid = list()
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
        
	minibatch_avg_cost = train_model(minibatch_index)
        #print minibatch_avg_cost
        track_train.append(minibatch_avg_cost)
    
    # calculate validation error
    validation_losses = [validate_model(i) for i in xrange(n_train_batches/60)]
    this_validation_loss = numpy.mean(validation_losses)
    
    # Printing and logging validation error
    print('epoch %i, validation error %f' % (epoch,this_validation_loss))
    track_valid.append(this_validation_loss)
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, validation error %f\n' %(epoch,this_validation_loss))
    r_log.close()

    # Saving parameters if best result
    if this_validation_loss < best_validation_loss:
        #store data
        f = file(savefilename, 'wb')
        for obj in [params]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
