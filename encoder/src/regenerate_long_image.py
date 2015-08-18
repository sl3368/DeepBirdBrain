from random import randint
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

n_epochs=30

#Number of filters per layer
#must change filter sizes in layer instantiation
layer0_filters = 200
layer1_filters = 100
layer2_filters = 100
layer3_filters = 100
layer4_filters = 50

minibatch_size = 10 #number of images
song_size = 1000 #dependent on padding (regular,low,none)

savefilename = '/vega/stats/users/sl3368/encoder/results/reconstructed/reconstructed_long_convae_nopad_1st_img.out'

datapathpre = '/vega/stats/users/sl3368/Data_LC/NopadNormData/'

load_params = True
load_params_filename = '/vega/stats/users/sl3368/encoder/saves/params/long/long_convae_5layer_nopad_20_1st.save'

#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

dataset_info = load_class_data_batch(datapathpre + 'LC_stim_5.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)
n_batches = data_set_x.shape[0].eval()/(song_size*minibatch_size)

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
layer0_input = x.reshape((minibatch_size, 1, 1000, 60))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(minibatch_size, 1, 1000, 60),
    filter_shape=( layer0_filters, 1, 5, 60),
    poolsize=(5, 1),
    dim2 = 1
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(minibatch_size, layer0_filters, 200, 60),
    filter_shape=( layer1_filters, layer0_filters, 2, 60),
    poolsize=(2, 1),
    dim2 = 1
)

layer2 = LeNetConvPoolLayer(
    rng,
    input=layer1.output,
    image_shape=(minibatch_size, layer1_filters, 100, 60),
    filter_shape=( layer2_filters, layer1_filters, 2, 60),
    poolsize=(2, 1),
    dim2 = 1
)

layer3 = LeNetConvPoolLayer(
    rng,
    input=layer2.output,
    image_shape=(minibatch_size, layer2_filters, 50, 60),
    filter_shape=( layer3_filters, layer2_filters, 2, 60),
    poolsize=(2, 1),
    dim2 = 1
)

layer4 = LeNetConvPoolLayer(
    rng,
    input=layer3.output,
    image_shape=(minibatch_size, layer3_filters, 25, 60),
    filter_shape=( layer4_filters, layer3_filters, 2, 60),
    poolsize=(1, 1),
    dim2 = 1
)


layer4.reverseConv(layer4.output,(minibatch_size,layer4_filters,25,60),(layer3_filters,layer4_filters,2,60))

layer3.reverseConv(layer4.reverseOutput,(minibatch_size,layer3_filters,50,60),(layer2_filters,layer3_filters,2,60))

layer2.reverseConv(layer3.reverseOutput,(minibatch_size,layer2_filters,100,60),(layer1_filters,layer2_filters,2,60))

layer1.reverseConv(layer2.reverseOutput,(minibatch_size,layer1_filters,200,60),(layer0_filters,layer1_filters,2,60))

layer0.reverseConv(layer1.reverseOutput,(minibatch_size,layer0_filters,1000,60),(1,layer0_filters,5,60)) #filter flipped on first two axes

reconstructed = layer0.reverseOutput

cost = T.mean( (layer0_input-reconstructed) **2 )

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

probe_model = theano.function([index], outputs=[reconstructed], givens={x:data_set_x[index * song_size * minibatch_size : (index + 1) * song_size * minibatch_size]})


###########################################################
###########################################################
############ LOAD PARAMETERS INTO CONVNET #################
###########################################################

if load_params:
    f = open( load_params_filename)
    old_p = cPickle.load(f)
    f.close()
    layer0.W.set_value(old_p[0].get_value(), borrow=True)
    layer0.b.set_value(old_p[1].get_value(), borrow=True)
    layer1.W.set_value(old_p[2].get_value(), borrow=True)
    layer1.b.set_value(old_p[3].get_value(), borrow=True)
    layer2.W.set_value(old_p[4].get_value(), borrow=True)
    layer2.b.set_value(old_p[5].get_value(), borrow=True)
    layer3.W.set_value(old_p[6].get_value(), borrow=True)
    layer3.b.set_value(old_p[7].get_value(), borrow=True)
    layer4.W.set_value(old_p[8].get_value(), borrow=True)
    layer4.b.set_value(old_p[9].get_value(), borrow=True)


###########################################################
###########################################################
############ MAIN TRAINING LOOP ###########################
###########################################################

print 'Generating...'
i_rand = randint(0,n_batches)

truth = data_set_x[i_rand * song_size * minibatch_size : (i_rand + 1) * song_size * minibatch_size].eval()
reconstructed = numpy.asarray( probe_model(i_rand) )

#store data
f = file(savefilename, 'wb')
for obj in [[truth]+[reconstructed]]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
