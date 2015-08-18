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
from random import randint
from numpy.random import rand

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

savefilename = '/vega/stats/users/sl3368/encoder/results/textures/final_gram_imgs_'
results_filename='/vega/stats/users/sl3368/encoder/results/textures/texture_gen_convae_5layer_10_'
datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'
originalfn = '/vega/stats/users/sl3368/encoder/results/textures/original_gram_imgs_'
initfn = '/vega/stats/users/sl3368/encoder/results/textures/init_gram_imgs_'

load_params = True
load_params_filename = '/vega/stats/users/sl3368/encoder/saves/params/conv_ae_5_layer_10_2nd.save'

#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

dataset_info = load_class_data_batch(datapathpre + 'LC_stim_1.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)

#validation and testing - for now, use last one
#dataset_info_vt = load_class_data_vt(datapathpre + 'LC_stim_15.mat')
#data_set_x_vt = dataset_info_vt[0]

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
init = rand(1,1,2000,60).astype('f')
x = shared(init,borrow=True)

T_mat0 = T.matrix('T_mat0')
T_mat1 = T.matrix('T_mat1')
T_mat2 = T.matrix('T_mat2')
T_mat3 = T.matrix('T_mat3')
T_mat4 = T.matrix('T_mat4')
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

layer0.calcGramMatrix(False,layer0_filters, 2000 * 60)
layer1.calcGramMatrix(False,layer1_filters, 1000 * 30)
layer2.calcGramMatrix(False,layer2_filters, 500 * 15)
layer3.calcGramMatrix(False,layer3_filters, 250 * 15)
layer4.calcGramMatrix(False,layer4_filters, 125 * 15)


g0_matrix = layer0.G
g1_matrix = layer1.G
g2_matrix = layer2.G
g3_matrix = layer3.G
g4_matrix = layer4.G

cost0 = T.sum( (T_mat0-g0_matrix) **2 )/ (layer0_filters*layer0_filters*2000*2000*60*60*4)
cost1 = T.sum( (T_mat1-g1_matrix) **2 )/ (layer1_filters*layer1_filters*1000*1000*30*30*4)
cost2 = T.sum( (T_mat2-g2_matrix) **2 )/ (layer2_filters*layer2_filters*500*500*15*15*4)
cost3 = T.sum( (T_mat3-g3_matrix) **2 )/ (layer3_filters*layer3_filters*250*250*15*15*4)
cost4 = T.sum( (T_mat4-g4_matrix) **2 )/ (layer4_filters*layer4_filters*125*125*15*15*4)

cost = cost4 + cost3 + cost2 + cost1 + cost0

########REAL GRAM MATRIX CALCULATION MODEL################
r_x = T.matrix('x')

rng = numpy.random.RandomState(1334)
	
# Reshape matrix of rasterized images of shape (batch_size, 2000 * 60)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
r_layer0_input = r_x.reshape((minibatch_size, 1, 2000, 60))

r_layer0 = LeNetConvPoolLayer(
    rng,
    input=r_layer0_input,
    image_shape=(minibatch_size, 1, 2000, 60),
    filter_shape=( layer0_filters, 1, 3, 3),
    poolsize=(2, 2),
    dim2 = 1
)

r_layer1 = LeNetConvPoolLayer(
    rng,
    input=r_layer0.output,
    image_shape=(minibatch_size, layer0_filters, 1000, 30),
    filter_shape=( layer1_filters, layer0_filters, 2, 2),
    poolsize=(2, 2),
    dim2 = 1
)

r_layer2 = LeNetConvPoolLayer(
    rng,
    input=r_layer1.output,
    image_shape=(minibatch_size, layer1_filters, 500, 15),
    filter_shape=( layer2_filters, layer1_filters, 2, 2),
    poolsize=(2, 1),
    dim2 = 1
)

r_layer3 = LeNetConvPoolLayer(
    rng,
    input=r_layer2.output,
    image_shape=(minibatch_size, layer2_filters, 250, 15),
    filter_shape=( layer3_filters, layer2_filters, 2, 2),
    poolsize=(2, 1),
    dim2 = 1
)

r_layer4 = LeNetConvPoolLayer(
    rng,
    input=r_layer3.output,
    image_shape=(minibatch_size, layer3_filters, 125, 15),
    filter_shape=( layer4_filters, layer3_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

r_layer0.calcGramMatrix(False,layer0_filters, 2000 * 60)
r_layer1.calcGramMatrix(False,layer1_filters, 1000 * 30)
r_layer2.calcGramMatrix(False,layer2_filters, 500 * 15)
r_layer3.calcGramMatrix(False,layer3_filters, 250 * 15)
r_layer4.calcGramMatrix(False,layer4_filters, 125 * 15)


r0_g_matrix = r_layer0.G
r1_g_matrix = r_layer1.G
r2_g_matrix = r_layer2.G
r3_g_matrix = r_layer3.G
r4_g_matrix = r_layer4.G

###########################################################
###########################################################

############ ESTABLISHING PARAMETERS AND GD  ##############
###########################################################

print '...establishing parameters and gradients'

# create a list of all model parameters
params = [x]

#print params

# updates from ADAM
updates = Adam(cost, params)

###########################################################
###########################################################

############ THEANO FUNC. FOR GD AND TRUTH MATRIX  #######
###########################################################

print '....compiling training and testing functions'

generate_image = theano.function([T_mat4,T_mat3,T_mat2,T_mat1,T_mat0], cost, updates=updates) 

get_T = theano.function([index], outputs=[r4_g_matrix,r3_g_matrix,r2_g_matrix,r1_g_matrix,r0_g_matrix], givens={r_x:data_set_x[index * song_size: (index + 1) * song_size]})

get_white_noise = theano.function([],[g4_matrix,g3_matrix,g2_matrix,g1_matrix,g0_matrix])

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
    r_layer0.W.set_value(old_p[0].get_value(), borrow=True)
    r_layer0.b.set_value(old_p[1].get_value(), borrow=True)
    r_layer1.W.set_value(old_p[2].get_value(), borrow=True)
    r_layer1.b.set_value(old_p[3].get_value(), borrow=True)
    r_layer2.W.set_value(old_p[4].get_value(), borrow=True)
    r_layer2.b.set_value(old_p[5].get_value(), borrow=True)
    r_layer3.W.set_value(old_p[6].get_value(), borrow=True)
    r_layer3.b.set_value(old_p[7].get_value(), borrow=True)
    r_layer4.W.set_value(old_p[8].get_value(), borrow=True)
    r_layer4.b.set_value(old_p[9].get_value(), borrow=True)


###########################################################
###########################################################

############ MAIN GENERATION LOOP #########################
###########################################################

print 'Selecting random image and getting gram matrix...'

i_rand = randint(0,2000)
true = get_T(i_rand)
#print true

#save original image and gram
orig_img = data_set_x[i_rand * song_size: (i_rand + 1) * song_size]
o_f = file(originalfn+str(i_rand)+'.save', 'wb')
for obj in [[orig_img.eval()]+[true[4]]+[true[3]]+[true[2]]+[true[1]]+[true[0]]]:
    cPickle.dump(obj, o_f, protocol=cPickle.HIGHEST_PROTOCOL)
o_f.close()

epoch = 0
last_e = time.time()

r_log=open(results_filename+str(i_rand)+'.out','w')
r_log.write('Starting training...\n')
r_log.close()

#Get initial gram and save image
init_white_noise_img = x.get_value()
init_gram = get_white_noise()
i_f = file(initfn+str(i_rand)+'.save','wb')
for obj in [[init_white_noise_img] + [init_gram[4]] + [init_gram[3]] + [init_gram[2]] + [init_gram[1]] + [init_gram[0]]]:
    cPickle.dump(obj,i_f, protocol=cPickle.HIGHEST_PROTOCOL)
i_f.close()


print 'Each epoch is: '+str(n_train_batches)
print 'Starting generation...'
while (epoch < n_epochs) :
    
    print 'Last epoch took: '+str(time.time()-last_e)
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()
    last_e = time.time()
    epoch = epoch + 1

    for minibatch_index in xrange(n_train_batches):
        
	minibatch_avg_cost = generate_image(true[0],true[1],true[2],true[3],true[4])
        print minibatch_avg_cost
	r_log=open(results_filename+str(i_rand)+'.out','a')
	r_log.write('epoch %i, minibatch error %f\n' %(epoch,minibatch_avg_cost))
	r_log.close()

    e_gram = get_white_noise()
    wn_img = numpy.asarray(x.eval())
    #store data
    f = file(savefilename+str(i_rand)+'.save', 'wb')
    for obj in [[wn_img] + [e_gram[4]] + [e_gram[3]] + [e_gram[2]] + [e_gram[1]] + [e_gram[0]]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
