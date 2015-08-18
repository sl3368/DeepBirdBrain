import sys
import time
import numpy 
import theano
from theano import shared
import cPickle, gzip
from theano import tensor as T
from layer_classes import LeNetConvPoolLayer, LogisticRegression, LinearRegression
from misc import Adam
from loading_functions import load_class_data_batch, load_class_data_vt, shared_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score

### Tunable parameters and variables ###
########################################

n_epochs=15

#Number of filters per layer
#must change filter sizes in layer instantiation
layer0_filters = 20
layer1_filters = 20
layer2_filters = 20
layer3_filters = 20
layer4_filters = 20

minibatch_size = 60

savefilename = '/vega/stats/users/sl3368/TIMIT/saves/params/deep_dualnet_5_20_15.save'
results_filename='/vega/stats/users/sl3368/TIMIT/results/deep_dualnet_5_20_15.out'
datapathpre = '/vega/stats/users/sl3368/Data_LC/LowNormData/'

output_imgs_filename='/vega/stats/users/sl3368/TIMIT/results/deep_dualnet_5_20_15.imgs'
#######################################
#######################################


######## LOADING TRAINING AND TESTING DATA #################
###########################################################

print 'Loading MNIST Data Set'

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / minibatch_size
print 'Number of training batches: '+str(n_train_batches)
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / minibatch_size
print 'Number of validation batches: '+str(n_valid_batches)
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / minibatch_size
print 'Number of test batches: '+str(n_test_batches)

###########################################################
###########################################################

############ CONSTRUCTING MODEL ARCHITECTURE ##############
###########################################################


print 'Building model...'

# allocate symbolic variables for the data

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
y = T.imatrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
y_enc = T.imatrix('y_enc')
rng = numpy.random.RandomState(1234)
number = T.matrix('number')
variation = T.matrix('variation')

# Reshape matrix of rasterized images of shape (batch_size, 2000 * 60)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((minibatch_size, 1, 28, 28))

layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(minibatch_size, 1, 28, 28),
    filter_shape=( layer0_filters, 1, 3, 3),
    poolsize=(2, 2),
    dim2 = 1
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(minibatch_size, layer0_filters, 14, 14),
    filter_shape=( layer1_filters, layer0_filters, 2, 2),
    poolsize=(2, 2),
    dim2 = 1
)

layer2 = LeNetConvPoolLayer(
    rng,
    input=layer1.output,
    image_shape=(minibatch_size, layer1_filters, 7, 7),
    filter_shape=( layer2_filters, layer1_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

layer3 = LeNetConvPoolLayer(
    rng,
    input=layer2.output,
    image_shape=(minibatch_size, layer2_filters, 7, 7),
    filter_shape=( layer3_filters, layer2_filters, 2, 2),
    poolsize=(1, 1),
    dim2 = 1
)

reg_input = layer3.output.flatten(2)

log_reg = LogisticRegression(reg_input,7*7*layer3_filters, 10)

lin_reg = LinearRegression(reg_input,7*7*layer3_filters,2,True)

log_input = log_reg.p_y_given_x
lin_input = lin_reg.E_y_given_x

log_reg.reconstruct(log_input)
lin_reg.reconstruct(lin_input)

reconstructed_regressions = T.concatenate([log_reg.reconstructed_x,lin_reg.reconstructed_x],axis=1)

reverse_layer = LinearRegression(reconstructed_regressions, 2*7*7*layer3_filters, 7*7*layer3_filters,False)

reconstruct = reverse_layer.E_y_given_x.reshape((minibatch_size,layer3_filters,7,7))

layer3.reverseConv(reconstruct,(minibatch_size,layer3_filters,7,7),(layer2_filters,layer3_filters,2,2))

layer2.reverseConv(layer3.reverseOutput,(minibatch_size,layer2_filters,7,7),(layer1_filters,layer2_filters,2,2))

layer1.reverseConv(layer2.reverseOutput,(minibatch_size,layer1_filters,14,14),(layer0_filters,layer1_filters,2,2))

layer0.reverseConv(layer1.reverseOutput,(minibatch_size,layer0_filters,28,28),(1,layer0_filters,3,3,))

difference = (layer0_input-layer0.reverseOutput) ** 2

encoder_cost = T.mean( difference )

cross_entropy_cost = T.mean(log_reg.cross_entropy_binary(y))

y_hat_mean = T.mean(log_reg.p_y_given_x,axis=0)

z_hat_mean = T.mean(lin_reg.E_y_given_x,axis=0)

z_variance = lin_reg.E_y_given_x - z_hat_mean
z_var = z_variance.reshape((60,2,1)) #must reshape for outer product

y_variance = log_reg.p_y_given_x - y_hat_mean
y_var = y_variance.reshape((60,1,10))

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

train_model = theano.function([index,y_enc], outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost], updates=updates,
      givens={
        x: train_set_x[index * minibatch_size: ((index + 1) * minibatch_size)],
        y: y_enc})

probe_model = theano.function([index,y_enc], outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost], givens={x: train_set_x[index * minibatch_size: (index + 1) * minibatch_size]
       								       ,y: y_enc})

validate_model = theano.function(inputs=[index,y_enc],
        outputs=[cost,encoder_cost,cross_entropy_cost,covariance_cost],
        givens={
           x: valid_set_x[index * minibatch_size: ((index + 1) * minibatch_size)],
           y: y_enc})

test_model_accuracy = theano.function([index], outputs = [log_input],
      givens={
        x: test_set_x[index * minibatch_size: ((index + 1) * minibatch_size)]})

build_model_reconstructions = theano.function([number,variation],outputs=[layer0.reverseOutput],
	givens={ log_input: number, lin_input: variation})

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

enc = OneHotEncoder(n_values=10,dtype=numpy.int32,sparse=False)

while (epoch < n_epochs) :
    
    print 'Last epoch took: '+str(time.time()-last_e)
    r_log=open(results_filename, 'a')
    r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
    r_log.close()
    last_e = time.time()
    epoch = epoch + 1

    for minibatch_index in xrange(n_train_batches):
	index = minibatch_index
	y_vals_single=train_set_y[index * minibatch_size: (index + 1) * minibatch_size].eval()
	y_vals=numpy.reshape(y_vals_single,(minibatch_size,1))
	target = enc.fit_transform(y_vals)
	minibatch_avg_cost = train_model(minibatch_index,target)
        print str(minibatch_avg_cost[0])+','+str(minibatch_avg_cost[1])+','+str(minibatch_avg_cost[2])+','+str(minibatch_avg_cost[3])
    
    # calculate validation error
    validation_losses=[]
    for minibatch_index in xrange(n_valid_batches):
	index = minibatch_index
	y_vals_single=valid_set_y[index * minibatch_size: (index + 1) * minibatch_size].eval()
	y_vals=numpy.reshape(y_vals_single,(minibatch_size,1))
	target = enc.fit_transform(y_vals)
	minibatch_avg_cost = validate_model(minibatch_index,target)
	validation_losses.append(minibatch_avg_cost)

    loss_totals = [i[0] for i in validation_losses]
    encoder = [i[1] for i in validation_losses]
    cross = [i[2] for i in validation_losses]
    cov = [i[3] for i in validation_losses]

    this_validation_loss = numpy.mean(loss_totals)
    encoder_loss = numpy.mean(encoder)
    cross_loss = numpy.mean(cross)
    cov_loss = numpy.mean(cov)
    print('epoch %i, validation error %f reconstruction error %f cross entropy error %f covariance error %f' % (epoch,this_validation_loss,encoder_loss,cross_loss,cov_loss))
    
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, validation error %f reconstruction error %f cross entropy error %f covariance error %f\n' % (epoch,this_validation_loss,encoder_loss,cross_loss,cov_loss))
    r_log.close()

    # Saving parameters if best result
    if this_validation_loss < best_validation_loss:
        #store data
        f = file(savefilename, 'wb')
        for obj in [params]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

	#perform testing for accuracy
	test_precision = []
        for minibatch_index in xrange(n_test_batches-1):
    	    index = minibatch_index
    	    y_vals_single=test_set_y[index * minibatch_size: (index + 1) * minibatch_size].eval()
    	    y_vals=numpy.reshape(y_vals_single,(minibatch_size,1))
    	    target = enc.fit_transform(y_vals)
    	    minibatch_avg_cost = test_model_accuracy(minibatch_index)
    	    precision = precision_score(y_vals_single,numpy.argmax(minibatch_avg_cost[0],axis=1),average='micro')
	    test_precision.append(precision)
	print 'Epoch test mean precision: '+str(numpy.mean(test_precision))
        r_log=open(results_filename, 'a')
        r_log.write('Epoch test mean precision: '+str(numpy.mean(test_precision))+'\n')
        r_log.close()

	new_enc = OneHotEncoder(n_values=10,dtype=numpy.float32,sparse=False)

	#perform reconstruction with variation
	numbers = [i for i in range(10)]
	number_reconstruction = []
	for number in numbers:
	    encoded_numbers = new_enc.fit_transform(numpy.reshape(numpy.asarray([number for k in range(minibatch_size)],numpy.float64),(minibatch_size,1)))
	    var_params = []
	    for v1 in range(-1,2):
		v2_params = []
		for v2 in range(-1,2):
		    v_params = numpy.reshape(numpy.asarray([[v1,v2] for k in range(minibatch_size)],numpy.float32),(minibatch_size,2))
		    reconstructed = build_model_reconstructions(encoded_numbers,v_params)
		    v2_params.append(reconstructed[0][0][0])
		var_params.append(v2_params)
	    number_reconstruction.append(var_params)
	
	f = file(output_imgs_filename,'wb')
	for obj in [number_reconstruction]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
