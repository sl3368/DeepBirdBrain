"""
Single-layer neural network with poisson output units
Code based on MLP tutorial code for theano

"""
__docformat__ = 'restructedtext en'

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

from loading_functions import load_all_data
from layer_classes import LinearRegression, Dropout, LSTM, RNN

class GradClip(theano.compile.ViewOp):
    """
    Here we clip the gradients as Alex Graves does in his
    recurrent neural networks. In particular this prevents
    explosion of gradients during backpropagation.
    The original poster of this code was Alex Lamb,
    [here](https://groups.google.com/forum/#!topic/theano-dev/GaJwGw6emK0).
    """

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]


def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)

def SGD_training(learning_rate=1, n_epochs=1000):
    """
    stochastic gradient descent optimization

   """
    dataset_info = load_all_data()

    data_set_x = dataset_info[0]

    maxBatchSize = numpy.int_(dataset_info[1])

    batch_size = maxBatchSize
    n_train_batches = 28
    #n_valid_batches = 1
    #n_test_batches = 1

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
    d_input = Dropout(rng, is_train, x)
    n_hidden = 100
    lstm_1 = LSTM(rng, d_input.output, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)
    #lstm_1 = RNN(rng, d_input.output, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden) #vanilla rnn
    d_lstm_1 = Dropout(rng, is_train, lstm_1.output)
    output = LinearRegression(input=d_lstm_1.output, n_in=n_hidden, n_out=data_set_x.get_value(borrow=True).shape[1])

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
            outputs=[cost, output.E_y_given_x],
            givens={
                x: data_set_x[index * batch_size:((index + 1) * batch_size - 1)],
                y: data_set_x[(index * batch_size + 1):(index + 1) * batch_size],
                is_train: numpy.cast['int32'](0)})

    # wanted to use below indexes and have different sized batches, but this didn't work
    #[int(batchBreaks[index]-1):int(batchBreaks[(index+1)]-1)]

    validate_model = theano.function(inputs=[index],
            outputs=cost,
            givens={
                x: data_set_x[index * batch_size:((index + 1) * batch_size - 1)],
                y: data_set_x[(index * batch_size + 1):(index + 1) * batch_size],
                is_train: numpy.cast['int32'](0)})

    #######################
    # Parameters and gradients
    #######################
    print '... parameters and gradients'

    # create a list (concatenated) of all model parameters to be fit by gradient descent
    #order: [self.W, self.b]
    params = lstm_1.params + output.params
    params_helper = lstm_1.params_helper + output.params_helper
    params_helper2 = lstm_1.params_helper2 + output.params_helper2

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
                y: data_set_x[(index * batch_size + 1):(index + 1) * batch_size],
                is_train: numpy.cast['int32'](0)})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
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

    #best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    #test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print minibatch_avg_cost
 
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute absolute error loss on validation set
                validation_losses = [validate_model(i) for i
                                     in [28]]
                this_validation_loss = numpy.mean(validation_losses) #mean over batches
                print('epoch %i, minibatch %i, validation error %f' %
                     (epoch, minibatch_index + 1,
                      this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in [29]]
                    #test_score = numpy.mean(test_losses)
                    test_cost, test_pred = test_model(29)
                    #test_cost, test_costs_separate, test_pred_separate, test_actual_separate = test_model(29)

                    print(('     epoch %i, minibatch %i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1,
                           numpy.sum(test_cost)))

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
    f = file('results/params.save', 'wb')
    for obj in [params + [test_cost] + [test_pred]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    #plot the predicted trace
    #plt.clf()
    #plt.subplot(2,1,1)
    #plt.subplot(2, 1, 1)
    #plt.plot(test_pred, 'k')
    #plt.title('test data prediction')
    #plt.ylabel('predicted rate')
    #plt.subplot(2, 1, 2)
    #plt.plot(test_actual, 'k')
    #plt.xlabel('timebins')
    #plt.ylabel('actual spikes')
    #plt.savefig('single_fit_results/trace_n_' + str(n_index) + '_lam_' + str(L1_reg) + '.png')

    #plot the params and then show
    #plt.clf()
    #vis = plt.imshow(numpy.reshape(Layer0.W.eval(),(60,10),order='F'))
    ##tmp = numpy.max(numpy.abs(params[0].eval()))
    ##vis.set_clim(-tmp,tmp)
    #plt.colorbar()
    #plt.savefig('single_fit_results/RF_n_' + str(n_index) + '_lam_' + str(L1_reg) + '.png')
    #print numpy.max(numpy.abs(params[0].eval()))    
    #print numpy.mean(numpy.abs(params[0].eval()))
    #print numpy.median(numpy.abs(params[0].eval()))

#loop over neurons and cross validation parameters
if __name__ == '__main__':
    SGD_training()





