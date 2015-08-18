"""
Collection of loading functions

"""
import cPickle
import gzip
import os
import sys
import time

import numpy
import h5py
from pylab import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_all_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'
    dataset='/vega/stats/users/jsm2183/A_no_lag.mat'
    f = h5py.File(dataset)
    stimuli = numpy.transpose(f['norm_songs'])
    maxBatchSize = numpy.array(f['maxBatchSize'])
    f.close()
    
    data_set_x = theano.shared(numpy.asarray(stimuli, dtype=theano.config.floatX), borrow=True)

    rval = [data_set_x, maxBatchSize]

    return rval


def load_class_data_batch(dataset):
    print '...loading data'
    f = h5py.File(dataset)
    stimuli = numpy.transpose(f['stimulus_zscore'])
    f.close()
    
    rval = [numpy.asarray(stimuli, dtype=theano.config.floatX)]
    
    return rval


def load_class_data_vt(dataset):
    print '...loading data'
    f = h5py.File(dataset)
    stimuli = numpy.transpose(f['stimulus_zscore'])
    f.close()
    stimuli = stimuli[0:20*10000,:]

    data_set_x_vt = theano.shared(numpy.asarray(stimuli, dtype=theano.config.floatX), borrow=True)
    
    rval = [data_set_x_vt]
    
    return rval


