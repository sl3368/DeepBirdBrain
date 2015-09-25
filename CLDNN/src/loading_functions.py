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
    stimuli = numpy.transpose(numpy.array(f['norm_songs'],dtype=theano.config.floatX))
    maxBatchSize = numpy.array(f['maxBatchSize'])
    f.close()
    
    data_set_x = theano.shared(numpy.asarray(stimuli, dtype=theano.config.floatX), borrow=True)

#    rval = [data_set_x, maxBatchSize]
    rval = [stimuli,maxBatchSize]

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

def load_neural_data():
    dataset = '/vega/stats/users/sl3368/Data_LC/all_regions_prep_padded_A.mat'

    f = h5py.File(dataset)
    L1_trials = numpy.transpose(numpy.array(f['ntrials_L1'],dtype=theano.config.floatX))
    L2_trials = numpy.transpose(numpy.array(f['ntrials_L2'],dtype=theano.config.floatX))
    L3_trials = numpy.transpose(numpy.array(f['ntrials_L3'],dtype=theano.config.floatX))
    NC_trials = numpy.transpose(numpy.array(f['ntrials_NC'],dtype=theano.config.floatX))
    mld_trials = numpy.transpose(numpy.array(f['ntrials_mld'],dtype=theano.config.floatX))
    
    L1_responses = numpy.transpose(numpy.array(f['responses_L1'],dtype=theano.config.floatX))
    L2_responses = numpy.transpose(numpy.array(f['responses_L2'],dtype=theano.config.floatX))
    L3_responses = numpy.transpose(numpy.array(f['responses_L3'],dtype=theano.config.floatX))
    NC_responses = numpy.transpose(numpy.array(f['responses_NC'],dtype=theano.config.floatX))
    mld_responses = numpy.transpose(numpy.array(f['responses_mld'],dtype=theano.config.floatX))

    return L1_trials,L1_responses,L2_trials,L2_responses,L3_trials,L3_responses,NC_trials,NC_responses,mld_trials,mld_responses
