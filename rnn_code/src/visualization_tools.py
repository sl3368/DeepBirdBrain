################################################
# Import statements
################################################

import cPickle
import os
from os import path
import sys
import time
import numpy
import theano
from theano import shared
import theano.tensor as T
from loading_functions import load_all_data, load_class_data_batch, load_class_data_vt, load_neural_data
from layer_classes import LinearRegression, Dropout, LSTM, RNN, hybridRNN , IRNN, PoissonRegression
from one_ahead import GradClip, clip_gradient
from misc import Adam
from numpy.random import rand

################################################
# Script Parameters
################################################


def generate_visualization_on_section(region,heldout,neuron_no,training_iterations,hidden,song_section,previous):
    
    region_dict = {'L1':0,'L2':2,'L3':4,'NC':6,'MLd':8}
    held_out_song = heldout
    brain_region = region
    brain_region_index = region_dict[brain_region]
    neuron = neuron_no
    prev_trained_size = previous.shape[0]
    n_epochs= training_iterations
    n_hidden = hidden
    song_depth = song_section
    
    print 'Running CV for held out song '+str(held_out_song)+' for brain region '+brain_region+' index at '+str(brain_region_index)+' iteration:'+str(song_depth)
    
    #Filepath for printing results
    results_filename='/vega/stats/users/sl3368/rnn_code/results/neural/dual_'+str(n_hidden)+'/visualizations/'+brain_region+'_'+str(held_out_song)+'_'+str(neuron)+'.out'
    #check if exists already, then load or not load 
    load_params_pr_filename = '/vega/stats/users/sl3368/rnn_code/saves/params/neural/dual_'+str(n_hidden)+'/'+brain_region+'_'+str(held_out_song)+'.save'
    if path.isfile(load_params_pr_filename):
        #print 'Will load previous regression parameters...'
        load_params_pr = True
    else:
        load_params_pr = False
    	
    song_size = 2459
    
    #filepath for saving parameters
    savefilename = '/vega/stats/users/sl3368/rnn_code/saves/params/neural/dual_'+str(n_hidden)+'/visualizations/'+brain_region+'_'+str(held_out_song)+'_'+str(neuron)+'.visualization'
    
    if path.isfile(savefilename):
        load_visualize = True
    else:
        load_visualize = False
    
    
    ################################################
    # Load Data
    ################################################
    dataset_info = load_all_data()
    stim = dataset_info[0]
    data_set_x = theano.shared(stim, borrow=True)
    
    #print 'Getting neural data...'
    
    neural_data = load_neural_data()
    
    ntrials = theano.shared(neural_data[brain_region_index],borrow=True)
    responses = theano.shared(neural_data[brain_region_index+1],borrow=True)
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    #print 'building the model...'
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    section = T.lscalar()
    
    init = numpy.zeros((song_depth,60)).astype('f')
    init[:prev_trained_size] = previous
    x = shared(init,borrow=True)
    
    y = T.matrix('y')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
    trial_no = T.matrix('trial_no')
    
    rng = numpy.random.RandomState(1234)
    
    # Architecture: input --> LSTM --> predict one-ahead
     
    lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)
   
    output = PoissonRegression(input=lstm_1.output, n_in=n_hidden, n_out=1)
    
    pred = output.E_y_given_x.T * trial_no[:,neuron]
    nll = output.negative_log_likelihood(y[:,neuron],trial_no[:,neuron],single=True)
    
    ################################
    # Objective function and GD
    ################################
    
    #print 'defining cost, parameters, and learning function...'
    
    # the cost we minimize during training is the negative log likelihood of
    # the model 
    cost = T.mean(nll)
    
    #Defining params
    params = [x]
    
    # updates from ADAM
    updates = Adam(cost, params)
    
    #######################
    # Objective function
    #######################
    
    #print 'compiling train....'
    
    train_model = theano.function(inputs=[index,section], outputs=cost,
            updates=updates,
            givens={
    	        trial_no: ntrials[index * song_size:((index * song_size)+section)],
                y: responses[index * song_size:((index * song_size)+section)]})
    
    #validate_model = theano.function(inputs=[index,section],
    #        outputs=[cost,nll.shape,pred.shape],
    #        givens={
    #	        trial_no: ntrials[index * song_size:((index * song_size)+section)],
    #            y: responses[index * song_size:((index * song_size)+section)]})
    
    #######################
    # Parameters and gradients
    #######################
    #print 'parameters and gradients...'
    
    if load_params_pr:
        #print 'loading LSTM parameters from file...'
        f = open( load_params_pr_filename)
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
        f.close()
    
    if load_params_pr:
        #print 'loading PR parameters from file...'
        f = open( load_params_pr_filename)
        old_p = cPickle.load(f)
        output.W.set_value(old_p[13].get_value(), borrow=True)
        output.b.set_value(old_p[14].get_value(), borrow=True)
        f.close()
    
    #if load_visualize:
    #    print 'loading visualization from file...'
    #    f = open(savefilename)
    #    old_v = cPickle.load(f)
    #    x.set_value(old_v, borrow=True)
    #    f.close()
    
    ###############
    # TRAIN MODEL #
    ###############
    #print 'training...'
    
    best_validation_loss = numpy.inf
    epoch = 0
    
    last_e = time.time()
    
    r_log=open(results_filename,'a')
    r_log.write('Starting training...\n')
    r_log.close()
    
    while (epoch < n_epochs):
        #print str(epoch)+' epoch took: '+str(time.time()-last_e)
        #r_log=open(results_filename, 'a')
        #r_log.write(str(epoch)+ ' epoch took: '+str(time.time()-last_e)+'\n')
        #r_log.close()
    
        last_e = time.time()
        epoch = epoch + 1
    
        mb_costs = []
    
        heldout = 0
    
        for minibatch_index in xrange(14):
            if(heldout==held_out_song):
                minibatch_avg_cost = train_model(minibatch_index,song_depth)
                #print minibatch_avg_cost
    	    	mb_costs.append(minibatch_avg_cost)
    	    heldout=heldout+1
    
        for minibatch_index in xrange(24,30):
            if(heldout==held_out_song):
    	        minibatch_avg_cost = train_model(minibatch_index,song_depth)
                #print minibatch_avg_cost
    	    	mb_costs.append(minibatch_avg_cost)
    	    heldout=heldout+1
    
        avg_cost = numpy.mean(mb_costs)
    
        
        # if we got the best validation score until now
        if avg_cost < best_validation_loss:
    	    
   	    best_validation_loss = avg_cost
            visualization = numpy.asarray(x.eval())
            
	    #store data
            f = file(savefilename, 'wb')
            for obj in [visualization]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            
    print('epoch %i, training error %f' %  (epoch, avg_cost))
    
    r_log=open(results_filename, 'a')
    r_log.write('epoch %i, training error %f' %  (epoch, avg_cost))
    r_log.close()
    
    return numpy.asarray(x.eval())

def generation_loop(region,heldout,neuron,iterations,hidden,step_size,song_sz):

    initial = numpy.zeros((1,60)).astype('f')
    for i in range(step_size,song_sz,step_size):
        initial = generate_visualization_on_section(region,heldout,neuron,iterations,hidden,i,initial)
    
    generate_visualization_on_section(region,heldout,neuron,iterations,hidden,song_sz,initial)

#arguments: script_name, region, held_out_song, neuron, iterations, hidden, stepsize, song_size
if __name__=='__main__':
    reg = sys.argv[1]
    heldout = int(sys.argv[2])
    neuron = int(sys.argv[3])
    it = int(sys.argv[4])
    hid = int(sys.argv[5])
    step_size = int(sys.argv[6])
    songsize = int(sys.argv[7])
    generation_loop(reg,heldout,neuron,it,hid,step_size,songsize)
