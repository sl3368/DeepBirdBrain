"""
Collection of layer types

"""

import numpy

import theano
import theano.tensor as T


# This class is to build a LSTM RNN layer (for now only excitatory)
# Used notation and equations from: http://deeplearning.net/tutorial/lstm.html
class LSTM(object):
    def __init__(self, rng, input, n_in, n_out):
        
        self.input = input
        
        w_mag = .01

        ##  Create parameters ##
        # initializations?? #
        # Weights from input to gates
        W_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to input gate
        W_i = theano.shared(value=W_i_values, name='W_i', borrow=True)
        W_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to forget gate
        W_f = theano.shared(value=W_f_values, name='W_f', borrow=True)
        W_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to memory cells directly
        W_c = theano.shared(value=W_c_values, name='W_c', borrow=True)
        W_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to output gate
        W_o = theano.shared(value=W_o_values, name='W_o', borrow=True)
        
        # Weights from previous output to gates
        U_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to input gate
        U_i = theano.shared(value=U_i_values, name='U_i', borrow=True)
        U_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to forget gate
        U_f = theano.shared(value=U_f_values, name='U_f', borrow=True)
        U_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to memory cell
        U_c = theano.shared(value=U_c_values, name='U_c', borrow=True)
        U_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to output gate
        U_o = theano.shared(value=U_o_values, name='U_o', borrow=True)
        
        V_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from memory cell state to output gate
        V_o = theano.shared(value=V_o_values, name='V_o', borrow=True)
        
        # Biases of gates
        b_i_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_i = theano.shared(value=b_i_values, name='b_i', borrow=True)
        b_f_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_f = theano.shared(value=b_f_values, name='b_f', borrow=True)
        b_c_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_c = theano.shared(value=b_c_values, name='b_c', borrow=True)
        b_o_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_o = theano.shared(value=b_o_values, name='b_o', borrow=True)
        
        
        # Assign parameters to self
        self.W_i = W_i
        self.W_f = W_f
        self.W_c = W_c
        self.W_o = W_o
        self.U_i = U_i
        self.U_f = U_f
        self.U_c = U_c
        self.U_o = U_o
        self.V_o = V_o
        self.b_i = b_i
        self.b_f = b_f
        self.b_c = b_c
        self.b_o = b_o
        
        # Helper variables for adagrad
        self.W_i_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_i_helper', borrow=True)
        self.W_f_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_f_helper', borrow=True)
        self.W_c_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_c_helper', borrow=True)
        self.W_o_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_o_helper', borrow=True)
        self.U_i_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_i_helper', borrow=True)
        self.U_f_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_f_helper', borrow=True)
        self.U_c_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_c_helper', borrow=True)
        self.U_o_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_o_helper', borrow=True)
        self.V_o_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='V_o_helper', borrow=True)
        self.b_i_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_i_helper', borrow=True)
        self.b_f_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_f_helper', borrow=True)
        self.b_c_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_c_helper', borrow=True)
        self.b_o_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_o_helper', borrow=True)
                                                          
        # Helper variables for L1
        self.W_i_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_i_helper2', borrow=True)
        self.W_f_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_f_helper2', borrow=True)
        self.W_c_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_c_helper2', borrow=True)
        self.W_o_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_o_helper2', borrow=True)
        self.U_i_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_i_helper2', borrow=True)
        self.U_f_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_f_helper2', borrow=True)
        self.U_c_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_c_helper2', borrow=True)
        self.U_o_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_o_helper2', borrow=True)
        self.V_o_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='V_o_helper2', borrow=True)
        self.b_i_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_i_helper2', borrow=True)
        self.b_f_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_f_helper2', borrow=True)
        self.b_c_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_c_helper2', borrow=True)
        self.b_o_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_o_helper2', borrow=True)
                                                          
        # parameters of this layer
        self.params = [self.W_i, self.W_f, self.W_c, self.W_o, self.U_i, self.U_f, self.U_c, self.U_o, self.V_o, self.b_i, self.b_f, self.b_c, self.b_o]
        self.params_helper = [self.W_i_helper, self.W_f_helper, self.W_c_helper, self.W_o_helper, self.U_i_helper, self.U_f_helper, self.U_c_helper, self.U_o_helper, self.V_o_helper, self.b_i_helper, self.b_f_helper, self.b_c_helper, self.b_o_helper]
        self.params_helper2 = [self.W_i_helper2, self.W_f_helper2, self.W_c_helper2, self.W_o_helper2, self.U_i_helper2, self.U_f_helper2, self.U_c_helper2, self.U_o_helper2, self.V_o_helper2, self.b_i_helper2, self.b_f_helper2, self.b_c_helper2, self.b_o_helper2]
                                                          
        # initial hidden state values
        h_0 = T.zeros((n_out,))
        ##intialize memory cell (c) values with zeros?? ##
        c_0 = T.zeros((n_out,))
                                                          
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1, c_tm1):
            input_gate = T.nnet.sigmoid(T.dot(u_t,self.W_i)+T.dot(h_tm1,self.U_i) + self.b_i)
            forget_gate = T.nnet.sigmoid(T.dot(u_t,self.W_f)+T.dot(h_tm1,self.U_f)+self.b_f)
            c_candidate = T.tanh(T.dot(u_t,self.W_c)+T.dot(h_tm1,U_c)+self.b_c)
            c_t = c_candidate*input_gate + c_tm1*forget_gate
            output_gate = T.nnet.sigmoid(T.dot(u_t,self.W_o)+T.dot(h_tm1,self.U_o)+ T.dot(c_t,self.V_o) + b_o)
            h_t = T.tanh(c_t)*output_gate
            return h_t, c_t
                                                                                  
        # compute timeseries
        [h, c], _ = theano.scan(step,
                            sequences=self.input,
                            outputs_info=[h_0, c_0],
                            truncate_gradient=-1)
                                                                                      
       # output activity
        self.output = h

    def load_params(in_params):
        self.W_i = in_params[0]
        self.W_f = in_params[1]
        self.W_c = in_params[2]
        self.W_o = in_params[3]
        self.U_i = in_params[4]
        self.U_f = in_params[5]
        self.U_c = in_params[6]
        self.U_o = in_params[7]
        self.V_o = in_params[8]
        self.b_i = in_params[9]
        self.b_f = in_params[10]
        self.b_c = in_params[11]
        self.b_o = in_params[12] 
                

class Dropout(object):
    def __init__(self, rng, is_train, input, p=0.5):
        """
        Layer to perform dropout
        """

        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        def drop(input, p=0.5, rng=rng): 
            """
            :type input: numpy.array
            :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
            :type p: float or double between 0. and 1. 
            :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
            """            
            mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
            return input * mask

        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(numpy.cast[theano.config.floatX](1./p) * input)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, input)
        


#This variant is the standard multiple input, multiple output version
class LinearRegression(object):
    """Linear Regression Class
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the poisson regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # compute vector of expected values (for each output) in symbolic form
        self.E_y_given_x = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]
    
    def load_params(in_params):
        self.W = in_params[13]
        self.b = in_params[14]
    
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        return T.mean(   (y - self.E_y_given_x)**2   , axis = 0)





# This class is to build a RNN 
class RNN(object):
    def __init__(self, rng, input, n_in, n_out):
        """
            RNN hidden layer: units are fully-connected and have
            an activation function (see below). Weights project inputs to the units which are recurrently connected.
            Weight matrix W is of shape (n_in,n_out)
            and the bias vector b is of shape (n_out,).
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)
            
            :type n_in: int
            :param n_in: dimensionality of input
            
            """
        self.input = input
        
        def get_orthogonal_vals(M,N):
            Q = numpy.random.randn(M, N).astype(theano.config.floatX)
            u, s, v = numpy.linalg.svd(Q)
            if M>N:
                return u[:,0:N]
            else:
                return v[0:M,:]

        W_values = get_orthogonal_vals(n_in, n_out)
        #W_values = .01*numpy.random.randn(n_in, n_out).astype(theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 0*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        Q = numpy.random.randn(n_out, n_out).astype(theano.config.floatX)
        W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        self.W = W
        self.b = b
        self.W_RNN = W_RNN


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.W_RNN_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.W_RNN_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W_RNN, self.W, self.b]
        self.params_helper = [self.W_RNN_helper, self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_RNN_helper2, self.W_helper2, self.b_helper2]

        #initial hidden state values
        h_0 = T.zeros((n_out,))
  
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1):
            lin_E = T.dot(u_t, self.W) - T.dot(h_tm1, self.W_RNN) + self.b
            #h_t = lin_E*(lin_E>0)
            h_t = T.tanh(lin_E)
            return h_t

        # compute the hidden E & I timeseries
        h, _ = theano.scan(step,
                   sequences=self.input,
                   outputs_info=h_0,
                   truncate_gradient=-1)
            
        # output activity is the hidden unit activity
        self.output = h




class hybridRNN(object):
    def __init__(self, rng, input, n_in, n_lstm, n_rnn):
        
        self.input = input

        def get_orthogonal_vals(M,N):
            Q = numpy.random.randn(M, N).astype(theano.config.floatX)
            u, s, v = numpy.linalg.svd(Q)
            if M>N:
                return u[:,0:N]
            else:
                return v[0:M,:]

        w_mag = .01

        #RNN params
        W_values = get_orthogonal_vals(n_in, n_rnn)
        #W_values = w_mag*numpy.random.randn(n_in, n_rnn).astype(theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 0*numpy.ones((n_rnn,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        Q = numpy.random.randn(n_rnn, n_rnn).astype(theano.config.floatX)
        W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        W_LSTM_values = get_orthogonal_vals(n_lstm, n_rnn)
        W_LSTM = theano.shared(value=W_LSTM_values, name='W_LSTM', borrow=True)

        ##  Create parameters ##
        # initializations?? # implement orthogonal initialization
        # Weights from input to gates
        W_i_values = get_orthogonal_vals(n_in, n_lstm)
        #W_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_lstm)), dtype=theano.config.floatX) #weights from input to input gate
        W_i = theano.shared(value=W_i_values, name='W_i', borrow=True)
        W_f_values = get_orthogonal_vals(n_in, n_lstm)
        #W_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_lstm)), dtype=theano.config.floatX) #weights from input to forget gate
        W_f = theano.shared(value=W_f_values, name='W_f', borrow=True)
        W_c_values = get_orthogonal_vals(n_in, n_lstm)
        #W_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_lstm)), dtype=theano.config.floatX) #weights from input to memory cells directly
        W_c = theano.shared(value=W_c_values, name='W_c', borrow=True)
        W_o_values = get_orthogonal_vals(n_in, n_lstm)
        #W_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_lstm)), dtype=theano.config.floatX) #weights from input to output gate
        W_o = theano.shared(value=W_o_values, name='W_o', borrow=True)
        
        # Weights from previous LSTM units to gates
        U_i_values = get_orthogonal_vals(n_lstm, n_lstm)
        #U_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_lstm, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to input gate
        U_i = theano.shared(value=U_i_values, name='U_i', borrow=True)
        U_f_values = get_orthogonal_vals(n_lstm, n_lstm)
        #U_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_lstm, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to forget gate
        U_f = theano.shared(value=U_f_values, name='U_f', borrow=True)
        U_c_values = get_orthogonal_vals(n_lstm, n_lstm)
        #U_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_lstm, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to memory cell
        U_c = theano.shared(value=U_c_values, name='U_c', borrow=True)
        U_o_values = get_orthogonal_vals(n_lstm, n_lstm)
        #U_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_lstm, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to output gate
        U_o = theano.shared(value=U_o_values, name='U_o', borrow=True)

        # Weights from previous RNN units to gates
        R_i_values = get_orthogonal_vals(n_rnn, n_lstm)
        #R_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_rnn, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to input gate
        R_i = theano.shared(value=R_i_values, name='R_i', borrow=True)
        R_f_values = get_orthogonal_vals(n_rnn, n_lstm)
        #R_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_rnn, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to forget gate
        R_f = theano.shared(value=R_f_values, name='R_f', borrow=True)
        R_c_values = get_orthogonal_vals(n_rnn, n_lstm)
        #R_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_rnn, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to memory cell
        R_c = theano.shared(value=R_c_values, name='R_c', borrow=True)
        R_o_values = get_orthogonal_vals(n_rnn, n_lstm)
        #R_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_rnn, n_lstm)), dtype=theano.config.floatX) #weights from last outputs to output gate
        R_o = theano.shared(value=R_o_values, name='R_o', borrow=True)
        
        V_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_lstm, n_lstm)), dtype=theano.config.floatX) #weights from memory cell state to output gate
        V_o = theano.shared(value=V_o_values, name='V_o', borrow=True)
        
        # Biases of gates
        b_i_values = numpy.zeros((n_lstm,), dtype=theano.config.floatX)
        b_i = theano.shared(value=b_i_values, name='b_i', borrow=True)
        b_f_values = numpy.zeros((n_lstm,), dtype=theano.config.floatX)
        b_f = theano.shared(value=b_f_values, name='b_f', borrow=True)
        b_c_values = numpy.zeros((n_lstm,), dtype=theano.config.floatX)
        b_c = theano.shared(value=b_c_values, name='b_c', borrow=True)
        b_o_values = numpy.zeros((n_lstm,), dtype=theano.config.floatX)
        b_o = theano.shared(value=b_o_values, name='b_o', borrow=True)
        
        
        # Assign parameters to self
        self.W = W
        self.b = b
        self.W_RNN = W_RNN
        self.W_LSTM = W_LSTM
        self.W_i = W_i
        self.W_f = W_f
        self.W_c = W_c
        self.W_o = W_o
        self.U_i = U_i
        self.U_f = U_f
        self.U_c = U_c
        self.U_o = U_o
        self.R_i = R_i
        self.R_f = R_f
        self.R_c = R_c
        self.R_o = R_o
        self.V_o = V_o
        self.b_i = b_i
        self.b_f = b_f
        self.b_c = b_c
        self.b_o = b_o
        
        # Helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_rnn), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.W_RNN_helper = theano.shared(value=numpy.zeros((n_rnn, n_rnn), \
            dtype=theano.config.floatX), name='W_RNN_helper', borrow=True)
        self.W_LSTM_helper = theano.shared(value=numpy.zeros((n_lstm, n_rnn), \
            dtype=theano.config.floatX), name='W_LSTM_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_rnn,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        self.W_i_helper = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_i_helper', borrow=True)
        self.W_f_helper = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_f_helper', borrow=True)
        self.W_c_helper = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_c_helper', borrow=True)
        self.W_o_helper = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_o_helper', borrow=True)
        self.U_i_helper = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_i_helper', borrow=True)
        self.U_f_helper = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_f_helper', borrow=True)
        self.U_c_helper = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_c_helper', borrow=True)
        self.U_o_helper = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_o_helper', borrow=True)
        self.R_i_helper = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_i_helper', borrow=True)
        self.R_f_helper = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_f_helper', borrow=True)
        self.R_c_helper = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_c_helper', borrow=True)
        self.R_o_helper = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_o_helper', borrow=True)
        self.V_o_helper = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='V_o_helper', borrow=True)
        self.b_i_helper = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_i_helper', borrow=True)
        self.b_f_helper = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_f_helper', borrow=True)
        self.b_c_helper = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_c_helper', borrow=True)
        self.b_o_helper = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_o_helper', borrow=True)
                                                          
        # Helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_rnn), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.W_RNN_helper2 = theano.shared(value=numpy.zeros((n_rnn, n_rnn), \
            dtype=theano.config.floatX), name='W_RNN_helper2', borrow=True)
        self.W_LSTM_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_rnn), \
            dtype=theano.config.floatX), name='W_LSTM_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_rnn,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        self.W_i_helper2 = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_i_helper2', borrow=True)
        self.W_f_helper2 = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_f_helper2', borrow=True)
        self.W_c_helper2 = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_c_helper2', borrow=True)
        self.W_o_helper2 = theano.shared(value=numpy.zeros((n_in, n_lstm), \
            dtype=theano.config.floatX), name='W_o_helper2', borrow=True)
        self.U_i_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_i_helper2', borrow=True)
        self.U_f_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_f_helper2', borrow=True)
        self.U_c_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_c_helper2', borrow=True)
        self.U_o_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='U_o_helper2', borrow=True)
        self.R_i_helper2 = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_i_helper2', borrow=True)
        self.R_f_helper2 = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_f_helper2', borrow=True)
        self.R_c_helper2 = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_c_helper2', borrow=True)
        self.R_o_helper2 = theano.shared(value=numpy.zeros((n_rnn, n_lstm), \
            dtype=theano.config.floatX), name='R_o_helper2', borrow=True)
        self.V_o_helper2 = theano.shared(value=numpy.zeros((n_lstm, n_lstm), \
            dtype=theano.config.floatX), name='V_o_helper2', borrow=True)
        self.b_i_helper2 = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_i_helper2', borrow=True)
        self.b_f_helper2 = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_f_helper2', borrow=True)
        self.b_c_helper2 = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_c_helper2', borrow=True)
        self.b_o_helper2 = theano.shared(value=numpy.zeros((n_lstm,), \
            dtype=theano.config.floatX), name='b_o_helper2', borrow=True)
                                                          
        # parameters of this layer
        self.params = [self.W_RNN, self.W_LSTM, self.W, self.b, self.W_i, self.W_f, self.W_c, self.W_o, self.U_i, self.U_f, self.U_c, self.U_o, self.R_i, self.R_f, self.R_c, self.R_o, self.V_o, self.b_i, self.b_f, self.b_c, self.b_o]
        self.params_helper = [self.W_RNN_helper, self.W_LSTM_helper, self.W_helper, self.b_helper, self.W_i_helper, self.W_f_helper, self.W_c_helper, self.W_o_helper, self.U_i_helper, self.U_f_helper, self.U_c_helper, self.U_o_helper, self.R_i_helper, self.R_f_helper, self.R_c_helper, self.R_o_helper, self.V_o_helper, self.b_i_helper, self.b_f_helper, self.b_c_helper, self.b_o_helper]
        self.params_helper2 = [self.W_RNN_helper2, self.W_LSTM_helper2, self.W_helper2, self.b_helper2, self.W_i_helper2, self.W_f_helper2, self.W_c_helper2, self.W_o_helper2, self.U_i_helper2, self.U_f_helper2, self.U_c_helper2, self.U_o_helper2, self.R_i_helper2, self.R_f_helper2, self.R_c_helper2, self.R_o_helper2, self.V_o_helper2, self.b_i_helper2, self.b_f_helper2, self.b_c_helper2, self.b_o_helper2]
                                                          
        # initial hidden state values
        h_l_0 = T.zeros((n_lstm,)) #lstm units are h_l
        h_r_0 = T.zeros((n_rnn,)) #rnn units are h_r
        ##intialize memory cell (c) values with zeros?? ##
        c_0 = T.zeros((n_lstm,))

        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_l_tm1, h_r_tm1, c_tm1):
            input_gate = T.nnet.sigmoid(T.dot(u_t,self.W_i) + T.dot(h_l_tm1,self.U_i) + T.dot(h_r_tm1,self.R_i) + self.b_i)
            forget_gate = T.nnet.sigmoid(T.dot(u_t,self.W_f)+T.dot(h_l_tm1,self.U_f) + T.dot(h_r_tm1,self.R_f) + self.b_f)
            c_candidate = T.tanh(T.dot(u_t,self.W_c)+T.dot(h_l_tm1,U_c) + T.dot(h_r_tm1,self.R_c) + self.b_c)
            c_t = c_candidate*input_gate + c_tm1*forget_gate
            output_gate = T.nnet.sigmoid(T.dot(u_t,self.W_o)+T.dot(h_l_tm1,self.U_o)+ T.dot(h_r_tm1,self.R_o) + T.dot(c_t,self.V_o) + b_o)
            h_l_t = T.tanh(c_t)*output_gate
            h_r_t = T.tanh( T.dot(u_t, self.W) + T.dot(h_r_tm1, self.W_RNN) + T.dot(h_l_tm1, self.W_LSTM) + self.b )
            return h_l_t, h_r_t, c_t
                           
        # compute timeseries
        [h_l, h_r, c], _ = theano.scan(step,
                            sequences=self.input,
                            outputs_info=[h_l_0, h_r_0, c_0],
                            truncate_gradient=-1)
                                                                                      
       # output activity
        self.output = T.concatenate([h_l, h_r], axis=1)




# This class is to build a RNN 
class IRNN(object):
    def __init__(self, rng, input, n_in, n_out):
        """
            RNN hidden layer: units are fully-connected and have
            an activation function (see below). Weights project inputs to the units which are recurrently connected.

            Weight matrix W is of shape (n_in,n_out)

            and the bias vector b is of shape (n_out,).
            

            :type rng: numpy.random.RandomState

            :param rng: a random number generator used to initialize weights
            

            :type input: theano.tensor.dmatrix

            :param input: a symbolic tensor of shape (n_examples, n_in)
            

            :type n_in: int

            :param n_in: dimensionality of input
            

            """
        self.input = input
        
        def get_orthogonal_vals(M,N):
            Q = numpy.random.randn(M, N).astype(theano.config.floatX)
            u, s, v = numpy.linalg.svd(Q)
            if M>N:
                return u[:,0:N]
            else:
                return v[0:M,:]

        W_values = get_orthogonal_vals(n_in, n_out)
        #W_values = .01*numpy.random.randn(n_in, n_out).astype(theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 0*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        #Q = numpy.random.randn(n_out, n_out).astype(theano.config.floatX)
        #W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN_values = numpy.identity(n_out, dtype=theano.config.floatX)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        self.W = W
        self.b = b
        self.W_RNN = W_RNN


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.W_RNN_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.W_RNN_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W_RNN, self.W, self.b]
        self.params_helper = [self.W_RNN_helper, self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_RNN_helper2, self.W_helper2, self.b_helper2]

        #initial hidden state values
        h_0 = T.zeros((n_out,))
  
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1):
            lin_E = T.dot(u_t, self.W) + T.dot(h_tm1, self.W_RNN) + self.b
            h_t = T.minimum(lin_E*(lin_E>0),100)
            #h_t = T.tanh(lin_E)
            return h_t

        # compute the hidden E & I timeseries
        h, _ = theano.scan(step,
                   sequences=self.input,
                   outputs_info=h_0,
                   truncate_gradient=-1)
            
        # output activity is the hidden unit activity
        self.output = h

