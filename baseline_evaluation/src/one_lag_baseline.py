import h5py
import theano
from theano import shared
import theano.tensor as T
import numpy 

def load_class_data_batch(dataset):
    print '...loading data'
    f = h5py.File(dataset)
    stimuli = numpy.transpose(f['stimulus_zscore'])
    f.close()

    rval = [numpy.asarray(stimuli, dtype=theano.config.floatX)]
    print '...done loading..'
    return rval


def main():

    #loading in data set
    dataset_for_error = '/vega/stats/users/sl3368/Data_LC/NormData/LC_stim_15.mat'
    stimuli = load_class_data_batch(dataset_for_error)
    stim = stimuli[0]
    data = theano.shared( stim, borrow=True)
    print 'Number of rows: '
    print stim.shape[0]

    #setting variable for error 
    init = numpy.float64(0.0)
    mean_error = shared(init)

    #writing theano functions for computing mean square error for one lag 
    
    prediction = T.fvector('predict') # 60 row vector representing time t

    real = T.fvector('real') #row representing time t+1 

    cost = T.mean( (real - prediction) ** 2)

    #function for updating mean error
    batch_error = theano.function([prediction,real],cost,updates=[(mean_error, mean_error + cost)])


    increment = stim.shape[0]/100
    #iterating over batch and computing the error
    for index in range(stim.shape[0]-1):
        if index % increment == 0:
		print str(index/increment)+'% done...'
	recent = batch_error(stim[index],stim[index+1])

    #m_e_avg = mean_error / 9000000

    #printing result
    print 'Total error: '
    print mean_error.get_value()

    print 'Finding padding amount...'
    num_zero = float(0.0)
    #calculating zeros amount
    for index in range(stim.shape[0]):
        is_zero = True
        for i in range(60):
            if stim[index][i] != 0:
               is_zero = False
   
        if is_zero:
            num_zero = num_zero + 1

    print 'Percent Zero: '+str(float(num_zero/(increment * 100))) 

if __name__== '__main__':
    main()
