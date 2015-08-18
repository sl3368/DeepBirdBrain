import h5py
import theano
from theano.tensor.extra_ops import repeat
import sklearn
from sklearn.preprocessing import OneHotEncoder
import numpy
import scipy
from scipy.io import wavfile

#get train and test wave form
def get_timit_waveform():
    
    #load training data wavefiles
    train_filenames = []
    with open('train.list') as f:
        for line in f:
            train_filenames.append(line.rstrip('\n'))

    file_pre = '/vega/stats/users/sl3368/TIMIT_process/TimitWav/train/'
    train_audio = []
    for filename in train_filenames:
        f,w = wavfile.read(file_pre+filename)
        train_audio.append(w)


    #load training data phoneme labels 
    phn = h5py.File('TIMIT_TRAIN.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    train_phn = []
    for i in range(len(train_audio)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))
	
	
	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 1600 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,160,axis=0).eval()

	train_phn.append(rep_enc_phonemes)

    print 'training done...'
    #load test data wavefiles
    test_filenames = []
    with open('test.list') as f:
        for line in f:
            test_filenames.append(line.rstrip('\n'))

    file_pre = '/vega/stats/users/sl3368/TIMIT_process/TimitWav/test/'
    test_audio = []
    for filename in test_filenames:
        f,w = wavfile.read(file_pre+filename)
        test_audio.append(w)


    #load testing data phoneme labels 
    phn = h5py.File('TIMIT_TEST.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    test_phn = []
    for i in range(len(test_audio)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))

	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 16000 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,160,axis=0).eval()

	test_phn.append(rep_enc_phonemes)

    return train_audio,train_phn,test_audio,test_phn


#get train and test spectrogram
def get_timit_specs():
    
    #get training spectrograms
    f = h5py.File('/vega/stats/users/sl3368/Data_LC/timit/train/timit_stim_1.mat')
    train_stim = numpy.transpose(f['stimulus_zscore'])
    
    train_filenames = []
    with open('train.list') as f:
        for line in f:
            train_filenames.append(line.rstrip('\n'))

    #load training data phoneme labels 
    phn = h5py.File('TIMIT_TRAIN.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    train_phn = []
    for i in range(len(train_filenames)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))

	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 10 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,10,axis=0).eval()

	train_phn.append(rep_enc_phonemes)

    
    #get testing spectrograms
    f = h5py.File('/vega/stats/users/sl3368/Data_LC/timit/test/timit_stim_1.mat')
    test_stim = numpy.transpose(f['stimulus_zscore'])
    
    #load test data wavefiles
    test_filenames = []
    with open('test.list') as f:
        for line in f:
            test_filenames.append(line.rstrip('\n'))

    #load testing data phoneme labels 
    phn = h5py.File('TIMIT_TEST.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    test_phn = []
    for i in range(len(test_filenames)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))

	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 10 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,10,axis=0).eval()

	test_phn.append(rep_enc_phonemes)

    return train_stim,train_phn,test_stim,test_phn



#get train and test spectrogram with images
def get_timit_specs_images(window_size):

    #get training spectrograms
    f = h5py.File('/vega/stats/users/sl3368/Data_LC/timit/train/timit_stim_1.mat')
    train_stim = numpy.transpose(f['stimulus_zscore'])
   
    #need to construct windows
    train_stim_windows = numpy.zeros((train_stim.shape[0]/5000,5000-window_size,window_size,60))
    half = window_size/2
    for j in range(len(train_stim)/5000):
        for i in range(j*5000,(j+1)*5000-window_size):
            temp_window = train_stim[i:i+window_size]
            train_stim_windows[j][i] = temp_window
            #single_window = numpy.reshape(temp_window,(1,window_size*train_stim.shape[1]))
 
    train_filenames = []
    with open('train.list') as f:
        for line in f:
            train_filenames.append(line.rstrip('\n'))

    #load training data phoneme labels 
    phn = h5py.File('TIMIT_TRAIN.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    train_phn = []
    for i in range(len(train_filenames)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))

	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 10 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,10,axis=0).eval()

	train_phn.append(rep_enc_phonemes)

    train_phn = train_phn[half:len(train_phn)-half]
    
    #get testing spectrograms
    f = h5py.File('/vega/stats/users/sl3368/Data_LC/timit/test/timit_stim_1.mat')
    test_stim = numpy.transpose(f['stimulus_zscore'])
    
    #need to construct windows
    test_stim_windows = numpy.zeros((test_stim.shape[0]/5000,5000-window_size,window_size,60))
    half = window_size/2
    for j in range(len(test_stim)/5000):
        for i in range(j*5000,(j+1)*5000-window_size):
            temp_window = test_stim[i:i+window_size]
            test_stim_windows[j][i] = temp_window
            #single_window = numpy.reshape(temp_window,(1,window_size*train_stim.shape[1]))

    #load test data wavefiles
    test_filenames = []
    with open('test.list') as f:
        for line in f:
            test_filenames.append(line.rstrip('\n'))

    #load testing data phoneme labels 
    phn = h5py.File('TIMIT_TEST.mat')
    phn_data = phn['data']

    #initializing encoder
    enc = OneHotEncoder(n_values=41,dtype=numpy.int16,sparse=False)

    test_phn = []
    for i in range(len(test_filenames)):
        ref = phn_data[i][0]
        labels = phn[ref]
	phonemes = labels[2]
	phonemes = numpy.reshape(phonemes,(len(phonemes),1))

	#need to encode and repeat for each sample
	encoded_phonemes = enc.fit_transform(phonemes)

        #repeat for the sampling rate 10 this case!!
        rep_enc_phonemes = repeat(encoded_phonemes,10,axis=0).eval()

	test_phn.append(rep_enc_phonemes)


    test_phn = test_phn[half:len(test_phn)-half]

    return train_stim_windows,train_phn,test_stim_windows,test_phn
