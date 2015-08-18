

################################################
# Import statements
################################################

import cPickle
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from loading_functions import load_all_data, load_class_data_batch
from layer_classes import LinearRegression, Dropout, LSTM, RNN, hybridRNN
from one_ahead import GradClip, clip_gradient
from random import randint


################################################
# Variables to be set
################################################

song_size = 2000

num_songs_in_batch = 3000

savefilename_pre = '/vega/stats/users/sl3368/rnn_code/results/lstm/3_layer/300/generated/'

params_file = '/vega/stats/users/sl3368/rnn_code/saves/params/lstm/3_layer/300/recent_2nd.save'

test_cost = .05 #general validation number

################################################
# Loading data
################################################


dataset_info = load_class_data_batch('/vega/stats/users/sl3368/Data_LC/LowNormData/LC_stim_5.mat')
stim = dataset_info[0]
data_set_x = theano.shared(stim, borrow=True)


###############################################################
# (Re-Define) Architecture: input --> LSTM --> predict one-ahead
###############################################################

x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
x = clip_gradient(x,1.0)    
is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

rng = numpy.random.RandomState(1234)

# The poisson regression layer gets as input the hidden units
# of the hidden layer
n_hidden = 300;

lstm_1 = LSTM(rng, x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=n_hidden)

lstm_2 = LSTM(rng, lstm_1.output, n_in=n_hidden, n_out=n_hidden-100)

lstm_3 = LSTM(rng, lstm_2.output, n_in=n_hidden-100, n_out=n_hidden-200)

output = LinearRegression(input=lstm_3.output, n_in=n_hidden-200, n_out=data_set_x.get_value(borrow=True).shape[1])

################################################
# Load learned params
################################################


f = file(params_file, 'rb')
old_p = cPickle.load(f)
f.close()

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
lstm_2.W_i.set_value(old_p[13].get_value(), borrow=True)
lstm_2.W_f.set_value(old_p[14].get_value(), borrow=True)
lstm_2.W_c.set_value(old_p[15].get_value(), borrow=True)
lstm_2.W_o.set_value(old_p[16].get_value(), borrow=True)
lstm_2.U_i.set_value(old_p[17].get_value(), borrow=True)
lstm_2.U_f.set_value(old_p[18].get_value(), borrow=True)
lstm_2.U_c.set_value(old_p[19].get_value(), borrow=True)
lstm_2.U_o.set_value(old_p[20].get_value(), borrow=True)
lstm_2.V_o.set_value(old_p[21].get_value(), borrow=True)
lstm_2.b_i.set_value(old_p[22].get_value(), borrow=True)
lstm_2.b_f.set_value(old_p[23].get_value(), borrow=True)
lstm_2.b_c.set_value(old_p[24].get_value(), borrow=True)
lstm_2.b_o.set_value(old_p[25].get_value(), borrow=True)
lstm_3.W_i.set_value(old_p[26].get_value(), borrow=True)
lstm_3.W_f.set_value(old_p[27].get_value(), borrow=True)
lstm_3.W_c.set_value(old_p[28].get_value(), borrow=True)
lstm_3.W_o.set_value(old_p[29].get_value(), borrow=True)
lstm_3.U_i.set_value(old_p[30].get_value(), borrow=True)
lstm_3.U_f.set_value(old_p[31].get_value(), borrow=True)
lstm_3.U_c.set_value(old_p[32].get_value(), borrow=True)
lstm_3.U_o.set_value(old_p[33].get_value(), borrow=True)
lstm_3.V_o.set_value(old_p[34].get_value(), borrow=True)
lstm_3.b_i.set_value(old_p[35].get_value(), borrow=True)
lstm_3.b_f.set_value(old_p[36].get_value(), borrow=True)
lstm_3.b_c.set_value(old_p[37].get_value(), borrow=True)
lstm_3.b_o.set_value(old_p[38].get_value(), borrow=True)
output.W.set_value(old_p[39].get_value(), borrow=True)
output.b.set_value(old_p[40].get_value(), borrow=True)



################################################
# Generate 1 ahead function
################################################
previous_samples = T.matrix()

ygen_model = theano.function(inputs=[previous_samples],
        outputs=output.E_y_given_x,
        givens={x: previous_samples})


################################################
# Generate the sequence
################################################
print 'Generating....'

index = randint(0,num_songs_in_batch)

print 'Song: '+str(index)

y_gen = data_set_x[index * song_size:((index + 1) * song_size - 1)].eval()
y_gen_truth = data_set_x[index * song_size:((index + 1) * song_size - 1)].eval()
for i in xrange(1001,song_size-1):
    y_pred = ygen_model(y_gen)
    y_gen[i] = numpy.random.normal(y_pred[i-1],
                            numpy.sqrt(test_cost))

print 'Finished Generating...'

################################################
# Plot it
################################################

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(numpy.transpose(y_gen2))
#plt.axvline(x=1000, ymin=0, ymax = 60, linewidth=3, color='r')
#ax.set_aspect(20)
#plt.show()

################################################
# Store it
################################################
print 'Saving...'

savefilename = savefilename_pre + str(index)+'_batch_5.gen'

f = file(savefilename, 'wb')
for obj in [[y_gen_truth] + [y_gen]]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()


