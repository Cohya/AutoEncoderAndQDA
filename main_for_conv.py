
import numpy as np 
from Autoencoder import ConvAutoEncoder
from QDA import QDA 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import tensorflow_probability as tfp
import tensorflow as tf 

input_dims = [32,100,2,1]
X1= np.random.normal(size = (64,100,2,1)).astype(np.float32)    


X2 =    np.random.normal(loc = 2.0, scale = 5.0, size = (64,100,2,1))  
Y = np.concatenate((
                        tf.zeros(shape = 64), 
                        tf.ones(shape = 64)))

X = np.concatenate((X1, X2), axis = 0)
from sklearn.utils import shuffle
X, Y = shuffle(X, Y)

# conv layer: num_of_feature_maps, filtersz , stride , f, add_bias
# $ dense layer: hidden_layer_out, activation=f 
#
layers = {'Conv_layers':[[10, 2, 2, tf.nn.relu , True]], 
              'Dense_layers': [[2, lambda x:x]],
              'last_layer_decoder': [[input_dims[-1], 2, 2, lambda x:x , True]]}                        # number of hidden layer at each layer (here we have only one layer)
    
autoencoder =  ConvAutoEncoder(input_dims, layers, num_of_channels = 1, same_activation_for_decoder_and_encoder = True)
            

autoencoder.fit(X)
X_encode = autoencoder.encode(X).numpy()
X1_encode = X_encode[Y == 0]
X2_encode = X_encode[Y == 1]
plt.scatter(X1_encode[:,0], X1_encode[:,1], c = 'k')
plt.scatter(X2_encode[:,0], X2_encode[:,1], c = 'r')
    
    
    
    
    
# b = autoencoder.encode(X)
# print("b", b.shape)
# bhat = autoencoder.dense_layers[0].forwardT(b)
# print(bhat.shape)
# Z = tf.reshape(bhat, shape = (bhat.shape[0],50,1 , 10))
# print(Z.shape)


# W = autoencoder.deconv_layers[0].W
# # W2 = tf.Variable(initial_value=tf.random.normal(shape = (2,2,1,10)))
# l = tf.nn.conv2d_transpose(input = Z, filters = W, 
#                                               output_shape= [32,100,2,1] ,
#                                               strides=[1, 2, 2, 1])#


# A_hat = autoencoder.deconv_layers[0].forward(Z)