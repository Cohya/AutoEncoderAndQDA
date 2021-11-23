
import numpy as np 
from Autoencoder import Autoencoder
from QDA import QDA 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import tensorflow_probability as tfp
import tensorflow as tf 

### Load the data 

# X, Y = load_data()

# for each label we will train a autoencoder (very small one- only one hidden layer)

### change the data into float32
Y = np.ones(shape = 5000).astype('int32')
X = np.random.multivariate_normal(mean=[1,2,3,1,2,3,4,3,2,4], cov = np.eye(10), size = 5000)

# Normelize X to -1 to 1 
mean_X = np.min(X, axis = 0)
max_X = np.max(X, axis = 0)
mix_X = np.min(X, axis = 0)
X = (X - mean_X)/( max_X - mix_X ) *2 - 1

X = X.astype('float32')
### Make sure that your data is 
all_autoencoders = []
all_gaussian_distributions = []

for c in np.unique(Y): # c is the unique label

    modul_i_data = X[Y == c]
    
    # Normalization 
    mean_X = np.min(X, axis = 0)
    max_X = np.max(X, axis = 0)
    mix_X = np.min(X, axis = 0)
    X = (X - mean_X)/( max_X - mix_X ) *2 - 1  
    
    
    # creat an AutoEncoder for this label with 6 hidden layer 
    N, dims = np.shape(X)                               # number of samples, dimensions
                                             
    layers = {'DenseLayers': [[2, tf.nn.tanh]]}                        # number of hidden layer at each layer (here we have only one layer)
    
    autoencoder = Autoencoder(dims, layers, name = str(c))   # define an autoencoder class
    
    autoencoder.fit(X, epochs= 1, batch_sz=100)          # training step X belongs to R^(NxD) where N is the 
                                                        # number of samples in class c
   
    all_autoencoders.append(autoencoder)
    
    
    c_features = autoencoder.encode(X)               # size c_features belongs to R(NxM)
    c_features = c_features.numpy()
    
    N, dims = c_features.shape
    mean_c_features = np.matmul(np.ones(shape = (1,N)) * 1/N, c_features).reshape(dims) 
    
    # Calculate the covariance matrix 
    I1 = np.ones(shape = (N,N))
    I = np.eye(N)
    cov_c_features = 1/N * np.matmul(
                                        np.matmul(c_features.T,(I - 1/N * I1)),
                                        c_features)
    
    # define a multivariate random variable 
    rv = multivariate_normal(mean = mean_c_features,
                             cov = cov_c_features)
    
    all_gaussian_distributions.append(rv)
    
    
    
    

Xtrue = np.random.multivariate_normal(mean=[1,2,3,1,2,3,4,3,2,4], cov = np.eye(10), size = 10)
Xfalse = np.random.multivariate_normal(mean=[1,0,2,5,12,3,3,1,2,4], cov = np.eye(10), size = 50)
Ytrue = np.concatenate((
                        tf.ones(shape = 10), 
                        tf.zeros(shape = 50)))

Xtest = np.concatenate((Xtrue, Xfalse))


#normelizition step 
Xtest = (Xtest - mean_X)/( max_X - mix_X ) *2 - 1
Xtest = Xtest.astype('float32')
# results = []
# values = []
# for x in Xtest:
    
#     x = np.expand_dims(x, axis = 0)
#     x_encode = autoencoder.encode(x.astype('float32'))
#     val = rv.cdf(x_encode)
    
#     if val < 0.99 and val > 0.01:
#         results.append(1)
#     else:
#         results.append(0)
        
#     values.append(val)

# print("acc: ", 1 - sum(abs(results - Ytrue))/len(Ytrue))
    
    
# plt.figure(2)
# Xencode = autoencoder.encode(Xtest).numpy()
# plt.scatter(x = Xencode[:,0], y =Xencode[:,1], c= Ytrue, s=10, alpha=1.5)

qda = QDA(all_gaussian_distributions, all_autoencoders)
results, values= qda.checkDistance(Xtest)
print("acc: ", 1 - sum(abs(results[0] - Ytrue))/len(Ytrue))


    
    
    
    
        
    
        
    
    
    