
import numpy as np 
from Autoencoder import Autoencoder
from QDA import QDA 

### Load the data 

# X, Y = load_data()

# for each label we will train a autoencoder (very small one- only one hidden layer)

### change the data into float32
Y =np.array([0,1,2]).astype('int32')
X = np.array([1.,2.,3])
X = X.astype('float32')

all_autoencoder = []
all_gaussian_distributions = []

for c in np.unique(Y): # c is the unique label 
    modul_i_data = X[Y == c]
    
    # creat an AutoEncoder for this label with 6 hidden layer 
    N, dims = np.shape(X) 
                             # dimensions
    M = 6 #                                         # number of hidden layer
    
    autoencoder = Autoencoder(dims, M, name = str(c))   # define an autoencoder class
    
    autoencoder.fit(X, epochs=1, batch_sz=100)     # training step X belongs to R^(NxD) where N is the 
                                                   # number of samples in class c
   
    all_autoencoder.append(autoencoder)
    
    
    c_features = autoencoder.encode(X)               # size c_features belongs to R(NxM)
   
    mean_c_features = np.matmul(np.ones(shape = (1,N)) * 1/N, c_features).reshape(dims) 
    
    cov_c_features = 
   
   
   
###



    
    
    
    
        
    
        
    
    
    