
import numpy as np 
import tensorflow as tf 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 


class DenseLayer(object):
    
    def __init__(self, M1, M2, activation = tf.nn.relu, name = ''):
        
        self.W = tf.Variable(tf.random.normal(shape=(M1, M2)) * np.sqrt(2.0 / M2), name = 'W_%s' % name)
        self.bf  = tf.Variable(initial_value = tf.zeros(shape = M2), name = "bf_%s" %name)
        self.f = activation
        
        
        self.bb  = tf.Variable(initial_value = tf.zeros(shape = M1), name = "bb_%s" %name)
        
        self.params = [self.W, self.bf, self.bb]
    def forward(self, X):
        Z = tf.matmul(X, self.W) + self.bf
        return self.f(Z)
    
    def forwardT(self, Z):
        Z = tf.matmul(Z, tf.transpose(self.W)) + self.bb
        return self.f(Z)
    
    
class Autoencoder(object):
    
    def __init__(self, dims, layers, name = '0'):
        
        self.D = dims 
        self.layerDic= layers
        self.name = name 
        
        self.build()
        
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.1)#SGD(learning_rate=0.01)#tf.keras.optimizers.Adam()


    def build(self):
        
        self.layers = []
        
        M1 = self.D
        #Dense layers
        count = 0
        for M2, f in self.layerDic['DenseLayers']:
            layer = DenseLayer(M1, M2, activation=f, name = 'dense %i' % count) 
            self.layers.append(layer)
            M1 = M2
            count += 1
        
        self.trainable_params = []
        
        # define different baises for the decoder
        for layer in self.layers:
            self.trainable_params += layer.params
            
        
    def encode(self, X):
        
        Z = X
        for layer in self.layers:
            Z = layer.forward(X)
            
        return Z
    
    def decode(self, Z):
        for i in range(len(self.layers)-1, -1, -1):
            Z = self.layers[i].forwardT(Z)
            
        return Z
    
    def forward(self, X):
        Z = self.encode(X)
        
        return self.decode(Z)
    
    def cost(self, X):
        
        Xhat = self.forward(X)
        N =len(Xhat)
        cost = 0
        for i in range(len(Xhat)):
            cost += tf.reduce_sum((X[i] - Xhat[i])**2)
            
        return cost/N
    
    def train(self, X):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            cost_i = self.cost(X)
            
        gradients = tape.gradient(cost_i, self.trainable_params)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_params))
   
        return cost_i
    
    def fit(self, X, learning_rate=0.01, epochs=50, batch_sz=100, show_fig=False):
        
        # self.optimizer(lr = learning_rate, beta_1=0.99, beta_2=0.999)
        
        
        n_batches = len(X) // batch_sz
        costVec = []
        
        for epoch in range(epochs):
            
            X = shuffle(X)
            
            for j  in range(n_batches):
                Xbatch = X[batch_sz * j : (j+1) *batch_sz]
                
                cost_i = self.train(Xbatch)
                costVec.append(cost_i.numpy())
                
            
                if j % (n_batches//2) == 0:
                    print("epoch: ", epoch, "j: ", j, "cost: ", costVec[-1])
                    
        
        # plt.plot(costVec)
        # plt.show()         
    
            
            
            
            
    
            
        