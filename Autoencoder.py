
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
    

class FractionallyStrideConvLayer(object):
    def __init__(self, name, mi, mo, filtersz = 4, stride = 2,f = tf.nn.relu , add_bias = False, output_shape = 1):
        # mi = input feature maps 
        # mo = output feature maps 
        # Note: shape is specified in the opposite wat from regular
        self.add_bias = add_bias
        self.output_shape = output_shape
        # print("sdfsd", self.output_shape)
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz, filtersz, mo, mi], stddev=0.02),
                             name = "W_%s" % name) # look the mo mi is opposite way 
        
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(shape = [mo,]), name = "b_%s" % name)
    
        
        self.name = name 
        self.f = f
        self.stride = stride
        
        
        if self.add_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        
 
    def forward(self, X):

        gh = [len(X)]
        gh += self.output_shape[1:]
        # print("gh: ", gh)
        conv_out = tf.nn.conv2d_transpose(input = X, filters = self.W, 
                                          output_shape=gh ,
                                          strides=[1, self.stride, self.stride, 1])#outpu
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        
        
        return self.f(conv_out)

class ConvLayer(object):
    def __init__(self, mi, mo, filtersz = 4, stride = 2, f = tf.nn.relu, pad = 'SAME', add_bias = False, layerNum = 0):
        
        # mi = input feature map size
        # mo = ouput feature map size
        self.f = f
        self.add_bias = add_bias
        self.stride = stride
        self.pad = pad
        
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz, filtersz, mi, mo], 
                                                              stddev = 0.02), name = 'W_conv_%i' % layerNum)
        
        self.params = [self.W]
        
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(mo,), name = 'b_conv_%i' % layerNum)
            
            self.params.append(self.b)
            
    
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, filters = self.W, strides = [1,self.stride, self.stride,1], padding=self.pad)
        
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
            
        return self.f(conv_out)
    

     
class ConvAutoEncoder(object):
    
    def __init__(self, img_size, layers, num_of_channels = 2, same_activation_for_decoder_and_encoder = True, batch_size = 32):
        
        # self.D 
        self.dimx = img_size[1]
        self.dimy = img_size[2]
        self.num_of_channels = num_of_channels
        self.layerDic= layers
        self.same_activation_for_decoder_and_encoder = same_activation_for_decoder_and_encoder
        self.build(batch_size)
        
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.1)#SGD(learning_rate=0.01)#tf.keras.optimizers.Adam()


    def build(self, batch_size):
        
        self.trainable_params = []
        
        ## Bulid the conv layer
        self.conv_layers = []
        mi = self.num_of_channels
        print("bulding the Eutoder!")
        for  mo, filtersz , stride , f, add_bias  in self.layerDic['Conv_layers']:
            conv_layer = ConvLayer(mi, mo, filtersz = filtersz, stride = stride, f = f, add_bias = add_bias)
            
            self.dimx = int(self.dimx/stride)
            self.dimy = int(self.dimy/stride)
            
            print("OutputDims:", "(", self.dimx, self.dimy, mo, ")")
            mi = mo
            
            self.conv_layers.append(conv_layer)
            
        

            
        # calculate the size of the flatten vector 
        self.mi = mi
        self.D = int(mi * self.dimx * self.dimy)
        print("D:", self.D)
        # creat all the dense layers
        self.dense_layers = []
        M1 = self.D
        #Dense layers
        count = 0
        for M2, f in self.layerDic['Dense_layers']:
            layer = DenseLayer(M1, M2, activation=f, name = 'dense %i' % count) 
            self.dense_layers.append(layer)
            M1 = M2
            count += 1
        
        
        self.dims_decoder = []
        dimx  = self.dimx
        dimy = self.dimy
        print("xy:", dimx, dimy)
        for _,_, stride, _, _ in reversed(self.layerDic['Conv_layers'][:-1]):
            # Determind the size of the data at each step
            dimx = int(np.ceil(float(dimx) * stride))
            dimy = int(np.ceil(float(dimy) * stride))
            
            self.dims_decoder.append((dimx, dimy))
        # Fro the last layer 
        self.dims_decoder.append((int(np.ceil(float(dimx) * stride)), int(np.ceil(float(dimy) * stride))))
        print("Decoder dims:", self.dims_decoder)
        
        
        ## Now let's creat the fractionallyStrideConvLayer
        decoder = []
        self.deconv_layers = []
        num = len(self.layerDic['Conv_layers']) - 1
        for i in range(num+1):
            decoder.append(self.layerDic['Conv_layers'][num-i])
        
        decoder.pop(0)
        print("Decoder")
        mi = mi
        for i in range(len(decoder)):
            mo, filtersz, stride, f, add_bias = decoder[i]
            name = "fsConv_decoder_%s" % i
            if not self.same_activation_for_decoder_and_encoder:
                try : 
                    f= self.layerDic['decoder_activations'][i]
                except: 
                    print(" Hi, there is no decoder_activationb key in your dictionary!!")
              
            output_shape = [batch_size, int(self.dims_decoder[i][0]),int(self.dims_decoder[i][1]),mo]
            # name, mi, mo, filtersz = 4, stride = 2,f = tf.nn.relu , add_bias = False
            layer =  FractionallyStrideConvLayer( name, mi, mo, filtersz=filtersz, stride=stride,
                                                 f = f, add_bias = add_bias, output_shape=output_shape)
            
            self.deconv_layers.append(layer)
            
            mi = mo
         
        # print("i", i)
        ## build the last layer 
        name = "last_layer"
        mo , filtersz, stride , f, add_bias = self.layerDic['last_layer_decoder'][0]
        # print("mo:", mo)
        output_shape = [batch_size, int(self.dims_decoder[i][0]),int(self.dims_decoder[i][1]),mo]
        # print("bal bal", output_shape)
        layer =  FractionallyStrideConvLayer( name , mi, mo, filtersz=filtersz, 
                                             stride=stride, f = f, add_bias = add_bias, output_shape = output_shape)
        self.deconv_layers.append(layer)
           
        
        ### Collect all the trainable prams from the convLAyers
        for conv in self.conv_layers:
            self.trainable_params += conv.params
            
        # define different baises for the decoder
        for layer in self.dense_layers:
            self.trainable_params += layer.params
            
        # collect al the trainable params from the deconvolution layers
        
        for layer in self.deconv_layers:
            self.trainable_params += layer.params
            
            
    def encode(self, X):
        
        Z = X
        # print(Z.shape)
        for layer in self.conv_layers:
            Z = layer.forward(Z)
            
        ### flatten the data 
        # print(Z.shape)
        Z = tf.reshape(Z, shape = (Z.shape[0], self.D))
     
        for layer in self.dense_layers:
            Z = layer.forward(Z)
            
            
        return Z
    
    def decode(self, Z):
 
        for i in range(len(self.dense_layers)-1, -1, -1):
            # print("hallo90")
            Z = self.dense_layers[i].forwardT(Z)
         
        # print("decodeZ:", Z.shape)
        Z = tf.reshape(Z, shape = (Z.shape[0],self.dimx ,self.dimy, self.mi))
        # print("decoder:", Z.shape)
        
        for layer in self.deconv_layers:
            Z = layer.forward(Z)
            
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
    
    def fit(self, X, learning_rate=0.01, epochs=50, batch_sz=32, show_fig=False):
        
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
            Z = layer.forward(Z)
            
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
    
            
            
            

        
