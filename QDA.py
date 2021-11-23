
import numpy as np 

class QDA(object):
    
    def __init__(self, all_rvs, all_autoencoders):
        
        self.all_rvs = all_rvs
        self.allAutoEncoders = all_autoencoders
        
        
    def checkDistance(self, x, lowerBond = 0.01 , upperBond = 0.99):
        # here x is already encode by 
        N = len(x)
        print(N)
        Y = []
        V = []
        
        for i in range(len(self.allAutoEncoders)):
            ae = self.allAutoEncoders[i]
            rv = self.all_rvs[i]
            x_encode = ae.encode(x)
            y = []
            values = []
            for xi in x_encode:
                val = rv.cdf(xi)
                if val < upperBond and val> lowerBond:
                    y.append(1)
                    values.append(val)
                else:
                    y.append(0)
                    values.append(val) 
            Y.append(y)
            V.append(values)

        return Y, V




#     x = np.expand_dims(x, axis = 0)
#     x_encode = autoencoder.encode(x.astype('float32'))
#     val = rv.cdf(x_encode)
    
#     if val < 0.99 and val > 0.01:
#         results.append(1)
#     else:
#         results.append(0)