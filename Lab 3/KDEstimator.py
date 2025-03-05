import numpy as np
from numpy.linalg import det, inv

class KDEstimator():
    def __init__(self, bump:str='Gauss', bandwidth:str='Silverman') -> None:
        # TODO 0: Set the hyperparameters from input
        self.bump = bump
        self.bandwidth = bandwidth
        
        # set during fit
        self.M, self.N = None, None         # number of observations and features in each
        self.x_train = None                 # store training data to use in inference
        self.h = None                       # the actual value of the bandwidth

        # validation
        assert bump in ['Gauss', 'Rect'], f"Only Gauss and Rect bumps are supported but you passed {bump} to KDEstimator"
        
        # TODO 1: Assert that bandwidth is either 'Silverman' or 'Scott' or an instance of int or float
        bandwidth_is_valid = bandwidth in ['Silverman', 'Scott'] or isinstance(bandwidth, (int, float))
        assert bandwidth_is_valid, f"bandwidth must be either 'Silverman' or 'Scott' or a number but you passed {bandwidth} to KDEstimator"
        
        
    def fit(self, x_train):
       # x_train [M, N] where M is the number of observations and N is the number of features.

        # TODO 2: Set M and N and x_train
        self.M, self.N = x_train.shape[0], x_train.shape[1]
        self.x_train = x_train
        
        M, N = self.M, self.N               # to avoid corrupting eqns with self
        
        # TODO 3: compute avg_σ as defined earlier
        # Step 1: compute the variance of each feature across all observations (rows).
        # Step 2: compute the average of these variances (the avg of the one row resulting from np.var()).
        avg_σ = np.std(x_train, axis=0).mean()
        
        # TODO 4: Set self.h in case of both Silverman and Scott
        if self.bandwidth == 'Silverman':    
            self.h = avg_σ * ((4/(M * (N + 2)))**(1/(N + 4)))
        elif self.bandwidth =='Scott':
            self.h = avg_σ * (M**(-1/(N + 4)))
        else:                           
            # it must be an int or float in this case so we use it directly
            self.h = self.bandwidth
        return self
    
    def g(self, x):
        if self.bump == 'Gauss':
            N = self.N
            π = np.pi

            # TODO 5: Implement the Gaussian bump while using einsum for vectorization

            # Notes from the notebook:
            # The input assumed here is a numpy array of dimensions (m,n)
            # and the output is a numpy array of dimensions (m)
            # the evaluates the probability of each point.
            
            # Hence we dedcue that:
            # nm, mn -> m

            scale = (2*π)**(N/2)

            # Ask:
            # I wrote this according to the given dimensions and equations, but I don't understand why it works.
            # In case of one feature per point (n = 1), g(xm) represented the contribution of the m-th training point to the probability of x.
            # Now that there are multiple features, I expected g(xm) to be a vector of contributions, one for each feature.
            # But it's still a single number. How?
            
            N = np.exp(-0.5 * np.einsum('nm, mn -> m', x.T, x))
            
            return N/scale
        
        elif self.bump == 'Rect':
            # TODO 6: Implement the Rectangular bump

            # axis = 1 means sum along the columns which represent the features, resulting in the same (m, ) array we reached in the "Gauss bump case".
            # I have the same question here.
            return np.all(np.abs(x) <= 0.5, axis=1).astype(int)
          
    def ϕ(self, x): # The bump function.
        h, N = self.h, self.N
        # TODO 7: Implement ϕ as defined earlier 
        return (1/(h**N)) * self.g(x/h)
    
    def P(self, x):
        scale = 1/(self.M)
        xₘ = self.x_train
        # TODO 8: Implement P as defined earlier; remember no for loops allowed for this file

        # axis = 0 means sum along the rows which represent the training data.

        # Ask;
        # Having to specify axis=0 indicated that phi(x - xm) is a matrix of dimensions (m, n), however, I expected it to be a vector of dimensions (m), according to the implementation of g(x).
        return scale * np.sum(self.ϕ(x - xₘ), axis=0)
    
    def transform(self, x_val):
        # TODO 9: Apply P to each row of (m,n) matrix x_val using np.apply_along_axis
        # if x_val is 1D apply P to x_val directly (same line of code)

        # Ask:
        # Same dimensions question.
        return np.apply_along_axis(self.P, 1, x_val) if len(x_val.shape) > 1 else self.P(x_val)
    
    def fit_transform(self, x_data):
        # fit on x_data then transform
        return self.fit(x_data).transform(x_data)