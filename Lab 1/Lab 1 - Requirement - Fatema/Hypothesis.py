import numpy as np

class HypothesisFunction:
    def __init__(self, l, m, n):
        # TODO [1]: Initialize the function's unknown parameters
        self.l = l
        self.m = m
        self.n = n
        
        # TODO [2]: Initialize Wh and Wo matrices from a standard normal distribution
        self.Wo = np.random.randn(n, m)
        self.Wh = np.random.randn(m, l)
        
        # TODO [3]: Initialize bo and bo column vectors as zero
        self.bo = np.zeros((n, 1))
        self.bh = np.zeros((m, 1))

    def forward(self, x):
        # Ensure input shape matches input size
        assert x.shape[0] == self.l, f"Your input must be consistent the value l={self.l}"
        
        # TODO [4]: Compute a as mentioned above
        a = np.tanh(np.dot(self.Wh, x) + self.bh)
        
        # TODO [5]: Compute output ignoring ReLU
        y = np.dot(self.Wo, a) + self.bo

        
        # TODO [6]: Apply ReLU on the output with numpy boolean masking
        y[y<0] = 0
        
        return y, a

    def double_forward(self, x1, x2):
        # Perform forward function for the two inputs
        y1, _ = self.forward(x1)
        y2, _ = self.forward(x2)
        
        # TODO [7]: Concatenate the two outputs
        z = np.concatenate((y1, y2))
        
        # TODO [8]: Normalize the concatenated result
        z_bar = (z - np.mean(z)) / np.std(z)
        
        return z_bar
    
        # TODO [9]: Annotate all initialized variables and functions above