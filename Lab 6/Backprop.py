from utils import batchify              # check this function out
from tqdm import tqdm
import numpy as np


class ClassificationNeuralNet():
    def __init__(self, structure, activation='relu', random_state=42, batch_size=128, epochs=20, α=0.02, eval_train=False):
        ### Architecture hyperparameters
        self.structure = structure
        self.num_layers = len(structure) - 1                    # avoids counting the input layer
        self.activation = activation
        
        # TODO 1: Define Sigmoid function using Numpy and its derivative  (derive it or search for it)
        def σ(z): return 1 / (1 + np.exp(-z))      
        def σࠤ(z): return σ(z) * (1 - σ(z))                                # write it in terms of σ
          
        # TODO 2: Define the ReLU function and its derivative (use Numpy)
        def relu(z): return np.maximum(0, z)
        def reluࠤ(z): return np.greater(z, 0).astype(int)                           # gift
        
        # TODO 3: Set the activation function and its derivative depending on whether activation=='sigmoid' or 'relu'
        self.h, self.hࠤ = (σ, σࠤ) if activation == 'sigmoid' else (relu, reluࠤ)
        
        # This is a binary classification NN: output activation must always be Sigmoid
        self.g = σ
        
        ### Network Parameters
        
        np.random.seed(random_state)
        
        # TODO 4: for each layer except the first, initialize a (n_l, 1) vector for the bias vector with randn
        self.Bₙ = [np.random.randn(n_l, 1) for n_l in structure[1:]]                # gift     
        
        # TODO 5: for each two layers except the first, initialize (n_l, n_l_prev) matrix for the weight matrix with randn
        
        # structure[1:] and structure[:-1] are creating two lists from the structure list.
        # The first list excludes the first element of structure, and the second list excludes the last element.
        
        # For each tuple, np.random.randn(n_l, n_l_prev) generates a matrix of random numbers. 
        # The dimensions of the matrix are determined by the values of n_l and n_l_prev.

        self.Wₙ = [np.random.randn(n_l, n_l_prev) for n_l, n_l_prev in zip(structure[1:], structure[:-1])]  # gift                                        
      
        # e.g., if structure is [a,b,c,d] then we want Wₙ to hold three weight matrices of dims [b,a] , [c,b] , [d,c] 
        # You must use zip to achieve this in one line: notice it's easy if we loop on [b,c,d] and [a,b,c] 

        ### Training hyperparameters
        self.epochs = epochs
        self.α = α
        self.batch_size = batch_size
        self.eval_train = eval_train
        
        
    def feedforward(self, x, store_outputs=False):
        Zₙ, Aₙ  = [], []                                            # we may need to store outputs of each layer l for backprop
        
        a = x                                                      # the previous activation for l=1 is just the original input
        for l, (b, W) in enumerate(zip(self.Bₙ, self.Wₙ)):          # loop on each layer (after input layer)
            
            # TODO 6: Compute z = WA+b for the current layer l
            # Matrix myltipkication followed by elementwise addition.
            z = W @ a + b
            
            # TODO 7: Compute a = f(WA+b) for the current layer l
            # Recall we set f(z) as h(z) for all layers except last. For last we set f(z) as g(z). Do this in one line.
           
            # For binary classification problems, the sigmoid function is often used because it squashes its input into the range [0, 1], 
            # which can be interpreted as a probability. 
            # This is useful because the output of a binary classification model is often interpreted as the probability of the positive class.   
            # On the other hand, the ReLU does not bound its input.      
               
            a = self.g(z) if l == self.num_layers-1 else self.h(z)  # gift: ask yourself why we use self.g for the last layer
            
            if store_outputs:  Zₙ.append(z); Aₙ.append(a)           # store the outputs if feedforward is called from backprop
        
        ŷ = a                                                      # final activation is the output of the final layer (network output)
        return (ŷ, Zₙ, Aₙ) if store_outputs else ŷ
    
    

    def backprop(self, xₘ , yₘ):
        # TODO 8: Initialize derivatives of all parameters in the network as zero
        # n subscript refers that this has all derivatives of the network and m subscript that it's only given the sample (xₘ, yₘ)

        # Backpropagation for a single point (xₘ, yₘ)
        # The dimension of the bias are "number of neurons in the layer" x 1
        # The dimension of the weight matrix are "number of neurons in the layer" x "number of neurons in the previous layer"
        მJⳆმBₙₘ = [np.zeros(b.shape) for b in self.Bₙ]              # gift: ask yourself why b.shape?
        მJⳆმWₙₘ = [np.zeros(W.shape) for W in self.Wₙ]                        

        # TODO 9: Perform the feedforward pass while storing outputs
        ŷ, Zₙ, Aₙ = self.feedforward(xₘ, store_outputs=True)

        # TODO 10: Perform the backward pass to compute the derivatives for the parameters of each layer
        H = self.num_layers-1                           # index of last layer
        for l in range(H, -1, -1):
            # TODO 10.1: Compute δ (and handle the case where l==H in the same line)

            # Sigma is the derivative of the loss function wrt the output before the activation function.
            # Output layer: sigma = The output vector after the activation function minus the true output vector.
            # Hidden layers: sigma = The weight matrix of the next layer transposed times the sigma of the next layer 
            #                        elementwise multiplied by the derivative of the activation function of the current layer.
            δ =  Aₙ[l] - yₘ if l == H else self.Wₙ[l+1].T @ δ * self.hࠤ(Zₙ[l])
           
            # TODO 10.2: Compute მJⳆმBₙₘ
            მJⳆმBₙₘ[l] = δ
            # TODO 10.3: Compute მJⳆმWₙₘ (and handle the case where l==0 in the same line)

            # For the input layer, the previous layer activations are the input vector.
            მJⳆმWₙₘ[l] = δ @ Aₙ[l-1].T if l != 0 else δ @ xₘ.T
        
        return (მJⳆმBₙₘ, მJⳆმWₙₘ)
    
    

    def SGD(self, x_batch, y_batch, α):
        # TODO 11: Initialize derivatives of all parameters in the network as zero
        მJⳆმBₙ = [np.zeros(b.shape) for b in self.Bₙ]
        მJⳆმWₙ = [np.zeros(W.shape) for W in self.Wₙ]

        # TODO 12: Compute მJⳆმBₙ, მJⳆმWₙ over batch by summing მJⳆმBₙₘ, მJⳆმWₙₘ over points
        for xₘ, yₘ in zip(x_batch, y_batch):
            xₘ = xₘ[..., np.newaxis]                          # because we assume the point is a column vector (indexing x_batch gives row)
            
            # TODO 12.1: Get მJⳆმBₙₘ, მJⳆმWₙₘ for the point with backprop
            მJⳆმBₙₘ, მJⳆმWₙₘ = self.backprop(xₘ, yₘ)
            
            # TODO 12.2: Add it to the total მJⳆმBₙ, მJⳆმWₙ
            # Sum over the batch
            for l in range(self.num_layers):
                მJⳆმBₙ[l] += მJⳆმBₙₘ[l]
                მJⳆმWₙ[l] += მJⳆმWₙₘ[l]
        
        # TODO 13: Perform parameter update for each layer
        self.Bₙ = [self.Bₙ[l] - α * მJⳆმBₙ[l] for l in range(self.num_layers)]      # gift
        self.Wₙ = [self.Wₙ[l] - α * მJⳆმWₙ[l] for l in range(self.num_layers)]      
        
        

    def fit(self, x_train, y_train):
        # Split the data into batches (i.e., (m,n) => (b,m,n)
        x_data_batches, y_data_batches = batchify(x_train, y_train, self.batch_size)
        
        train_acc = ''
        range_epochs_bar = tqdm(range(self.epochs))
        for epoch in range_epochs_bar:
            
            # TODO 14: Call gradient descent to perform an update for each batch (pass self.α)
            for x_batch, y_batch in zip(x_data_batches, y_data_batches):
                self.SGD(x_batch, y_batch, self.α) 
            
            # If eval_train is true, we add the accuracy to tqdm progress bar as computed by score
            if self.eval_train:    
                train_acc = self.score(x_train, y_train)
                desc = f"Train Acc: {train_acc}" 
                range_epochs_bar.set_description(desc)
                

    def predict(self, x_val):
        # transform each x in x_val into column vector if it isn't (i.e., (m,n) => (m,n,1))
        if len(x_val.shape) == 2:   x_val = x_val[..., np.newaxis]     
        # compute the probability given by sigmoid for each x in x_val
        probs = np.array([self.feedforward(x).item() for x in x_val])
        # TODO 15: Round it with Numpy to get final predictions
        # Round to the nearest integer
        return np.round(probs)
            
    def score(self, x_val,y_val):  
        # compute the accuracy
        accuracy = (y_val == self.predict(x_val)).mean()
        return round(accuracy, 2)
