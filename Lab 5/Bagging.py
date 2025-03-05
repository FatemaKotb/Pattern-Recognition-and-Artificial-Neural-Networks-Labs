import numpy as np
import math
from sklearn.base import clone
from scipy.stats import mode

class Bagging:
    def __init__(self, model, n_estimators=100, max_samples=1.0, random_state=None):
        # Hyperparameters
        self.n_estimators = n_estimators    # the number of estimators (i.e., T)
        self.max_samples = max_samples      # the ratio determining the size of the bootstrap samples D_1, D_2,..,D_T
        
        # Reproducibility
        self.random_state = random_state
        
        # TODO 1: Initialize the estimators by cloning the given model 
        # Using a function from sklearn.base
        self.estimators = [clone(model) for _ in range(self.n_estimators)]

    # We will implement this once and then use it T times to generate D_1, D_2,..,D_T in training
    def _get_random_subset(self, x_data, y_data):        
        # TODO 2: Compute sample_size given dataset size and max_samples
        # Remember that "D" are the datasets.
        sample_size = int(x_data.shape[0] * self.max_samples)
        
        # TODO 3: Generating `sample_size` indices from 0 to x_data size -1 with replacement
        # Use np.random.choice
        # random.choice(a, size=None, replace=True, p=None)
        # a: 
        # 1-D array-like or int. 
        # If an ndarray, a random sample is generated from its elements. 
        # If an int, the random sample is generated as if it were np.arange(a)
        rand_inds = np.random.choice(x_data.shape[0], size=sample_size, replace=True)

        # Use the random_inds to form the sample D_t from D
        x_data_s = x_data[rand_inds]
        y_data_s = y_data[rand_inds]

        return x_data_s, y_data_s
    
    def fit(self, x_data, y_data):
        # set labels (will need in prediction)
        # Auestion: Why unique?
        self.num_labels  = np.unique(y_data)

        for t, estimator in enumerate(self.estimators):
            # Set random seem for _get_random_subset. We have to add t so next iteration its a different D_t.
            np.random.seed(self.random_state + t)  
            # TODO 4: Call _get_random_subset to get D_t
            x_data_s, y_data_s = self._get_random_subset(x_data, y_data)
            # TODO 5: Fit model M_t on the random bootstrap sample D_t
            estimator.fit(x_data_s, y_data_s)

    # Assume y_preds has dims (n_estimators, num_val_points, num_labels)
    # where y_preds[i,j,k] has the probability of the jth point in the validation set being in the kth class assigned by the ith model
    # thinking of it as a tree of matrices may help (as explained before).
    def _get_soft_vote(self, y_preds):
        # Computing the mean of y_preds along the first dimension (i), which corresponds to averaging the predictions of all the models. 
        # The result, y_pred_mean, is a 2D array where y_pred_mean[j, k] is the average probability of the j-th data point 
        # being in the k-th class.

        # TODO 6: Apply the soft voting equation presented in the notebook
        y_pred_mean = np.mean(y_preds, axis=0)
        # TODO 7.0: What's the shape of y_pred_mean now? 
        # TODO 7.1: Apply argmax over the probabilities to get the most probable class of each point by soft voting
        y_pred = np.argmax(y_pred_mean, axis=1)
        return y_pred


    def predict(self, x_val):
        # TODO 8: Define y_preds as assumed above
        y_preds = np.empty((self.n_estimators, x_val.shape[0], len(self.num_labels)))
        for i, estimator in enumerate(self.estimators):
            # TODO 9: Predict the class probabilities for all points with predict_proba
            y_preds[i] = estimator.predict_proba(x_val)
        y_pred = self._get_soft_vote(y_preds)
        return y_pred

    # Implemented as a gift 🎁.
    def score(self, x_val, y_val):
        y_pred = self.predict(x_val)
        return np.mean(y_pred == y_val)