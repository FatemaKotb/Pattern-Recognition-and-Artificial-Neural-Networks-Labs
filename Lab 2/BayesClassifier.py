import numpy as np

class BayesClassifier():
    def __init__(self, mode:str='QDA'):
        # the class labels (will be set upon fit)
        self.labels = []
        # model learnable parameters will be set upon fit
        self.means = []
        self.covs = []
        self.priors = []
        self.weighted_cov = None        # in case of LDA only
        
        # TODO 1: Assert that mode is one of ['QDA', 'LDA', 'Naive'] and set it to self.mode
        # assert goes here

        # The syntax for the assert statement is assert condition, message
        assert mode in ['QDA', 'LDA', 'Naive'], "Invalid mode. Mode must be one of ['QDA', 'LDA', 'Naive']"
        self.mode = mode

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # TODO 2: Extract the labels from y_train using Numpy
        
        # Each element in the "y_train" array corresponds to the label of a particular training example.
        # So there are duplicates of the same label in the "y_train" array.
        # We can use the np.unique() function to extract the unique labels from the "y_train" array.
        self.labels = np.unique(y_train)

        # TODO 3: Compute the model parameters
        for label in self.labels:
            # TODO 3.1: Extract the data belonging to the current class

            # An application for "Boolean Masking" covered earlier in the course.
            x_train_given_class = x_train[y_train == label]
            
            # TODO 3.2: Compute the class mean and add it to self.means
            
            # Suppose x_train_given_class is a 2D array, where each row represents a data point and each column represents a feature. 
            # When we set axis=0, the np.mean() function will compute the mean along each column, resulting in a 1D array. 
            class_mean = np.mean(x_train_given_class, axis=0)
            self.means.append(class_mean)

            # TODO 3.3: Compute the class covariance using np.cov and add it to self.cov
            # Check the documentation of np.cov and what layout it assumes about the input
            # Set bias=False in case of QDA only and use a one-line if for that

            # numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None)
                # (m) Each row of m represents a variable, and each column a single observation of all those variables.
                # (rowvar) If it equals False then, each column represents a variable, while the rows contain observations.
                # (bias) Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate). 
            # Ask: What is the difference between the biased and unbiased estimate of the covariance matrix?
            class_cov = np.cov(x_train_given_class, rowvar=False, bias = not (self.mode == 'QDA'))
            self.covs.append(class_cov)
            
            # TODO 3.4: Compute the class prior and add it to self.priors

            # shape[0] represents the number of rows in the 2D array, which in this case is the number of data points.
            class_prior = x_train_given_class.shape[0] / x_train.shape[0]
            self.priors.append(class_prior)
        
        # TODO 4: Convert the model parameters from lists to numpy arrays
        
        # Ask: why were the model parameters stored in lists and not numpy arrays in the first place?
        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.priors = np.array(self.priors)

        # TODO 5: In case of LDA, compute the weighted covariance matrix (self.weighted_cov)
        # In one line, sum the covariance matrices for each class weighted (multiplied) by their priors
        if self.mode == 'LDA':
            # self.covs:                    (num_classes, num_features, num_features)
            # self.priors[:, None, None]:   (num_classes, 1, 1)
            # On multiplication, the shape of self.priors is broadcasted to match the shape of self.covs.

            # When axis=0, it means that the summation is performed along the first axis (num_classes) of the input array.
            self.weighted_cov = np.sum(self.covs * self.priors[:, None, None], axis=0)

            # reset self.covs to an empty list to save memory as we no longer need it
            self.covs = []

        # TODO 6: In case of Naive Bayes, diagonalize each of the covariance matrices to enforce independence
        elif self.mode == 'Naive':
            # diagonalize the covariance matrices in one line

            # numpy.diag(v, k=0)
            # If v is a 2-D array, return a copy of its k-th diagonal. 
            # If v is a 1-D array, return a 2-D array with v on the k-th diagonal.
            # K is by default the main diagonal (0).
            
            # The np.diag() function takes a square matrix as input and returns a 1D array containing the diagonal elements of the matrix.
            # np.diag(np.diag(cov)) creates a new matrix where all the off-diagonal elements are set to zero, and only the diagonal elements remain.
            self.covs = np.array([np.diag(np.diag(cov)) for cov in self.covs])

        # In case of QDA, we maintain the general setup and just return
        else: 
             return
         
    
    # TODO 7: Define the normal distribution N(x, μ, Σ) probability density function
    @staticmethod
    def N(x:np.ndarray, μ:np.ndarray, Σ:np.ndarray)->float:
        # Message to future me: Don't forget the 1/ in the scale.
        scale = 1 / np.sqrt((2 * np.pi) ** len(x) * np.linalg.det(Σ))
        exponent = -0.5 * np.dot(np.dot((x - μ).T, np.linalg.inv(Σ)), (x - μ))
        prob = scale * np.exp(exponent)
        return prob
    
    
    # TODO 8: Given a single x, compute P(C|x) using Bayes rule for each class
    def predict_proba_x(self, x:np.ndarray)->np.ndarray:
        # Message to future me: "for each class" is an indication that you should loop over all classes and to whatever idea popped into your head or was written in the TODOs.
        if self.mode != 'LDA':
            # TODO 8.1: Compute P(x|C)P(C) for each class in a numpy array in one line while using covariance matrices in self.covs
            prob_product = np.array([self.N(x, self.means[i], self.covs[i]) * self.priors[i] for i in range(len(self.labels))])
        else:
            # TODO 8.2: Compute P(x|C)P(C) for each class in a numpy array in one line while using the same covariance matrix self.weighted_cov
            prob_product = np.array([self.N(x, self.means[i], self.weighted_cov) * self.priors[i] for i in range(len(self.labels))])
        # Ask: I do not understand this normalization step.
        return prob_product/np.sum(prob_product)
    
    
    def predict_proba(self, x_val:np.ndarray)->np.ndarray:
        # given x_val of dimensions (m,n) apply predict_proba_x to each row (point x) to return array of probabilities (m, k)
        return np.apply_along_axis(self.predict_proba_x, axis=1, arr=x_val)
    
    
    def predict(self, x_val:np.ndarray)->np.ndarray:
        # TODO 9: Get the final predictions by computing argmax over the result from self.predict_proba
        y_pred_prob = self.predict_proba(x_val)
        # Ask: What is y_pred_inds?
        y_pred_inds = np.argmax(y_pred_prob, axis=1)

        # replace each prediction with label in self.labels
        y_pred = self.labels[y_pred_inds]
        return y_pred
    
    def score(self, x_val:np.ndarray, y_val:np.ndarray)->float:
        y_pred = self.predict(x_val)
        # TODO 10: compute accuracy in one line by comparing y_val and y_pred 
        # (True, False) => (1, 0)
        acc = np.mean(y_val == y_pred)
        return acc