import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Adaboost:
    def __init__(self, T, random_state=42):
        self.T = T
        self.weak_clfs = [DecisionTreeClassifier(max_depth=1, random_state=random_state) for _ in range(T)]
        self.αs = []

    def fit(self, x_train, y_train):
        
        m = x_train.shape[0]
        
        # TODO 1: Initialize the weights of each point in the training set to 1/m
        # should have shape (m,)
        
        # creates a new array filled with the specified value.
        W = np.full(m, 1/m)                       

        # loop over the boosting iterations 
        for t, weak_clf in enumerate(self.weak_clfs):

            # TODO 2: fit the current week classifier on the weighted training data
            # read the docs of the fit method in sklearn.tree.DecisionTreeClassifier to see how the weights can be passed
        
            # fit(X, y, sample_weight=None, check_input=True)
            self.weak_clfs[t].fit(x_train, y_train, sample_weight=W)

            # TODO 3: Compute the indicator function Iₜ for each point. This is a (m,) array of 0s and 1s.
            hₜ = weak_clf.predict(x_train)
            Iₜ = np.array([1 if y_train_point != hₜ_point else 0 for y_train_point, hₜ_point in zip(y_train, hₜ)])
            
            # TODO 4: Use the indicator function Iₜ in boolean masking to compute the error
            errₜ =  np.sum(W[Iₜ == 1])

            # TODO 5: Compute the estimator coefficient αₜ
            αₜ = np.log((1 - errₜ) / errₜ)
            self.αs.append(αₜ)                  

            # TODO 6: Update the weights using the estimator coefficient αₜ and the indicator function Iₜ
            W = W * np.exp(αₜ * Iₜ)
            
            # TODO 7: Normalize the weights
            weights_sum = np.sum(W)
            W = W / weights_sum

        return self
    
    def predict(self, x_val):
        # TODO 8: Compute a (T, m) array of predictions that maps each estimator to its predictions of x_val weighted by its alpha
        # Use zip
        weighted_opinions = np.array([α * clf.predict(x_val) for α, clf in zip(self.αs, self.weak_clfs)])
        # Now have T evaluations of x_val each weighted (multiplied) by the corresponding alpha, 
        # so as per the formula we only need to take the sign of the sum of the different evaluations
        return np.sign(np.sum(weighted_opinions, axis=0))
            
    def score(self, x_val, y_val):
        y_pred = self.predict(x_val)
        return np.mean(y_pred == y_val)