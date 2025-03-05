import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from Bagging import Bagging

class RandomForestClassifier(Bagging):  
    def __init__(
        self,
        # Bagging Hyperparameters
        n_estimators: int = 100,
        max_samples: float = 1.0,
        # Tree Hyperparameters
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        min_samples_split: int = 2,
        # Common
        random_state: Optional[int] = None,
        **kwargs
    ):
        # Init new parameters. You will use these in the next two TODOs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Question: What is the max_features parameter responsible for?
        self.max_features = max_features or "sqrt"
        self.random_state = random_state

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        # TODO 1.0: Create a new DecisionTreeClassifier and pass to it the relevant hyperparameters (from self)
        # TODO 1.1: Which of the features passed is responsible for column subsampling?
        self.base_model: DecisionTreeClassifier = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # TODO 2: Initialize the Bagging base class (from self)
        # def __init__(self, model, n_estimators=100, max_samples=1.0, random_state=None):

        super().__init__(self.base_model, self.n_estimators, self.max_samples, self.random_state)
    
    # ✅ Random Forest Implementation is done here. Go back to Ensemble.ipynb for a quick test and some analysis.
    
    # Ignore this function. It allows setting parameters for debugging or visualization.
    def set_params(self, **params):
        # Get the current parameters
        current_params = self.__dict__.copy()        
        current_params.update(params)        
        self.__init__(**current_params)
        