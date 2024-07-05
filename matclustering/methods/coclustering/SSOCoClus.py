import pandas as pd
import numpy as np

from matclustering.core import TrajectoryClustering

class SSOCoClus(TrajectoryClustering):
    # UNDER DEV.
    def __init__(self,
                 # Params here
                 
                 random_state=1, # Not used, only for compatibility
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('SSOCoClus', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        #self.add_config(k=k)
        
        #self.grid_search(k)
        
    def if_config(self, config=None):
        if config == None:
            config = [
                
            ]
        return config
    
    def create(self, config=None):
        pass
    
    def fit(self, X, config=None):
        pass
    