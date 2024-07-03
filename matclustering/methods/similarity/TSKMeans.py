import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from matclustering.core import SimilarityClustering

class TSKMeans(SimilarityClustering): # Trajectory KMeans
    def __init__(self,
                 k=5,
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSKMeans', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(k=k)
        
        if isinstance(k, list):
            self.grid_search(k) # list of k values transform in a 2D configs
        else:
            self.grid = [[k]] # just one config
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['k']
            ]
        return config
        
    def create(self, config=None):
        k, = self.if_config(config)
        
        return KMeans(n_clusters = k, random_state=self.config['random_state'])
    