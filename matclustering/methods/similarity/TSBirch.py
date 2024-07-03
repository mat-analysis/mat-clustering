import pandas as pd
import numpy as np
from sklearn.cluster import Birch

from matclustering.core import SimilarityClustering

class TSBirch(SimilarityClustering): # Trajectory KMeans
    def __init__(self,
                 k=None,
                 
                 random_state=1, # Not used, only for compatibility
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSBirch', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(n_clusters=k)
        
        if isinstance(k, list):
            self.grid_search(k) # list of k values transform in a 2D configs
        else:
            self.grid = [[k]] # just one config
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['n_clusters']
            ]
        return config
    
    def create(self, config=None):
        n_clusters, = self.if_config(config)
        
        return Birch(n_clusters=n_clusters)
    