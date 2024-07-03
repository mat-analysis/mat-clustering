import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering

from matclustering.core import SimilarityClustering

class TSpectral(SimilarityClustering): # Trajectory KMeans
    def __init__(self,
                 k=5,
                 assign_labels='discretize', # 'kmeans', 'discretize', 'cluster_qr'
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSpectral', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(k=k,
                        assign_labels=assign_labels)
        
        if isinstance(k, list):
            self.grid_search(k, assign_labels) # list of k values transform in a 2D configs
        else:
            self.grid = [[k, assign_labels]] # just one config
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['k'],
                self.config['assign_labels']
            ]
        return config
        
    def create(self, config=None):
        k, assign_labels = self.if_config(config)
        
        return SpectralClustering(n_clusters = k, assign_labels=assign_labels, random_state=self.config['random_state'])
    