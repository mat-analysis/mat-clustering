import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering 

from matclustering.core import SimilarityClustering

class TSAgglomerative(SimilarityClustering):
    """Hierarchical Agglomerative Clustering.

    """
    def __init__(self,
                 k=5,
                 linkage='single', # ['single', 'complete', 'average']
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSKMeans', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(k=k,
                        linkage=linkage)
        
        if isinstance(k, list):
            self.grid_search(k, linkage) # list of k values transform in a 2D configs
        else:
            self.grid = [[k, linkage]] # just one config
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['k'],
                self.config['linkage']
            ]
        return config
    
    def create(self, config=None):
        k, linkage = self.if_config(config)
        
        return AgglomerativeClustering(n_clusters = k, affinity='precomputed', linkage=linkage)
    