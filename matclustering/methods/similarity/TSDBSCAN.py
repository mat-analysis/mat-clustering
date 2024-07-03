import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from matclustering.core import SimilarityClustering

class TSDBSCAN(SimilarityClustering):
    """DBSCAN Clustering.

    References
    ----------
    `Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996, August). A density-
    based algorithm for discovering clusters in large spatial databases with
    noise. In Kdd (Vol. 96, No. 34, pp. 226-231).
    <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>`__
    """
    def __init__(self,
                 eps=0.5, 
                 min_samples=5,
                 
                 random_state=1, # Not used, only for compatibility
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSDBSCAN', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(eps=eps,
                        min_samples=min_samples)
        
        if isinstance(eps, list):
            self.grid_search(eps, min_samples) # list of k values transform in a 2D configs
        else:
            self.grid = [[eps, min_samples]] # just one config
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['eps'],
                self.config['min_samples']
            ]
        return config
    
    def create(self, config=None):
        eps, min_samples = self.if_config(config)
        
        return DBSCAN(eps=eps, min_samples=min_samples)
    