# -*- coding: utf-8 -*-
"""
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present application offers a tool, to support the user in the clustering of multiple aspect trajectory data.It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Apr, 2024
Copyright (C) 2024, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Yuri Santos
"""
import pandas as pd
import numpy as np
from sklearn.cluster import Birch

from matclustering.core import SimilarityClustering

class TSBirch(SimilarityClustering):
    """
    BIRCH Clustering for trajectory data using a similarity matrix.

    Parameters
    ----------
    k : int or list of int, optional
        The number of clusters to find. Can be a single value or a list of values for grid search.
    random_state : int, default=1
        Seed for reproducibility. Not actively used in the BIRCH algorithm.
    n_jobs : int, default=1
        Number of parallel jobs to run. Default is 1 (no parallelism).
    verbose : bool, default=False
        If True, enables verbose output during the clustering process.

    Methods
    -------
    create(config=None)
        Creates an instance of the Birch model using the provided configuration.
    """
    
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
    