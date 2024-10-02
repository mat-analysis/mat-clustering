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
from sklearn.cluster import AgglomerativeClustering 

from matclustering.core import SimilarityClustering

class TSAgglomerative(SimilarityClustering):
    """
    Hierarchical Agglomerative Clustering for trajectory data using a similarity matrix.

    Parameters
    ----------
    k : int, default=5
        The number of clusters to find.
    linkage : str, default='single'
        The linkage criterion to use, must be one of ['single', 'complete', 'average'].
        - 'single': minimizes the distance between the closest elements of the clusters.
        - 'complete': maximizes the distance between the furthest elements of the clusters.
        - 'average': uses the average of the distances of each point in one cluster to every point in the other cluster.
    random_state : int, default=1
        Seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs to run. Default is 1 (no parallelism).
    verbose : bool, default=False
        If True, enables verbose output during the clustering process.

    Methods
    -------
    create(config=None)
        Creates an instance of the AgglomerativeClustering model using the provided configuration.
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
    