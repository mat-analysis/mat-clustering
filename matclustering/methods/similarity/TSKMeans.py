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
from sklearn.cluster import KMeans

from matclustering.core import SimilarityClustering

class TSKMeans(SimilarityClustering): # Trajectory KMeans
    """
    Trajectory K-Means Clustering for Trajectory Data using similarity matrix.

    The `TSKMeans` class implements the KMeans clustering algorithm, specifically 
    designed for clustering trajectory data. This implementation allows 
    for dynamic configuration of the number of clusters (k) and supports grid 
    search for hyperparameter tuning.

    Parameters
    ----------
    k : int or list of int, optional
        The number of clusters to form. Can also be a list for grid search. 
        Default is 5.
    random_state : int, optional
        Seed for random number generation, ensuring reproducibility. 
        Default is 1.
    n_jobs : int, optional
        The number of jobs to run in parallel for both `fit` and `predict`. 
        Default is 1.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Methods
    -------
    create(config=None):
        Initializes and returns a KMeans model with the specified parameters.
    """
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
    