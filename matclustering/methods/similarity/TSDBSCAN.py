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
from sklearn.cluster import DBSCAN

from matclustering.core import SimilarityClustering

class TSDBSCAN(SimilarityClustering):
    """
    Trajectory Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using similarity matrix.

    The `TSDBSCAN` class implements the DBSCAN clustering algorithm, which is a 
    density-based clustering method designed to discover clusters in large spatial 
    datasets while effectively identifying noise. This implementation allows 
    configuration of key parameters such as epsilon (eps) and minimum samples.

    References
    ----------
    Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996, August). A density-
    based algorithm for discovering clusters in large spatial databases with
    noise. In Kdd (Vol. 96, No. 34, pp. 226-231).
    <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>

    Parameters
    ----------
    eps : float or list of floats, optional
        The maximum distance between two samples for them to be considered 
        as in the same neighborhood. Can also be a list for grid search.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point 
        to be considered as a core point. Default is 5.
    random_state : int, optional
        Seed for random number generation, primarily for compatibility 
        with other components. Default is 1.
    n_jobs : int, optional
        The number of jobs to run in parallel for both `fit` and 
        `predict`. Default is 1.
    verbose : bool, optional
        If True, prints verbose output during processing. Default is False.

    Methods
    -------
    create(config=None):
        Initializes and returns a DBSCAN model with the specified parameters.
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
    