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
from sklearn.cluster import SpectralClustering

from matclustering.core import SimilarityClustering

class TSpectral(SimilarityClustering):
    """
    Trajectory Spectral Clustering using similarity matrix.

    Parameters
    ----------
    k : int, optional
        The number of clusters to form. Default is 5.
    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, optional
        Method of assigning labels to the clusters. 
        - 'kmeans': uses K-Means to assign labels.
        - 'discretize': uses discretization for label assignment.
        - 'cluster_qr': uses QR clustering. Default is 'discretize'.
    random_state : int, optional
        Seed for random number generation, ensuring reproducibility. Default is 1.
    n_jobs : int, optional
        The number of jobs to run in parallel for both `fit` and `predict`. Default is 1.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Methods
    -------
    create(config=None):
        Initializes and returns a Spectral Clustering model with the specified parameters.
    """
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
    