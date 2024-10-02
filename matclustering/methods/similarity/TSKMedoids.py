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

from matmodel.util.parsers import df2trajectory

from matsimilarity.core.utils import similarity_matrix
from matsimilarity.methods.mat.MUITAS import *

from matclustering.core import SimilarityClustering

class TSKMedoids(SimilarityClustering):
    """
    Trajectory K-Medoids Clustering using similarity matrix.

    The `TSKMedoids` class implements the K-Medoids clustering algorithm, 
    which is a robust alternative to K-Means, especially in the presence of noise and outliers. 

    References
    ----------
    Park, H. S., & Jun, C. H. (2009). A simple and fast algorithm for 
    K-medoids clustering. Expert systems with applications, 36(2), 3336-3341.
    <https://www.sciencedirect.com/science/article/pii/S095741740800081X>

    Parameters
    ----------
    k : int, optional
        The number of clusters to form. Default is 5.
    init : array-like or str, optional
        Initial medoids. If None, medoids will be chosen randomly. 
        If 'park', uses the method proposed by Park and Jun (2009). Default is None.
    max_iter : int, optional
        Maximum number of iterations for the algorithm to run. Default is 300.
    random_state : int, optional
        Seed for random number generation, ensuring reproducibility. Default is 1.
    n_jobs : int, optional
        The number of jobs to run in parallel for both `fit` and `predict`. Default is 1.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Methods
    -------
    create(config=None):
        Initializes and returns a K-Medoids model with the specified parameters.
    fit(X, config=None):
        Runs the K-Medoids clustering algorithm on the input data X.
    """
    def __init__(self,
                 k=5,
                 init=None, 
                 max_iter=300,
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('TSKMedoids', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.add_config(k=k,
                        init=init,
                        max_iter=max_iter,
                        n_jobs=n_jobs)
        
        self.grid_search(k, init, max_iter)
        
    def create(self, config=None):
        pass
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['n_clusters'],
                self.config['init'],
                self.config['max_iter']
            ]
        return config
    
    def fit(self, X, config=None):
        n_clusters, init, max_iter = self.if_config(config)
        
        distances = np.array(X)
            
        random_state = self.config['random_state']

        if not init:
            if random_state is not None:
                random.seed(random_state)

            idxs = np.r_[0:len(distances)]
            random.shuffle(idxs)
            medoids = idxs[:n_clusters]
        elif init == 'park':
            scores = np.zeros(len(distances))

            for j in range(0, len(distances)):
                scores[j] = 0
                for i in range(0, len(distances)):
                    scores[j] += distances[i][j] / \
                        np.sum(distances[i])

            medoids = scores.argsort()[0:n_clusters]
        else:
            medoids = init

        medoids = np.sort(medoids).astype(int)
        clusters = {}

        for self.iter in range(1, max_iter + 1):
            new_medoids = np.zeros(n_clusters)

            d = distances[:, medoids].argmin(axis=1)
            clusters = dict(zip(np.r_[0:n_clusters], [np.where(d == k)[0] for k in range(n_clusters)]))

            for k in range(n_clusters):
                kcluster=clusters[k]
                if len(kcluster)>0:
                    d = distances[np.ix_(clusters[k], clusters[k])].mean(axis=1)
                    j = d.argmin()
                    new_medoids[k] = clusters[k][j]

            new_medoids = np.sort(new_medoids).astype(int)

            if np.array_equal(medoids, new_medoids):
                break

            medoids = new_medoids.copy()
        else:
            d = distances[:, medoids].argmin(axis=1)
            clusters = dict(zip(np.r_[0:n_clusters], [np.where(d == k)[0] for k in range(n_clusters)]))

        self.clusters = np.zeros(len(distances))

        for key in clusters:
            self.clusters[clusters[key]] = key + 1

        self.clusters = self.clusters.astype(int)
        
        self._report = self.score(self.labels, self.clusters, X)
        
        return self._report, self.clusters