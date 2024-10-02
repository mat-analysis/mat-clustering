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


from matclustering.core import TrajectoryClustering

from matclustering.methods.hierarchical.mattree.algorithm.TreeNodeObject import TreeNodeObject
from matclustering.methods.hierarchical.mattree.algorithm.dashtree import dashtree

from graphviz import Digraph
from matclustering.methods.hierarchical.mattree.algorithm.graphic_tree import graphic_tree

# MAT-Tree: A Tree-Based Method for Multiple Aspect Trajectory Clustering
class MATTree(TrajectoryClustering): 
    """
    MAT-Tree: A Tree-Based Method for Multiple Aspect Trajectory Clustering.

    The `MATTree` class provides an implementation of a tree-based hierarchical clustering 
    method for analyzing trajectory data with multiple aspects. 

    Parameters
    ----------
    exclude_aspects : list, optional
        A list of aspects to exclude from the clustering process. If empty, all aspects 
        are used for clustering. Default is an empty list.
        
    random_state : int, optional
        Seed for random number generation, ensuring reproducibility. Default is 1 (not used).
    n_jobs : int, optional
        The number of jobs to run in parallel for both `fit` and `predict`. Default is 1.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Methods
    -------
    prepare_input(X, metric=None, dataset_descriptor=None, tid_col='tid', label_col='label'):
        Prepares the input data for clustering by extracting relevant trajectory information.
    create(config=None):
        Initializes and returns a tree-based model for clustering (currently not implemented).
    fit(X, config=None):
        Fits the model to the trajectory data `X`, applying the clustering algorithm.
    digraph():
        Creates a directed graph representation of the clustering tree.

    """
    
    def __init__(self,
                 exclude_aspects = [], # use all aspects
                 
                 random_state=1, # Not used, only for compatibility
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('MATTree', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        self.X = None
        
        self.add_config(exclude_aspects=exclude_aspects)
        
        self.grid = [[exclude_aspects]] # just one config
        
    def prepare_input(self, X, metric=None, dataset_descriptor=None, tid_col='tid', label_col='label'):
        self.tid_col = 'tid'
        self.label_col = 'label'
        
        self.X = X.copy()
        
        self.labels = np.array(X[[self.tid_col, self.label_col]].drop_duplicates().label)
        
        return self.X
        
    def if_config(self, config=None):
        if config == None:
            config = [
                self.config['exclude_aspects']
            ]
        return config
    
    def create(self, config=None):
        pass
    
    def fit(self, X, config=None):
        
        self.X = X
        
        if not self.model:
            self.model = self.create(config)
        
        exclude_aspects, = self.if_config(config)
        
        self.model = TreeNodeObject(df=self.X)
        dashtree(self.model, self.X, exclude_aspects+[self.label_col])
    
        tids = X[[self.tid_col,self.label_col]].drop_duplicates().tid

        keys = list(self.model.df_leaves.keys())
        
        clusters = list(map(lambda traj: keys.index(next(filter(lambda cluster: int(traj) in self.model.df_leaves[cluster].tid.unique().tolist(), self.model.df_leaves))), tids))
        clusters = np.array(clusters)
        
        self._report = self.score(self.labels, clusters)
        
        self.clusters = clusters
        
        return self._report, self.clusters

    def digraph(self):
        graph = Digraph()
        graphic_tree(self.model, graph)
        return graph