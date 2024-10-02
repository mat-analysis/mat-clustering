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

class SSOCoClus(TrajectoryClustering):
    # UNDER DEV.
    def __init__(self,
                 # Params here
                 
                 random_state=1, # Not used, only for compatibility
                 n_jobs=1,
                 verbose=False):
        
        super().__init__('SSOCoClus', random_state=random_state, n_jobs=n_jobs, verbose=verbose)

        #self.add_config(k=k)
        
        #self.grid_search(k)
        
    def if_config(self, config=None):
        if config == None:
            config = [
                
            ]
        return config
    
    def create(self, config=None):
        pass
    
    def fit(self, X, config=None):
        pass
    