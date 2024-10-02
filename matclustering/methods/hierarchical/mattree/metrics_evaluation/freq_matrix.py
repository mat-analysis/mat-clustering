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
import numpy as np
import pandas as pd


def generate_freq_matrix(self, exclude_aspects=None):
    """
    Method to generate the frequency matrix

    INPUT
        exclude_aspects: A list of aspects to exclude for clustering or a empty list to use all aspects.
                        e.g.: aspects_for_clustering = [] # use all aspects
                        aspects_for_clustering = ['day', 'weather', 'root_type'] # exclude aspects for fsny
                        aspects_for_clustering = ['data', 'idVotacao', 'parlamentar'] # exclude aspects for basometro
    """

    #weather_ = ['data', 'idVotacao', 'parlamentar']
    #columns = self.data.drop(exclude_aspects, axis=1)
    # dummies = pd.get_dummies(columns, prefix_sep='~')
    # vals = dummies.drop(['tid'], axis=1)

    if isinstance(exclude_aspects, type(None)):    
        dummies = pd.get_dummies(self.data.drop(['label'], axis=1), prefix_sep='~')
    else:
        exclude_aspects.append('label')
        dummies = pd.get_dummies(self.data.drop(exclude_aspects, axis=1), prefix_sep='~')

    self.freqMatrix = pd.pivot_table(dummies, index=['tid'],
                                     values=dummies.drop(['tid'], axis=1).columns,
                                     aggfunc=np.sum)
