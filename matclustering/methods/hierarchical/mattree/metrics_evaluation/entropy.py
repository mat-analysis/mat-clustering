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
import math


def get_entropy(dataset):
    """
      Calculates the entropy value of a given dataset.

      Parameters
      ----------
      dataset : pandas.DataFrame
        Dataset of trajectories of a given cluster

      Returns
      -------
      Float
        Entropy value of a given cluster.
    """
    df1 = dataset.copy()
    entropy_dict = {}
    for e in df1.label.unique():
        num_traj = df1[df1.label == e].tid.unique()
        entropy_dict[e] = len(num_traj)
    total = sum(entropy_dict.values())

    entropy_value = 0
    for key, value in entropy_dict.items():
        p = value / total
        entropy_value += p * math.log2(p)

    try:
        max_entropy = math.log2(len(df1.label.unique()))
        return -entropy_value / max_entropy
    except Exception as e:
        return -entropy_value