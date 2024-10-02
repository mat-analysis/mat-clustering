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
def set_level(id_dict, depth):
    """
    Defines the cluster label.

    Parameters
    ----------
    id_dict
    depth : int
      Tree depth level.
    """
    id = 0
    while True:
        if id not in id_dict[depth]:
            id_dict[depth].append(id)
            return f'Lvl {depth} - {id}'
        if id in id_dict[depth]:
            id += 1