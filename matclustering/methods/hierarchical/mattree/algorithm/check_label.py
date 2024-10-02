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
import itertools


def check_label(self, label, depth):
    """
    Method used to verify cluster aspect label in order to avoid duplicate
    names in Sankey Diagram.

    Parameters
    ----------
    self
    label : str
      Cluster aspect label.
    depth : int
      Cluster depth level.
    """

    items = list(itertools.chain.from_iterable(dict(self.label).values()))

    if label in items:
        if '#' in label:
            label, num = label.split('#')
            label += f'#{int(num) + 1}'
        else:
            label += f'#1'
            check_label(self, label, depth)
    else:
        self.label[depth].append(label)
