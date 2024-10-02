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
import plotly.graph_objects as go


def sankey(cls, label, value):
    """
      Creates a sankey diagram from class dataset. Sankey diagram is a type of
      flow diagram in which the width of the arrows is proportional to the
      flow rate.
    """

    label = list(itertools.chain.from_iterable(label.values()))
    value = list(itertools.chain.from_iterable(value.values()))
    source = [val for val in cls.source for _ in (0, 1)]
    target = [tar for tar in range(1, len(source) + 1)]

    link = dict(source=source, target=target, value=value)

    node = dict(label=label, pad=50, thickness=5)
    data = go.Sankey(link=link, node=node)
    fig = go.Figure(data)
    fig.show()
