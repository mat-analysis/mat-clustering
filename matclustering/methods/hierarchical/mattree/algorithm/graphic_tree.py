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
from datetime import datetime

from IPython.core.display import display_png
from graphviz import Digraph


def graphic_tree(self, graphTree):
    """
      Shows info about each cluster node in the tree generated by Digraph
      plot.

      Parameters
      ----------
      graphTree : Digraph
        A base class for directed graphs.
    """

    threshold = '{:.4f}'.format(self.thresholdVal)
    if self.left.thresholdVal is not None and self.right.thresholdVal is not None:
        thresholdLeft = '{:.4f}'.format(self.left.thresholdVal)
        thresholdRight = '{:.4f}'.format(self.right.thresholdVal)
    elif self.left.thresholdVal is not None and self.right.thresholdVal is None:
        thresholdLeft = '{:.4f}'.format(self.left.thresholdVal)
    elif self.left.thresholdVal is None and self.right.thresholdVal is not None:
        thresholdRight = '{:.4f}'.format(self.right.thresholdVal)

    tres_ = "\nMean_tres["
    if self.split == 'binary':
        left = 'NO'
        right = 'YES'
        thres = ''
    else:
        left = 'less than the average'
        right = 'greater than the average'
        thres = tres_ + str(threshold) + "]"

    if self.left.done == 'No' and self.right.done == 'No':
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       str(self.left.id) + "\n" + self.leftChildName + tres_ + str(thresholdLeft) + "]",
                       left)
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       str(self.right.id) + "\n" + self.rightChildName + tres_ + str(thresholdRight) + "]",
                       right)

    elif self.left.done == 'No' and self.right.done == 'Yes':
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       str(self.left.id) + "\n" + self.leftChildName + tres_ + str(thresholdLeft) + "]",
                       left)
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       # str(self.right.id)+"\n"+self.rightChildName+"\n["+self.division['value']+"]",right)
                       str(self.right.id) + "\n" + self.rightChildName, right)

    elif self.left.done == 'Yes' and self.right.done == 'No':
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       # str(self.left.id)+"\n"+self.leftChildName+"\n["+self.division['value']+"]",left)
                       str(self.left.id) + "\n" + self.leftChildName, left)
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       str(self.right.id) + "\n" + self.rightChildName + tres_ + str(thresholdRight) + "]",
                       right)

    else:
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       # str(self.left.id)+"\n"+self.leftChildName+"\n["+self.division['value']+"]",left)
                       str(self.left.id) + "\n" + self.leftChildName + "\nU.T.", left)
        graphTree.edge(str(self.id) + "\n" + self.parentName + tres_ + str(threshold) + "]",
                       # str(self.right.id)+"\n"+self.rightChildName+"\n["+self.division['value']+"]",right)
                       str(self.right.id) + "\n" + self.rightChildName + "\nU.T.", right)

    if self.left.done == 'No':
        graphic_tree(self.left, graphTree)
    if self.right.done == 'No':
        graphic_tree(self.right, graphTree)


def generate_graphic_tree(self, dir_path):
    tree = 'Clustering-Tree-{}'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    graph = Digraph(tree, comment='Arvore de clusters')
    graph.attr(size="22")
    graph.attr()

    graphic_tree(self, graph)
    graph.format = 'png'

    graph.render(directory=dir_path, view=True, format='png')
    display_png(graph)
