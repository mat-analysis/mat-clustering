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

def Rec_error(data,clusters):
    '''
    This evaluation measure is computed during the algorithm life time.
    '''
    reconstructed_ococlus = np.zeros(data.shape,dtype=int)
    for nc in range(len(clusters)):
        for i in clusters[nc][1]: # object cluster
            for j in clusters[nc][0]: # attribute cluster
                reconstructed_ococlus[int(i)][int(j)] = 1
    print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_ococlus)))
    
def build_clustering_output_omega(clusters,clus_labels,trad=False):
    # build the clustering output format to use in the omega index evaluation
    clustering = {}
    num_of_elements = 0
    
    if trad:
        for cluster_id in clusters:
            clustering["c"+str(cluster_id)] = []
            for index, value in enumerate(clus_labels[0]):#looping over the labels
                if cluster_id == value:
                    [clustering["c"+str(cluster_id)].append(("01"+str(index)+"02"+str(j))) for j in range(clus_labels[1])]# num of attributes
            num_of_elements+=len(clustering["c"+str(cluster_id)])
    else:
        if isinstance(clusters,int):
            num = clusters
        else:
            num = (clusters[0]*clusters[1])

        seq_index = 0
        for nc in range(num):
            ## check if exist some co-cluster with no element assignment for object and attribute
            if len(clus_labels.get_indices(nc)[0]) != 0 and len(clus_labels.get_indices(nc)[1]) != 0:
                clustering["c"+str(seq_index)] = []
                for i in clus_labels.get_indices(nc)[0]:
                    for j in clus_labels.get_indices(nc)[1]:
                        clustering["c"+str(seq_index)].append(("01"+str(i)+"02"+str(j)))
                num_of_elements += len(clustering["c"+str(seq_index)])
                seq_index+=1

    return clustering

def build_coclustering_output_omega(co_clusters):
# def build_clustering_output_omega(rowClusters,columnClusters):
    '''
    Build the clustering output format to use in the omega index evaluation from Remy Cazabet version.
    It is optional and we just present this version as a complementary information. If you are interested,
    check it out on his team work group at https://github.com/isaranto/omega_index.
    '''
    
    num_of_clusters = len(co_clusters)    
    clustering = {}
    
    for nc in range(num_of_clusters):
        rowCluster = co_clusters[nc][1]
        columnCluster = co_clusters[nc][0]
        clustering["c"+str(nc)] = []
        
        for i in rowCluster:
            for j in columnCluster:
                clustering["c"+str(nc)].append(("01"+str(i)+"02"+str(j)))
        
    return clustering

def xmeasures_format(dict_gt):
    '''
    This function build the xmeasure format to use it on their evaluation measure.
    '''
    newData = []
    for i in range(len(dict_gt)):
#         print(dict_gt['c'+str(i)])
        stringLine = dict_gt['c'+str(i)][0]
        for j in range(1,len(dict_gt['c'+str(i)])):
#             stringLine = stringLine+" "+dict_gt['c'+str(i)][j]
            stringLine += " "+dict_gt['c'+str(i)][j]
        newData.append(stringLine)
    
    return newData