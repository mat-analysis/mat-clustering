import pandas as pd
import numpy as np

from matmodel.util.parsers import df2trajectory

from matsimilarity.methods.mat.MUITAS import *
from matsimilarity.core.utils import similarity_matrix

from matclustering.core import HSTrajectoryClustering

class SimilarityClustering(HSTrajectoryClustering): # Trajectory KMeans
    def __init__(self,
                 name,
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__(name=name, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
        
    def default_metric(self, dataset_descriptor):
        # Default similarity metric is MUITAS:
        muitas = MUITAS(dataset_descriptor)

        # Default Config:
        for feat in dataset_descriptor.attributes:
            muitas.add_feature([feat], 1)

        return muitas
    
    def prepare_input(self, X, metric=None, dataset_descriptor=None):
        
        if isinstance(X, pd.DataFrame):
            T, dataset_descriptor = df2trajectory(X.copy())
        else: 
            T = X # Trajectories already converted
        
        if not metric:
            if self.isverbose:
                print('\n['+self.name+':] Default metric set to MUITAS.')
                
            self.metric = self.default_metric(dataset_descriptor)
        else:
            self.metric = metric
        
#        self.X = list(map(lambda t1: list(map(lambda t2: self.metric(t1, t2), T)), T))
        self.X = 1 - similarity_matrix(T, measure=self.metric, n_jobs=self.config['n_jobs'])

        #classes = list(map(lambda t1: t1.label, T))
        self.labels = list(map(lambda t1: t1.label, T))
        
        return self.X, self.labels