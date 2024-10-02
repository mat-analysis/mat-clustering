# -*- coding: utf-8 -*-
"""
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present application offers a tool, to support the user in the clustering of multiple aspect trajectory data.It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Apr, 2024
Copyright (C) 2024, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
"""
import os
import pandas as pd
from datetime import datetime
import itertools

from tqdm.auto import tqdm

from abc import ABC, abstractmethod

from sklearn.metrics import *

class TrajectoryClustering(ABC):
    """
    Abstract base class for trajectory clustering algorithms.

    This class provides a framework for clustering multiple-aspect trajectory data. 
    It allows the configuration of clustering parameters, performs hyperparameter 
    combinations for grid search optimization (see HSTrajectoryClustering class), 
    and evaluates clustering results using various metrics.

    Attributes
    ----------
    name : str
        Name of the clustering model.
    isverbose : bool
        Flag indicating whether to print verbose output.
    save_results : bool
        Flag to indicate if results should be saved.
    config : dict
        Configuration dictionary to hold hyperparameters and settings.
    model : object
        The clustering model instance.
    report : DataFrame
        Report of clustering evaluation metrics.
    test_report : DataFrame
        Report of clustering evaluation metrics on test data.
    
    Methods
    -------
    add_config(**kwargs):
        Updates the configuration dictionary with new parameters.
    
    grid_search(*args):
        Generates combinations of hyperparameters for grid search.
    
    duration():
        Returns the elapsed time since the model was initialized in milliseconds.
    
    clear():
        Clears the current model instance.
    
    message(pbar, text):
        Displays a message during model training/testing.
    
    prepare_input(X, metric=None, dataset_descriptor=None):
        Prepares the input data for clustering (to be implemented by subclasses).
    
    create(config=None):
        Creates and returns the clustering model instance (to be implemented by subclasses).
    
    score(y_test, y_pred, X=None):
        Calculates and returns various clustering evaluation metrics as a DataFrame.
    
    summary():
        Returns a summary of the clustering results.
    
    fit(X, config=None):
        Fits the clustering model to the input data and returns the report and cluster labels.
    
    save(dir_path='.', modelfolder='model'):
        Saves the clustering model and its results to the specified directory.
    
    cluestering_report():
        Generates a DataFrame of cluster predictions and labels.
    """
    def __init__(self,
                 name='NAME?',
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        self.name = name
        
        self.isverbose = verbose >= 0
        self.save_results = False # TODO
        
        self.config = dict()
        self.add_config(n_jobs=n_jobs,
                        verbose=verbose,
                        random_state=random_state)
        
        self.model = None
        
        self.report = None
        self.test_report = None
        
        if self.isverbose:
            print('\n['+self.name+':] Building model')
        self.start_time = datetime.now()
        
    def add_config(self, **kwargs):
        self.config.update(kwargs)
    
    def grid_search(self, *args):
        params = []
        for arg in args:
            if not isinstance(arg, list):
                arg = [arg]
            params.append(arg)
        
        self.grid = list(itertools.product(*params))
    
    def duration(self):
        return (datetime.now()-self.start_time).total_seconds() * 1000
    
    def clear(self):
        del self.model 
        
    def message(self, pbar, text):
        if isinstance(pbar, list):
            if self.isverbose:
                print(text)
        else:
            pbar.set_postfix_str(text)
    
    @abstractmethod
    def prepare_input(self, X, metric=None, dataset_descriptor=None):
        pass
    
    @abstractmethod
    def create(self, config=None):
        pass
    
    def score(self, y_test, y_pred, X=None):
        # Calculate clustering metrics        
        ari = adjusted_rand_score(y_test, y_pred)
        mi = mutual_info_score(y_test, y_pred)
        
        ami = adjusted_mutual_info_score(y_test, y_pred)
        completeness = completeness_score(y_test, y_pred)
        fmi = fowlkes_mallows_score(y_test, y_pred)
        homogeneity = homogeneity_score(y_test, y_pred)
        nmi = normalized_mutual_info_score(y_test, y_pred)
        ri = rand_score(y_test, y_pred)
        v_measure = v_measure_score(y_test, y_pred)
        
        dic_model = {
            'rand_index': ri, # Rand index.
            'adusted_rand_index': ari, # Adjusted Rand Index (ARI) 
            'mutual_info': mi, # Mutual Information (MI)
            'adusted_mutual_info': ami, # Adjusted Mutual Information between two clusterings.
            'norm_mutual_info': nmi, # Normalized Mutual Information between two clusterings.
            'fm_index': fmi, # The Fowlkes-Mallows index (FMI)
            'v_measure': v_measure, # V-measure cluster labeling given a ground truth.
            
            # Attention as this metrics need y_test and y_pred as ground truth
            'completeness': completeness, # Compute completeness metric of a cluster labeling given a ground truth.
            'homogeneity': homogeneity, # Homogeneity metric of a cluster labeling given a ground truth.
        } 
        
        if X is not None:
            silhouette = silhouette_score(X, y_pred) 
            db_index = davies_bouldin_score(X, y_pred)
            ch_index = calinski_harabasz_score(X, y_pred)
            
            dic_model.update( {
                'silhouette': silhouette, # Silhouette Score
                'db_index': db_index, # Davies-Bouldin Index
                'ch_index': ch_index, # Calinski-Harabasz Index (Variance Ratio Criterion)
            } )
        
        return pd.DataFrame(dic_model, index=[0])
    
    def summary(self):
        if self.test_report is not None:
            return pd.DataFrame(self.test_report.mean()).T
        else:
            return pd.DataFrame(self._report.mean()).T
    
    def fit(self, X, config=None):
        
        if not self.model:
            self.model = self.create(config)
            
        self.model.fit(X)
        
        self._report = self.score(self.labels, self.model.labels_, X)
        
        self.clusters = self.model.labels_
        
        return self._report, self.clusters
    
    def save(self, dir_path='.', modelfolder='model'):
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))

        report = self.cluestering_report()
        if report is not None:
            report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_report.csv'), index = False)

    def cluestering_report(self):
        df = pd.DataFrame(self.labels, self.clusters, columns=['cluster'])
        df.rename_axis(index='prediction', inplace=True)
        return df

class HSTrajectoryClustering(TrajectoryClustering): # Hyperparam search model
    """
    Class for hyperparameter search in multiple-aspect trajectory clustering algorithms.

    This class extends the TrajectoryClustering class to include hyperparameter 
    optimization (tuning through grid search) and model validation. It implements methods for training, testing, 
    and saving models, along with detailed reporting of the results.

    Attributes
    ----------
    best_config : list
        The best hyperparameter configuration found during training.

    Methods
    -------
    prepare_input(X):
        Prepares the input data for training (to be implemented by subclasses).
    
    if_config(config=None):
        Returns the current configuration or default configuration.
    
    train(dir_validation='.'):
        Trains the model using grid search over hyperparameters and returns the training report.
    
    test(rounds=1, dir_evaluation='.'):
        Tests the best model on the dataset and returns evaluation metrics.
    
    save(dir_path='.', modelfolder='model'):
        Saves the model and its training/testing results to the specified directory.
    
    training_report():
        Returns the training report DataFrame.
    
    testing_report():
        Returns the testing report DataFrame.
    """
    def __init__(self,
                 name='NAME?',
                 
                 random_state=1,
                 n_jobs=1,
                 verbose=False):
        
        super().__init__(name=name, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
        
        self.best_config = None
        
    @abstractmethod
    def prepare_input(self, X):
        pass
    
    # Override this method
    @abstractmethod
    def if_config(self, config=None):
        return config
    
    def train(self, dir_validation='.'):
        
        # This implementation, trains only one model 
        # (but, you may overwrite the method following this method)
        
        self.start_time = datetime.now()
        
        X = self.X       
        
        if self.isverbose:
            print('['+self.name+':] Training model')
        
        # Hiper-param data:
        data = []
        
        self.best_config = [-1, []]
        
        if self.isverbose:
            pbar = tqdm(self.grid, desc='['+self.name+':] Model Training')
        else:
            pbar = self.grid
        
        for config in pbar:
            
            # concat config:
            params = '-'.join([str(y) for y in config])
            filename = os.path.join(dir_validation, 'val_'+self.name.lower()+'-'+params+'.csv')
            
            if os.path.exists(filename):
                self.message(pbar, 'Skip ---> {}'.format(filename))
                data.append(self.read_report(filename))
                
            else:
                self.message(pbar, 'Trainning Config - '+params)
                
                self.model = self.create(config)
                
                validation_report, y_pred = self.fit(X, config)

                if self.save_results:
                    validation_report.to_csv(filename, index=False)

                for index, (att, val) in enumerate(zip(['p'+str(y) for y in range(len(config))], config)):
                    validation_report[att] = [val]

                data.append( validation_report )
                
            # Choose the best model based on higher acc:
            ari = validation_report.iloc[0]['adusted_rand_index']#['acc']
            if ari > self.best_config[0]:
                self.best_config = [ari, config]
                self.best_model = self.model
            
            # TODO Save model results ??
            self.clear()
            # ------------------------------------------->
            
        self.best_config = self.best_config[1]
        
        self.report = pd.concat(data)
        self.report.reset_index(drop=True, inplace=True)

#        self.report.sort_values('acc', ascending=False, inplace=True)
        self.report.sort_values('adusted_rand_index', ascending=False, inplace=True)
        
        self.model = self.best_model
        
        return self.report
    
    def test(self,
             rounds=1,
             dir_evaluation='.'):
        
        X = self.X
        
        filename = os.path.join(dir_evaluation, 'eval_'+self.name.lower()+'.csv')
        
        if os.path.exists(filename):
            if self.isverbose:
                print('['+self.name+':] Model previoulsy built.')
            # TODO read
            #return self.read_report(filename, prefix='eval_')
        else:
            if self.isverbose:
                print('['+self.name+':] Creating a model to test set')
            
                pbar = tqdm(range(rounds), desc="Model Testing")
            else:
                pbar = list(range(rounds))
                
            random_state = self.config['random_state']
            
            config = self.if_config(self.best_config)
            
            evaluate_report = []
            for e in pbar:
                re = (random_state+e)
                self.config['random_state'] = re
                
                self.message(pbar, 'Round {} of {} (random {})'.format(e, rounds, re))
                
                self.model = self.create(config)
                
                eval_report, y_pred = self.fit(X, config)
                
                eval_report['random_state'] = re
                eval_report['runtime'] = self.duration()
                
                evaluate_report.append(eval_report)
                        
            self.config['random_state'] = random_state
            self.test_report = pd.concat(evaluate_report)
            self.test_report.reset_index(drop=True, inplace=True)
            
            if self.isverbose:
                print('['+self.name+':] Processing time: {} milliseconds. Done.'.format(self.duration()))

            return self.test_report, y_pred
        
    def save(self, dir_path='.', modelfolder='model'):
        super().save(dir_path, modelfolder)
            
        train_report = self.training_report()
        if train_report is not None:
            train_report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_history.csv'))
            
        test_report = self.testing_report()
        if test_report is not None:
            test_report.to_csv(os.path.join(dir_path, modelfolder, 'model_'+self.name.lower()+'_summary.csv'))

    def training_report(self):
        return self.report
    def testing_report(self):
        return self.test_report