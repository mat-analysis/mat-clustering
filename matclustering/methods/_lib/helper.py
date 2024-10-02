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
from collections import deque
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import plotly
import chart_studio.plotly as py
import plotly.graph_objects as go
from timeit import default_timer as timer
import datetime
import sys, os, shutil, gc
from csv import DictWriter

class Performance():
#     perf_df_clustering_output_measures = pd.DataFrame(columns = ['Iteration_i','Candidate_iteration_k',
#                                                             'Candidate_cost'])
    
#     df_quality_clustering = pd.DataFrame(columns= ['Dataset','Clustering_approach','Cocluster_reference',
#                                                    'Cocluster_statistic','Num_of_candidates','Num_of_clusters',
#                                                    'Overall_entropy','Purity'])
#     df_scability = pd.DataFrame(columns = ['num_of_elements','time_minutes','dataset','run_simulation'])

    def __init__(self):
        self.df_scability = pd.DataFrame(columns = ['num_of_elements','time_minutes','run_simulation',
                                                    'num_of_candidates','dataset'])
        
        self.df_scab_rows_cost = pd.DataFrame(columns = ['candidate_id','candidate_num_rows','candidate_cost',
                                                         'time_discovered_minutes','num_of_traj_points','num_of_elements',
                                                         'run_simulation','dataset'])
        self.df_scab_rows_cost.candidate_id = self.df_scab_rows_cost.candidate_id.astype(float)
        self.df_scab_rows_cost.candidate_num_rows = self.df_scab_rows_cost.candidate_num_rows.astype(float)
        self.df_scab_rows_cost.candidate_cost = self.df_scab_rows_cost.candidate_cost.astype(float)
    
    def plot_scability_test(self,file):
        self.dpi = 600
        self.fig = plt.figure(figsize=(3, 2),dpi=self.dpi)
        self.ax = sns.lineplot(x = "num_of_elements", y = "time_minutes", hue='dataset', err_style='bars', data = self.df_scability)
        
        self.ax.legend(loc='upper left', fontsize=4)
        sns.despine(offset=0, trim=True, left=True)
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        self.ax.set_yticklabels(self.ax.get_ymajorticklabels(), fontsize = 6)
        self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.xticks(horizontalalignment='center',fontsize=6)
#         ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 6)
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(70))
#         ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.ylabel('AVG time (minutes)',fontsize=7)
        plt.xlabel('Number of elemetns',fontsize=7)
        self.axes = plt.gca()
        self.axes.yaxis.grid(color='black',linewidth=.1)
        plt.tight_layout()
#         fig.savefig('C:/Users/yurin/Downloads/'+file+'.png',transparent=True,bbox_inches = 'tight',pad_inches=0,dpi=dpi)
        plt.show()
#         print(self.df_scability.head())

    def plot_scability_rows_cost(self,file):
        self.dpi = 600
        self.fig = plt.figure(figsize=(3, 2),dpi=self.dpi)
        self.ax = sns.lineplot(x = "candidate_id",y = "candidate_cost",hue = "dataset",err_style='bars', data = self.df_scab_rows_cost)
        
#         self.ax.legend(loc='upper right', fontsize=4)
        self.ax.legend(fontsize=4)
        sns.despine(offset=0, trim=True, left=True)
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        self.ax.set_yticklabels(self.ax.get_ymajorticklabels(), fontsize = 6)
        self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.xticks(horizontalalignment='center',fontsize=6)
#         ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 6)
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(70))
#         ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.ylabel('Cost value per candidate',fontsize=7)
        plt.xlabel('Number of candidates',fontsize=7)
#         self.axes = plt.gca()
#         self.axes.yaxis.grid(color='black',linewidth=.1)
#         plt.tight_layout()
#         fig.savefig('C:/Users/yurin/Downloads/'+file+'.png',transparent=True,bbox_inches = 'tight',pad_inches=0,dpi=dpi)
        plt.tight_layout()
        plt.show()
        
#         self.dpi = 600
#         self.fig2 = plt.figure(figsize=(3, 2),dpi=self.dpi)
#         self.ax2 = sns.lineplot(x = "candidate_id", y = "candidate_num_rows", hue = "dataset",data = self.df_scab_rows_cost)
#         self.ax2.legend(loc='upper right', fontsize=4)
#         sns.despine(offset=0, trim=True, left=True)
#         self.ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
#         self.ax2.set_yticklabels(self.ax2.get_ymajorticklabels(), fontsize = 6)
#         self.ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
#         plt.xticks(horizontalalignment='center',fontsize=6)
#         plt.ylabel('Num of trajs. per candidate',fontsize=7)
#         plt.xlabel('Number of candidates',fontsize=7)
        
#         plt.tight_layout()
#         plt.show()
    
    def store_data_scability_rows_cost(self,cc_id,num_of_rows,cost,seq_len,time_elapse,num_of_traj_points,num_of_els,run_sim,dataset):
        if dataset == 'fs_ny_top_users_193.dat':
            self.__sdst_dataset = 'All users'
        elif dataset == 'fs_ny_top_users_81.dat':
            self.__sdst_dataset = 'Top 81 users'
        elif dataset == 'fs_ny_top_users_10.dat':
            self.__sdst_dataset = 'Top 10 users'
        elif (dataset == 'sjgs.dat') or (dataset == 'sjgs2.dat'):
            self.__sdst_dataset = dataset.split('.')[0].upper()
        else:
            self.__sdst_dataset = 'Undefined dataset.'
        
#         self.df_scab_rows_cost = self.df_scab_rows_cost.append({'candidate_id':cc_id,
#                                                                 'candidate_num_rows':int(num_of_rows),
#                                                                 'candidate_cost':int(cost),
#                                                                 'time_discovered':time_elapse,
#                                                                 'num_of_traj_points':int(num_of_traj_points),
#                                                                 'num_of_elements':num_of_els,
#                                                                 'run_simulation':run_sim,
#                                                                 'dataset':self.__sdst_dataset},
#                                                                ignore_index=True)
        #list of column names
        self.frc_field_names = ['candidate_id','candidate_num_rows','candidate_cost','candidate_seq_len',
                                'time_discovered_minutes','num_of_traj_points','num_of_elements','run_simulation',
                                'dataset']
        
        # Dictionary
        self.frc_data = {'candidate_id':cc_id,'candidate_num_rows':int(num_of_rows),'candidate_cost':int(cost),
                         'candidate_seq_len':int(seq_len),'time_discovered_minutes':time_elapse,
                         'num_of_traj_points':int(num_of_traj_points),'num_of_elements':num_of_els,
                         'run_simulation':run_sim,'dataset':self.__sdst_dataset}
        
        with open('./coclustering_file_outputs/df_scab_rows_cost.csv', 'a+', newline='') as self.frc_object:
      
            # Pass the file object and a list 
            # of column names to DictWriter()
            # You will get a object of DictWriter
            self.rc_dictwriter_object = DictWriter(self.frc_object, delimiter=";", fieldnames=self.frc_field_names)

            #Pass the dictionary as an argument to the Writerow()
            self.rc_dictwriter_object.writerow(self.frc_data)

            #Close the file object
            self.frc_object.close()
    
    def store_data_scability_test(self,candidates_ref_values,num_of_els,time_elapse,dataset,run_sim):
#         datasets = ['fs_ny_top_users_193.dat','fs_ny_top_users_81.dat','fs_ny_top_users_10.dat']
        if dataset == 'fs_ny_top_users_193.dat':
            self.__sdst_dataset = 'All users'
        elif dataset == 'fs_ny_top_users_81.dat':
            self.__sdst_dataset = 'Top 81 users'
        elif dataset == 'fs_ny_top_users_10.dat':
            self.__sdst_dataset = 'Top 10 users'
        elif (dataset == 'sjgs.dat') or (dataset == 'sjgs2.dat'):
            self.__sdst_dataset = dataset.split('.')[0].upper()
        else:
            self.__sdst_dataset = 'Undefined dataset.'
        
#         print('Salvando scability.',end=' ')
        
#         for cc, value in candidates_ref_values.items():
#             self.__sdst_cc_num_of_rows = candidates_ref_values[cc]['rows']
#             self.__sdst_cc_cost = candidates_ref_values[cc]['cost']
            
#         self.df_scability = self.df_scability.append({'num_of_elements':str(num_of_els),
#                                                       'time_minutes':time_elapse,
#                                                       'run_simulation':run_sim,
#                                                       'num_of_candidates':len(candidates_ref_values),
#                                                       'dataset':self.__sdst_dataset},
#                                                      ignore_index=True)
        
        #list of column names
        self.sdst_field_names = ['num_of_elements','time_minutes','run_simulation','num_of_candidates','dataset']
        
        # Dictionary
        self.sdst_data = {'num_of_elements':str(num_of_els),'time_minutes':time_elapse,'run_simulation':run_sim,
                     'num_of_candidates':len(candidates_ref_values),'dataset':self.__sdst_dataset}
        
        with open('./coclustering_file_outputs/df_scability.csv', 'a+', newline='') as self.fsdst_object:
      
            # Pass the file object and a list 
            # of column names to DictWriter()
            # You will get a object of DictWriter
            self.sdst_dictwriter_object = DictWriter(self.fsdst_object, delimiter=";", fieldnames=self.sdst_field_names)

            #Pass the dictionary as an argument to the Writerow()
            self.sdst_dictwriter_object.writerow(self.sdst_data)

            #Close the file object
            self.fsdst_object.close()

    
    def compute_measures_at_once(self,set_of_candidates,candidates_ref_values,map_tid_to_el,trajs_data_dict_list,file_dataset,store_dist=None):
        '''
        Method to compute the measures at once for given dataset.
        It is aimed to avoid unecessary recomputation for the candidates.
        '''
        print('Compute_measures_at_once method class Performance.')
#         short_path_metrics(value_ref,set_of_clusters,cc_type_process='incremental'):
        spm_metric_list = ['mean','z_score']
        smp_ref_list = ['rows','cost','combine']
        smp_z_thres_list = [-1,0,1]

# short_path_metrics(ref,cc_type_analysis,value_ref,set_of_clusters,cc_type_process='incremental',cc_z_threshold=None):
        for cc_type_analysis in spm_metric_list:
            for ref in smp_ref_list:
                if cc_type_analysis == 'mean':
                    print('Process: {}; Metric: {}; Ref: {}'.format(cc_type_process,cc_type_analysis,ref))
#                     candidates_to_remove = bad_candidates(set_of_clusters,cc_type_analysis,ref)
#                     tmp_set_of_candidates = value_ref.copy()
#                     for candidate in candidates_to_remove:#at this point, value_ref is the set of candidates
# #                             del value_ref[candidate]
#                         del tmp_set_of_candidates[candidate]
#                         return tmp_set_of_candidates
                    final_coclusters = short_path_metrics(ref,cc_type_analysis,set_of_candidates,candidates_ref_values,'sample')
                    self.summary_clusters(final_coclusters, map_tid_to_el, trajs_data_dict_list)
                    self.calculate_entropy_purity(file_dataset)
                    self.__store_clustering_statistics(file_dataset,candidates_ref_values,cc_type_analysis,ref)
                else:
                    for cc_z_threshold in smp_z_thres_list:
                        print('Process: {}; Metric: {}; Ref: {}; z_thres: {}'.format(cc_type_process,cc_type_analysis,ref,cc_z_threshold))
#                         candidates_to_remove = bad_candidates(set_of_clusters,cc_type_analysis,ref,cc_z_threshold)
#                         tmp_set_of_candidates = value_ref.copy()
#                         for candidate in candidates_to_remove:#at this point, value_ref is the set of candidates
# #                                 del value_ref[candidate]
#                             del tmp_set_of_candidates[candidate]
#                             return tmp_set_of_candidates
                        final_coclusters = short_path_metrics(ref,cc_type_analysis,set_of_candidates,candidates_ref_values,'sample',cc_z_threshold)
                        self.summary_clusters(final_coclusters, map_tid_to_el,trajs_data_dict_list)
                        self.calculate_entropy_purity(file_dataset)
                        self.__store_clustering_statistics(file_dataset,candidates_ref_values,cc_type_analysis,ref,cc_z_threshold)

#         final_coclusters = short_path_metrics(set_of_candidates,candidates_ref_values,'sample')
#         self.summary_clusters(final_coclusters, map_tid_to_el,trajs_data_dict_list)
#         self.calculate_entropy_purity(file_dataset)
        print('END of measures_at_once method')
    
    def __store_clustering_statistics(self,dataset,candidates_ref_values,cc_type_analysis,ref,cc_z_threshold=''):
        
        if dataset == 'fs_ny_top_users_193.dat':
            self.__scstats_dataset = 'All users'
        elif dataset == 'fs_ny_top_users_81.dat':
            self.__scstats_dataset = 'Top 81 users'
        elif dataset == 'fs_ny_top_users_10.dat':
            self.__scstats_dataset = 'Top 10 users'
        elif (dataset == 'sjgs.dat') or (dataset == 'sjgs2.dat'):
            self.__scstats_dataset = datset.split('.')[0].upper()
        else:
            self.__scstats_dataset = 'Undefined dataset.'

        #list of column names
        self.scs_field_names = ['dataset','metric','cc_reference','num_of_candidates','num_of_clusters',
                                'avg_std_cv_rows','avg_std_cv_cost','num_of_groupped_elements','avg_std_cv_seq_len',
                                'avg_std_cv_relative_rows_compression','avg_std_cv_num_of_users','overall_entropy']
        
        if cc_z_threshold == '':
            self.__scstats_metric = cc_type_analysis
        else:
            self.__scstats_metric = cc_type_analysis+'['+str(cc_z_threshold)+']'
            
        self.__scstats_num_of_cc = len(candidates_ref_values)
        self.__scstats_avg_rows = np.round(np.mean(self.num_of_trajs_per_cluster),3)
        self.__scstats_std_rows = np.round(np.std(self.num_of_trajs_per_cluster),3)
        self.__scstats_cv_rows = np.round((self.__scstats_std_rows/self.__scstats_avg_rows)*100,3)
        self.__scstats_str_rows = str(self.__scstats_avg_rows)+'\u00B1'+str(self.__scstats_std_rows)+'['+str(self.__scstats_cv_rows)+']'
        self.__scstats_avg_cost = np.round(np.mean(self.cost_value_per_cluster),3)
        self.__scstats_std_cost = np.round(np.std(self.cost_value_per_cluster),3)
        self.__scstats_cv_cost = np.round((self.__scstats_std_cost/self.__scstats_avg_cost)*100,3)
        self.__scstats_str_cost = str(self.__scstats_avg_cost)+'\u00B1'+str(self.__scstats_std_cost)+'['+str(self.__scstats_cv_cost)+']'
        self.__scstats_avg_seq_len = np.round(np.mean(self.seq_len_per_cluster),3)
        self.__scstats_std_seq_len = np.round(np.std(self.seq_len_per_cluster),3)
        self.__scstats_cv_seq_len = np.round((self.__scstats_std_seq_len/self.__scstats_avg_seq_len)*100,3)
        self.__scstats_str_seq_len = str(self.__scstats_avg_seq_len)+'\u00B1'+str(self.__scstats_std_seq_len)+'['+str(self.__scstats_cv_seq_len)+']'
        self.__scstats_avg_relative_compress = np.round(np.mean(self.relative_clusters_value),3)
        self.__scstats_std_relative_compress = np.round(np.std(self.relative_clusters_value),3)
        self.__scstats_cv_relative_compress = np.round((self.__scstats_std_relative_compress/self.__scstats_avg_relative_compress)*100,3)
        self.__scstats_str_relative_compress = str(self.__scstats_avg_relative_compress)+'\u00B1'+str(self.__scstats_std_relative_compress)+'['+str(self.__scstats_cv_relative_compress)+']'
        self.__scstats_avg_num_of_users = np.round(np.mean(self.num_of_users_per_cluster),3)
        self.__scstats_std_num_of_users = np.round(np.std(self.num_of_users_per_cluster),3)
        self.__scstats_cv_num_of_users = np.round((self.__scstats_std_num_of_users/self.__scstats_avg_num_of_users)*100,3)
        self.__scstats_str_num_of_users = str(self.__scstats_avg_num_of_users)+'\u00B1'+str(self.__scstats_std_num_of_users)+'['+str(self.__scstats_cv_num_of_users)+']'
        
        
        
        # Dictionary
        self.scs_data = {'dataset':self.__scstats_dataset,'metric':self.__scstats_metric,'cc_reference':ref,
                     'num_of_candidates':self.__scstats_num_of_cc,'num_of_clusters':len(self.perf_cc_clusters),
                     'avg_std_cv_rows':self.__scstats_str_rows,
                     'avg_std_cv_cost':self.__scstats_str_cost,
                     'num_of_groupped_elements':len(self.unique_elements_grouped),
                     'avg_std_cv_seq_len':self.__scstats_str_seq_len,
                     'avg_std_cv_relative_rows_compression':self.__scstats_str_relative_compress,
                     'avg_std_cv_num_of_users':self.__scstats_str_num_of_users,
                     'overall_entropy':self.overall_entropy}
        
        with open('./coclustering_file_outputs/df_clustering_stats.csv','a+',newline='',encoding='utf8') as self.fscs_object:
      
            # Pass the file object and a list 
            # of column names to DictWriter()
            # You will get a object of DictWriter
            self.scs_dictwriter_object = DictWriter(self.fscs_object, delimiter=";", fieldnames=self.scs_field_names)

            #Pass the dictionary as an argument to the Writerow()
            self.scs_dictwriter_object.writerow(self.scs_data)

            #Close the file object
            self.fscs_object.close()
    
    def set_variables(self,num_objs):
        self.perf_df_clustering_output_measures = pd.DataFrame(columns = ['Iteration_i','Candidate_iteration_k',
                                                                          'Candidate_cost'])
        self.total_num_of_objs_df = num_objs
    
    ### descontinuado
    def append_result(self,it_i,cc_it_k,cc_cost):
        self.perf_df_clustering_output_measures = self.perf_df_clustering_output_measures.append({'Iteration_i':int(it_i),
                                                                                        'Candidate_iteration_k':'Candidate_'+str(cc_it_k),
                                                                                        'Candidate_cost':float(cc_cost)},
                                                                                        ignore_index=True)
    def plot_cost(self):
        '''
        Method to show the cost function value along the iterations.
        '''
#         print(self.df_clustering_output_measures.head())
#         self.df_clustering_output_measures['Cocluster_cost'] = self.df_clustering_output_measures['Cocluster_cost'] / self.df_clustering_output_measures['Cocluster_cost'].abs().max()
#         print(self.df_clustering_output_measures.head())
        a4_dims = (11.7, 8.27)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.lineplot(data=self.perf_df_clustering_output_measures, x="Iteration_i", y="Candidate_cost"
                     , hue="Candidate_iteration_k")#, style="Cluster_iteration_k", markers=True, dashes=False)
        
        if self.perf_df_clustering_output_measures['Candidate_iteration_k'].nunique() > 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#         self.plt.show()

    def summary_clusters(self, cc_dict, map_id_to_att,trajs_data_dict_list):
        '''
        Method to map back the attributes to its original value and put it avaible as final result visualization.
        '''
        self.perf_cc_clusters = {}
        self.num_of_trajs_per_cluster = []
        self.cost_value_per_cluster = []
        self.seq_len_per_cluster = []
        self.unique_elements_grouped = []
        
        for cluster_k,value in cc_dict.items():
#             print('Cluster ',cluster_k)
            remap_seq_output = []
            for att_id in cc_dict[cluster_k]['cc_atts'].split('-'):
                remap_seq_output.append(map_id_to_att[att_id])
            
            for el in remap_seq_output:
                if el not in self.unique_elements_grouped:
                    self.unique_elements_grouped.append(el)
            
            self.seq_len_per_cluster.append(len(remap_seq_output))
            sequence = '-'.join(remap_seq_output).strip()
            self.num_of_trajs_per_cluster.append(len(cc_dict[cluster_k]['cc_objs']))
            self.cost_value_per_cluster.append(cc_dict[cluster_k]['cc_cost'])
            
            self.perf_cc_clusters.update({cluster_k:{'cc_atts':sequence,'cc_objs':cc_dict[cluster_k]['cc_objs'],
                                                     'cc_cost':cc_dict[cluster_k]['cc_cost'],
                                                     'cc_over_coef':cc_dict[cluster_k]['cc_over_coef']}})
#             self.set_of_objects = self.set_of_objects.union(set(cc_dict[cluster_k]['cc_objs']))
        
#         self.perf_cc_clusters.update({'num_of_objects': len(self.set_of_objects)})
        print('### Clustering statistics ###')
        print('Number of co-clusters: ',len(self.perf_cc_clusters))
        self.__avg_rows = np.round(np.mean(self.num_of_trajs_per_cluster),3)
        self.__std_rows = np.round(np.std(self.num_of_trajs_per_cluster),3)
        self.__cv_rows = np.round((self.__std_rows/self.__avg_rows)*100,3)
        self.__avg_cost = np.round(np.mean(self.cost_value_per_cluster),3)
        self.__std_cost = np.round(np.std(self.cost_value_per_cluster),3)
        self.__cv_cost = np.round((self.__std_cost/self.__avg_cost)*100,3)
        print('AVG rows:{:.2f}\u00B1{:.2f}[CV:{:.2f}%], AVG cost:{:.2f}\u00B1{:.2f}[CV:{:.2f}%]'.format(self.__avg_rows,
                                                                                                        self.__std_rows,
                                                                                                        self.__cv_rows,
                                                                                                        self.__avg_cost,
                                                                                                        self.__std_cost,
                                                                                                        self.__cv_cost))
        self.__avg_seq_len = np.round(np.mean(self.seq_len_per_cluster),3)
        self.__std_seq_len = np.round(np.std(self.seq_len_per_cluster),3)
        self.__cv_seq_len = np.round((self.__std_seq_len/self.__avg_seq_len)*100,3)
        print('AVG sequece length:{:.2f}\u00B1{:.2f}[CV:{:.2f}%]'.format(self.__avg_seq_len,
                                                                         self.__std_seq_len,self.__cv_seq_len))
        print('Number of unique elements grouped: '+str(len(self.unique_elements_grouped)))
        
        
        if VERBOSE:
            for key, value in self.perf_cc_clusters.items():
                print('Co-cluster-{}, Sequence: {}, Num of trajs: {}, Cost: {}'.format(key,
                                                                           self.perf_cc_clusters[key]['cc_atts'],
                                                                           len(self.perf_cc_clusters[key]['cc_objs']),
                                                                           self.perf_cc_clusters[key]['cc_cost']))
#         self.perf_cc_clusters.update({'num_of_objects': len(trajs_data_dict_list)})
    
    def get_clusters(self):
        '''
        Method to show the found co-clusters as follows:
        1. It shows the current co-cluster K with the absolute number of objects into it and the relative number regarding
        the total number of objects in the dataset;
        2. It shows the co-cluster sequence of elements and the objects containing it.
        '''
        self.__it_k = 0
        for cluster_k, value in self.perf_cc_clusters.items():
            if cluster_k != 'num_of_objects':
                self.__it_k += 1
                self.relative = len(self.perf_cc_clusters[cluster_k]['cc_objs'])/self.total_num_of_objs_df
                print('Cluster #'+str(self.__it_k)+' - Candidate {0} [Absolute:{1} | Relative:{2:2.2f} | Cost: {3:} | Ov_coef: {4:1.2f} | Seq length: {5}]'.format(
                                                                                               cluster_k,
                                                                                               len(self.perf_cc_clusters[cluster_k]['cc_objs']),
                                                                                               self.relative,
                                                                                               self.perf_cc_clusters[cluster_k]['cc_cost'],
                                                                                               self.perf_cc_clusters[cluster_k]['cc_over_coef'],
                                                                                               len(self.perf_cc_clusters[cluster_k]['cc_atts'].split('-'))))
                
                if len(self.perf_cc_clusters[cluster_k]['cc_objs']) < 10:
                    print('Attributes sequence "{}" and trajectories "{}".'.format(self.perf_cc_clusters[cluster_k]['cc_atts'],
                                                                                   str(self.perf_cc_clusters[cluster_k]['cc_objs']).strip('{}')))
                else:
                    print('Attributes sequence "{}" and trajectories "{},[...]".'.format(self.perf_cc_clusters[cluster_k]['cc_atts'],
                                                                                str(list(self.perf_cc_clusters[cluster_k]['cc_objs'])[0:8]).strip('[]')))
                print('')
    
    def create_alluvial_diagram(self):
        # Function to create the CSV file that contains the data to generate the Sankey diagram
        
        if self.__sdst_dataset == 'All users':
            df_traj_user = pd.read_csv('./data/real_application/foursquare_NY/fs_ny_top_users_193.csv', sep=';')
            self.__file_alluvial_csv = './coclustering_file_outputs/df_build_alluvial_top193.csv'
        elif self.__sdst_dataset == 'Top 10 users':
            df_traj_user = pd.read_csv('./data/real_application/foursquare_NY/fs_ny_top_users_10.csv', sep=';')
            self.__file_alluvial_csv = './coclustering_file_outputs/df_build_alluvial_top10.csv'
        elif self.__sdst_dataset == 'Top 81 users':
            df_traj_user = pd.read_csv('../data/real_application/foursquare_NY/fs_ny_top_users_81.csv', sep=';')
            self.__file_alluvial_csv = './coclustering_file_outputs/df_build_alluvial_top81.csv'
        elif (self.__sdst_dataset == 'SJGS') or (self.__sdst_dataset == 'SJGS2'):
            df_traj_user = pd.read_csv('./data/real_application/gene_sequences/SJGS/splice_data.csv', sep=';')
            self.__file_alluvial_csv = './coclustering_file_outputs/df_build_alluvial_'+str(self.__sdst_dataset.lower())+'.csv'
    
    #     df_traj_user.drop(columns=['tid','lat_lon','time','day','type','root_type','rating','weather'],inplace=True)
        df_traj_user = df_traj_user[['new_tid','label']]
        #     print(df_traj_user.head())

        traj_id = 50
        user_label = df_traj_user[df_traj_user['new_tid'] == traj_id]['label'].unique()[0]
        print('Trajectory "{}" belongs to User: "{}"'.format(traj_id,user_label))
        
        self.__max_seq_len = 0
        for cluster_k, value in self.perf_cc_clusters.items():
            self.__tmp = len(self.perf_cc_clusters[cluster_k]['cc_atts'].split('-'))
            if self.__tmp > self.__max_seq_len:
                self.__max_seq_len = self.__tmp
        
        print('Max sequence length: '+str(self.__max_seq_len))
        
        self.__levels = ['lvl'+str(i) for i in range(1,self.__max_seq_len+3)]
        print('Levels: '+str(self.__levels))
        self.__columns_df_alluvial = self.__levels
        self.__columns_df_alluvial.append('count')
        self.__columns_df_alluvial.append('cluster')
        print('Columns alluvial df: '+str(self.__columns_df_alluvial))
        
        self.__df_alluvial = pd.DataFrame(columns = self.__columns_df_alluvial)    
        self.__df_alluvial.to_csv(self.__file_alluvial_csv,index=False,sep=';')
        
        self.__alluvial_field_names = self.__columns_df_alluvial
        self.__iter_k = 0
        for cluster_k, value in self.perf_cc_clusters.items():       
            self.__iter_k += 1
            self.__tmp_seq = self.perf_cc_clusters[cluster_k]['cc_atts'].split('-')
            self.__alluvial_data = dict.fromkeys(self.__columns_df_alluvial)
            for traj_id in self.perf_cc_clusters[cluster_k]['cc_objs']:
                user_label = df_traj_user[df_traj_user['new_tid'] == int(traj_id)]['label'].unique()[0]
                self.__alluvial_data['lvl1'] = 'U-'+str(user_label)
                self.__alluvial_data['lvl2'] = traj_id
                
                for element_level in range(0,len(self.__tmp_seq)):
                    self.__alluvial_data['lvl'+str(element_level+3)] = self.__tmp_seq[element_level]
                self.__alluvial_data['count'] = 1
                self.__alluvial_data['cluster'] = 'Cluster-'+str(self.__iter_k)
                
                with open(self.__file_alluvial_csv, 'a+', newline='') as self.__alluvial_object:
      
                    # Pass the file object and a list 
                    # of column names to DictWriter()
                    # You will get a object of DictWriter
                    self.__alluvial_dictwriter_object = DictWriter(self.__alluvial_object, delimiter=";",
                                                                   fieldnames=self.__alluvial_field_names)

                    #Pass the dictionary as an argument to the Writerow()
                    self.__alluvial_dictwriter_object.writerow(self.__alluvial_data)

                    #Close the file object
                    self.__alluvial_object.close()
        
        ### plot alluvial ###
        '''It is ploted out of this process. Check the supplement material.'''
#         self.__df_alluvial = pd.read_csv(self.__file_alluvial_csv,sep=';')
#         self.__fig = self.__genSankey(self.__df_alluvial,cat_cols=self.__levels,value_cols='count')
#         plotly.offline.plot(self.__fig, validate=False)
            
    def __genSankey(self,df,cat_cols=[],value_cols='',title='Sankey Diagram'):
        # old generate sankey. it works well with no repeated variable values
        self.__dpi=300
        self.__board = plt.figure(figsize=(3, 2),dpi=self.__dpi)
        # maximum of 6 value cols -> 6 colors
#         colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
        # maximum of 20 value cols -> 20 colors
        colorPalette = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
                        '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
                        '#bcbd22', '#dbdb8d','#17becf', '#9edae5']
        labelList = []
        colorNumList = []
        for catCol in cat_cols:
            labelListTemp =  list(set(df[catCol].values))
            colorNumList.append(len(labelListTemp))
            labelList = labelList + labelListTemp

        # remove duplicates from labelList
        labelList = list(dict.fromkeys(labelList))

        # define colors based on number of levels
        colorList = []
        for idx, colorNum in enumerate(colorNumList):
            colorList = colorList + [colorPalette[idx]]*colorNum

        # transform df into a source-target pair
        for i in range(len(cat_cols)-1):
            if i==0:
                sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
                sourceTargetDf.columns = ['source','target','count']
            else:
                tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
                tempDf.columns = ['source','target','count']
                sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
            sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()

        # add index for source-target pair
        sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
        sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))

        # creating the sankey diagram
        data = dict(
            type='sankey',
            node = dict(
              pad = 10,
              thickness = 15,
              line = dict(
                color = "black",
                width = 0.5
              ),
              label = labelList,
              color = colorList
            ),
            link = dict(
              source = sourceTargetDf['sourceID'],
              target = sourceTargetDf['targetID'],
              value = sourceTargetDf['count']
            )
          )

        layout =  dict(
            title = title,
#             height = 372,
#             width = 550,
            font = dict(
              size = 10
            )
        )

    #     fig = go.Figure(data = [go.Sankey(data,layout)])
#         fig = go.Figure(data = [go.Sankey(data)])
        fig = dict(data=[data], layout=layout)
        return fig
    
    def store_dist(self,set_of_candidates):
        self.set_of_candidates = set_of_candidates
    
    def __get_entropy_purity(self):
        print('Overall entropy H: '+str(self.overall_entropy))
        print('Purity: '+str(self.purity))
        self.__gep_avg_relative = np.round(np.mean(self.relative_clusters_value),3)
        self.__gep_std_relative = np.round(np.std(self.relative_clusters_value),3)
        self.__gep_cv_relative = np.round((self.__gep_std_relative/self.__gep_avg_relative)*100,3)
        print('AVG relative co-clusters: {:.2f}\u00B1{:.2f}[CV:{:.2f}%]'.format(self.__gep_avg_relative,
                                                                                self.__gep_std_relative,
                                                                                self.__gep_cv_relative))
        self.__gep_avg_num_users = np.round(np.mean(self.num_of_users_per_cluster),3)
        self.__gep_std_num_users = np.round(np.std(self.num_of_users_per_cluster),3)
        self.__gep_cv_num_users = np.round((self.__gep_std_num_users/self.__gep_avg_num_users)*100,3)
        print('AVG num. of users: {:.2f}\u00B1{:.2f}[CV:{:.2f}%]'.format(self.__gep_avg_num_users,
                                                                         self.__gep_std_num_users,
                                                                         self.__gep_cv_num_users))
        print('')
    
    def calculate_entropy_purity(self, file_dataset):
        
        self.split = file_dataset.split('.')
        if self.split[0] != 'sjgs' or self.split[0] != 'sjgs2' or split[0] != 'splice_data':
            if self.split[-1] == 'dat':
                self.path = './data/real_application/foursquare_NY/concat_dimensions/'
                self.df_trajs_users = create_df_map_traj_user(pd.read_csv(self.path+self.split[0]+'.csv', sep=";"))
            else:
                self.path = './data/real_application/foursquare_NY/concat_dimensions/'
                self.df_trajs_users = create_df_map_traj_user(pd.read_csv(self.path+file_dataset, sep=";"))
        else:
            if self.split[-1] == 'dat':
                self.path = './data/real_application/gene_sequences/SJGS/'
                self.df_trajs_users = create_df_map_traj_user(pd.read_csv(self.path+self.split[0]+'.csv', sep=";"))
            else:
                self.path = './data/real_application/gene_sequences/SJGS/'
                self.df_trajs_users = create_df_map_traj_user(pd.read_csv(self.path+file_dataset, sep=";"))
        
        self.relative_clusters_value = []
        self.entropy_per_cluster = []
        self.num_of_objs_per_cluster = []
        self.total_num_of_objs = len(self.df_trajs_users)
        self.overall_entropy = 0
        self.max_prob_per_cluster = []
        self.purity = 0
        self.num_of_users_per_cluster = []
        
        for cluster_k, value in self.perf_cc_clusters.items():
            self.users = {}
            if cluster_k != 'num_of_objects':
                self.num_of_objs_k = len(self.perf_cc_clusters[cluster_k]['cc_objs'])
                self.relative_clusters_value.append((self.num_of_objs_k/self.total_num_of_objs)*100)
#                 print('Cluster-'+str(cluster_k)+' | # of trajs: '+str(self.num_of_objs_k))
                self.trajs = list(map(int,self.perf_cc_clusters[cluster_k]['cc_objs']))
                self.df_cluster = self.df_trajs_users[self.df_trajs_users['Tid'].isin(self.trajs)]
##                 self.users = self.df_cluster['User'].value_counts().to_dict()
                self.users = self.df_cluster['User'].value_counts()
                self.num_of_users_per_cluster.append(len(self.users))
                self.h_k = entropy(self.users,base=2)
##                 self.users = self.users.to_dict()
                #print(self.users,end=' | ')
                #print("Entropy h_k: "+str(self.h_k))
                self.entropy_per_cluster.append(self.h_k)
                self.num_of_objs_per_cluster.append(self.num_of_objs_k)
##                 self.total_num_of_objs += self.num_of_objs_k
##                self.max_prob_per_cluster.append(np.array(list(self.users.values())).max())
                self.max_prob_per_cluster.append(list(self.users)[0])
        
        self.overall_entropy = np.sum((np.array(self.entropy_per_cluster)*
                                       (np.array(self.num_of_objs_per_cluster)/self.total_num_of_objs)))
        self.purity = np.array(self.max_prob_per_cluster).sum()/self.total_num_of_objs
        self.__get_entropy_purity()
    
    def show_boxplot(self):
        '''
        Method to show the distribution values of the candidates.
        '''
        #         array = np.random.uniform(size=20)
        self.array = list(self.set_of_candidates.values())
        self.ref = ''
        if np.mean(self.array) < 0:
            self.ref = 'Cost ref'
        else:
            self.ref = 'Rows ref'
        ax = sns.boxplot(data=self.array)
        ax = sns.swarmplot(data=self.array, color=".25")
        plt.xticks([0],[self.ref])
#         plt.xlabel("Reference")
        plt.ylabel("Values")
        plt.title(self.ref+" distribution")
    #     plt.show(ax)
    
    def test_norm_dist(self):
        import scipy.stats as stats
        try:
            self.mean = np.mean(list(self.set_of_candidates.values()))
            self.std = np.std(list(self.set_of_candidates.values()),ddof=1)
            self.dist_values = list(self.set_of_candidates.values())
#             self.shapiro_stat, self.shapiro_p_value = stats.shapiro(self.dist_values)
#     #         print('O valor da estatística de shapiro-wilk = '+str(self.shapiro_stat))
#     #         print('O valor do p-value de shapiro-wilk = '+str(self.shapiro_p_value))
#             if self.shapiro_p_value >=0.5:
#                 print('Com 95% de confiança, os dados são similares a uma distribuição normal segundo o teste de Shapiro-Wilk.')
#             else:
#                 print('Com 95% de confiança, os dados NÃO são similares a uma distribuição normal segundo o teste de Shapiro-Wilk.')
#             print('')
            self.__shapiro_wilk_test()
            self.__kolomogorov_smirnov_test()
            self.__anderson_darling_test()
            print('')
        except Exception as inst:
            print('Please, test the variable individually.')
            print('Error:',inst)
#             self.dist_values_rows = []
#             self.dist_values_cost = []

#             for key,value in self.set_of_candidates.items():
#                 self.dist_values_rows.append(self.set_of_candidates[key]['rows'])
#                 self.dist_values_cost.append(self.set_of_candidates[key]['cost'])
            
#             self.shapiro_stat_rows, self.shapiro_p_value_rows = stats.shapiro(self.dist_values_rows)
#             self.shapiro_stat_cost, self.shapiro_p_value_cost = stats.shapiro(self.dist_values_cost)
    
    def __shapiro_wilk_test(self):
        self.shapiro_stat, self.shapiro_p_value = stats.shapiro(self.dist_values)
#         print('O valor da estatística de shapiro-wilk = '+str(self.shapiro_stat))
#         print('O valor do p-value de shapiro-wilk = '+str(self.shapiro_p_value))
        if self.shapiro_p_value >=0.5:
            print('Segundo o teste de Shapiro-Wilk, com 95% de confiança, os dados são similares a uma distribuição normal.')
        else:
            print('Segundo o teste de Shapiro-Wilk, com 95% de confiança, os dados NÃO são similares a uma distribuição normal.')
    
    def __kolomogorov_smirnov_test(self):
        self.ks_stat, self.ks_p_value = stats.kstest(self.dist_values,cdf='norm', args=(self.mean,self.std), N=len(self.dist_values))
        self.ks_critico = self.__kolmogorov_smirnov_critico(len(self.dist_values))
        if self.ks_critico >= self.ks_stat:
            print('Segundo o teste Kolomogorov-Smirnov, com 95% de confiança, os dados são similares a uma distribuição normal.')
        else:
            print('Segundo o teste Kolomogorov-Smirnov, com 95% de confiança, os dados NÃO são similares a uma distribuição normal.')
    # Checking the critical value of the Kolmogorov-Smirnov test
    def __kolmogorov_smirnov_critico(self,n):
        # table of critical values for the kolmogorov-smirnov test - 95% confidence
        # Source: https://www.soest.hawaii.edu/GG/FACULTY/ITO/GG413/K_S_Table_one_Sample.pdf
        # Source: http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
        # alpha = 0.05 (95% confidential level)

        if n <= 40:
            # valores entre 1 e 40
            self.kolmogorov_critico = [0.97500, 0.84189, 0.70760, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925, 
                          0.39122, 0.37543, 0.36143, 0.34890, 0.33760, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408, 
                          0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25907, 0.25438, 0.24993, 0.24571, 0.24170, 
                          0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012]
            self.ks_critico = self.kolmogorov_critico[n - 1]
        elif n > 40:
            # valores acima de 40:
            self.kolmogorov_critico = 1.36/(np.sqrt(n))
            self.ks_critico = self.kolmogorov_critico
        else:
            pass            

        return self.ks_critico
    
    def __anderson_darling_test(self):
        self.ad_stat, self.ad_critico, self.ad_teorico = stats.anderson(self.dist_values,'norm')
        if self.ad_stat < self.ad_critico[2]:
            print('Segundo o teste de Anderson-Darling, com 95% de confiança, os dados são similares a uma distribuição normal.')
        else:
            print('Segundo o teste de Anderson-Darling, com 95% de confiança, os dados NÃO são similares a uma distribuição normal.')

    def test_skewness(self):
        self.dist_values = list(self.set_of_candidates.values())
        self.mean = np.mean(self.dist_values)
        self.median = np.median(self.dist_values)
        vals,counts = np.unique(self.dist_values, return_counts=True)
        index = np.argmax(counts)
        self.mode = vals[index]
        
        
        if (self.mean == self.median) and (self.mean == self.mode):
            print('Distribuição normal | Mean:{} = Median: {} = Mode: {}.'.format(self.mean,self.median,self.mode))
        #positive values
        if (self.mean < self.median) and (self.median < self.mode):
            print('Assimetria à Esquerda (negativa) | Mean:{} < Median: {} < Mode: {}.'.format(self.mean,self.median,self.mode))
        if (self.mode < self.median) and (self.median < self.mean):
            print('Assimetria à Direita (positiva) | Mode:{} < Median: {} < Mean: {}.'.format(self.mode,self.median,self.mean))
        #negativa values
        if self.mean < 0:
            print('mean:',self.mean,' median:',self.median,' mode:',self.mode)
            if (self.mean > self.median) and (self.median > self.mode):
                print('Assimetria à Esquerda (negativa) | Mean:{} < Median: {} < Mode: {}.'.format(self.mean,self.median,self.mode))
            if (self.mode > self.median) and (self.median > self.mean):
                print('Assimetria à Direita (positiva) | Mode:{} < Median: {} < Mean: {}.'.format(self.mode,self.median,self.mean))
        print('')



#### Modificação do candidate deviation método. Esta modificação é para fazer os cálculos das medidas sem ter que
#### recalcular os candidatos no mesmo dataset. Os candidatos não mudam no modo de descoberta automática.
# def short_path_metrics(ref,value_ref,set_of_clusters,cc_type_process='incremental'):
def short_path_metrics(ref,cc_type_analysis,value_ref,set_of_clusters,cc_type_process='incremental',cc_z_threshold=None):
    '''
    Method to return the avg number of the reference in the set of co-clusters.
    If the set is bigger than 1 it calculates the avg, otherwise it is 0.
    Parameters:
        ref_analysis: 1. index_rows_set -> considers the rows; 2. cost_function -> considers the cost.
        test_value: The value to test.
        set_of_clusters: The current set of co-clusters containing its values for the ref_analysis    
    '''
   
    if len(set_of_clusters) >= 2:
        try:# single ref
            mean = np.mean(list(set_of_clusters.values()))
            std = np.std(list(set_of_clusters.values()))
        except:# double ref
            sum_rows = []
            sum_cost = []
            for key,value in set_of_clusters.items():
                sum_rows.append(set_of_clusters[key]['rows'])
                sum_cost.append(set_of_clusters[key]['cost'])
            mean_rows = np.mean(sum_rows)
            mean_cost = np.mean(sum_cost)
            std_rows = np.std(sum_rows)
            std_cost = np.std(sum_cost)
        
        if cc_type_process == 'incremental':
            if ref == "rows":
                ### normal mean
                if cc_type_analysis == 'mean':
                    return len(value_ref['index_rows_set']) >= np.floor(mean)
                else:
                ### z-score: we consider values greater than -1 once it is a positive distribution
                    try:
                        z = (len(value_ref['index_rows_set'])-mean)/std
                    except:
                        z = (value_ref-mean)/std
                    print('Z-score(rows): ',z)
                    return z >= cc_z_threshold
            elif ref == "cost":
                ### normal mean
                if cc_type_analysis == 'mean':
                    return value_ref['cost_function'] <= np.ceil(mean)
                else:
                ### z-score: we consider values smaller than 1 once it is a negative distribution
                    try:
                        z = (value_ref['cost_function']-mean)/std
                    except:
                        z = (value_ref-mean)/std
                    print('Z-score(cost): ',z)
                    return z <= -cc_z_threshold
            else:#combine
                ### normal mean
                if cc_type_analysis == 'mean':
#                     print('Mean(rows):',mean_rows,' Mean(cost):',mean_cost)
                    return ((len(value_ref['index_rows_set']) >= np.floor(mean_rows)) or 
                            (value_ref['cost_function'] <= np.ceil(mean_cost)))
                else:
                ### z-score
                    z_rows = (len(value_ref['index_rows_set'])-mean_rows)/std_rows
                    z_cost = (value_ref['cost_function']-mean_cost)/std_cost
#                     print('Z-score(rows): ',z_rows,' Z-score(cost): ',z_cost)
                    return ((z_rows >= -cc_z_threshold) or (z_cost <= cc_z_threshold))
                
        elif cc_type_process == 'sample':
            candidates_to_remove = []
            
            if ref != 'combine':
#             try:# single ref: rows OR cost
#                 mean
                if cc_type_analysis == 'mean':
                    for key,value in set_of_clusters.items():
#                         print('Candidate-'+key+' Mean:',mean,' Value ref:',value,end='')
                        if ref == 'rows' and set_of_clusters[key]['rows'] < mean_rows:
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        elif ref == 'cost' and set_of_clusters[key]['cost'] > mean_cost:
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        else:
#                             print(' -> Keep')
                            pass
                else:#z-score
                    for key,value in set_of_clusters.items():
#                         z = (value-mean)/std
#                         print('Candidate-'+key+' Z-score:',z,end='')
                        if ref == 'rows':
                            z = (set_of_clusters[key]['rows']-mean_rows)/std_rows
                            if z < cc_z_threshold:
                                candidates_to_remove.append(key)
#                             print(' -> Remove')
                        elif ref == 'cost':
                            z = (set_of_clusters[key]['cost']-mean_cost)/std_cost
                            if z > -cc_z_threshold:
                                candidates_to_remove.append(key)
#                             print(' -> Remove')
                        else:
#                             print(' -> Keep')
                            pass

            else:
#             except:#double ref combine: rows AND cost
                
                if cc_type_analysis == 'mean':
                    for key,value in set_of_clusters.items():
#                         print('Candidate-'+key+' Mean(rows):',mean_rows,' Mean(cost):',mean_cost,end='')
                        if (set_of_clusters[key]['rows'] < mean_rows) and (set_of_clusters[key]['cost'] > mean_cost):
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        else:
#                             print(' -> Keep')
                            pass

                else:#z-score
                    for key,value in set_of_clusters.items():
                        z_rows = (set_of_clusters[key]['rows']-mean_rows)/std_rows
                        z_cost = (set_of_clusters[key]['cost']-mean_cost)/std_cost
#                         print('Candidate-'+key+' Z-score(row):',z_rows,' Z-score(cost):',z_cost,end='')
                        if (z_rows < cc_z_threshold) and (z_cost > -cc_z_threshold):
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        else:
#                             print(' -> Keep')
                            pass
            
#             return candidates_to_remove  
#             print("Remove candidates: ",candidates_to_remove)
#             print("Number of candidates to remove: ",len(candidates_to_remove))
            tmp_set_of_candidates = value_ref.copy()
            for candidate in candidates_to_remove:#at this point, value_ref is the set of candidates
#                 del value_ref[candidate]
                del tmp_set_of_candidates[candidate]
            return tmp_set_of_candidates

        else: # just pass step. Storing the candidate co-clusters to analyze them with sample analysis if desirable
            return True               
                
    else:# pass step to reach a minimum number of elements to perform computation
        return True
    
def show_boxplot():
    array = np.random.uniform(size=20)
    ax = sns.boxplot(data = array)
    ax = sns.swarmplot(data=array, color=".25")
    plt.xticks([0],['SDF'])
    plt.xlabel("Reference")
    
def sort_attributes(data_res):
    
    try:
        ##usar este for caso o value seja uma lista
        freq_res_dict = {}
        for key,value in data_res.items():
            freq_res_dict[key] = len(value)

        # Create a list of tuples sorted by index 1 i.e. value field     
        listofTuples = sorted(freq_res_dict.items() , reverse=True, key=lambda x: x[1])# usar se value for lista
        # Iterate over the sorted sequence
        # for elem in listofTuples :
        #     print(elem[0] , " ::" , elem[1] )
    #     print(listofTuples)
        sorted_attributes = [elem[0] for elem in listofTuples]
    except:
        ## este é usado caso value seja um número
        sorted_attributes = {k: v for k, v in sorted(data_res.items(), reverse=True, key=lambda item: item[1])}
    
#     if VERBOSE:
#         print("Sorted att: ",sorted_attributes)
    return sorted_attributes

def update_residual_dataset(res_data, attributes_cocluster, objects_cocluster):
    for key, value in res_data.items():
        if key in attributes_cocluster:
            diff_objs = set(res_data[key]).difference(set(objects_cocluster))
            res_data[key] = list(diff_objs)
    return res_data

def format_time_output(time_in_sec):
    ''' 
    This function converts the seconds for the format Hours:Minutes:Seconds.
    '''
    hours = np.floor((time_in_sec/3600))
    mins = np.floor((time_in_sec - (hours*3600))/60)
    secs = np.floor(time_in_sec%60)
#     print(str(int(hours))+'h:'+str(int(mins))+'m:'+str(int(secs))+'s')
    
    return str(int(hours))+'h:'+str(int(mins))+'m:'+str(int(secs))+'s'

# this function converts the seconds for the format in minutes
def format_time_minutes(time_in_sec):
    return np.round(time_in_sec/60,2)
#     return time_in_sec/60

def writeFileOutput(co_clusters, dataset, method='OCoClus', fileName='OCoClusResult'):
    text = ""
#    for c in range(len(data.rows_)):
#        res = [i for i, val in enumerate(data.columns_[c]) if val]
#        for j in res:
#            text += str(j)+" "

#        res = [i for i, val in enumerate(data.rows_[c]) if val]
#        text += "["
#        for j in res:
#            text += str(j)+" "
#        text += "]\n"
    
    num_of_clusters = len(co_clusters)
    
#     for c in range(len(cols)):
    for c in range(num_of_clusters):
#         for i in cols[c]:
        for i in co_clusters[c][0]: # get the attributes in cluster c
            text += str(i)+" "
        
        text += "("+str(len(co_clusters[c][1]))+") [" # get the number of objects in clusters c
        for j in range(len(co_clusters[c][1])): # save in the file each obj
            if j+1 != len(co_clusters[c][1]):
                text += str(co_clusters[c][1][j])+" "
            else:
                text += str(co_clusters[c][1][j])
        text += "]\n"
    
    #print(text)
    if method == 'Dhillon':
        f = open('./datasets/outputs/'+fileName+'.txt', 'w+')#saving at dataset folder
        f.write(text)
        f.close()
        print("Output file saved in: "+"./datasets/outputs/"+fileName+".txt")
    elif method == 'Kluger':
        f = open('./datasets/outputs/'+fileName+'.txt', 'w+')#saving at dataset folder
        f.write(text)
        f.close()
        print("Output file saved in: "+"./datasets/outputs/"+fileName+".txt")
    elif method == 'OCoClus':
        f = open('./OutputAnalysis/ococlus/'+dataset+'/'+fileName+'.txt', 'w+')#saving at dataset folder
        f.write(text)
        f.close()
        print("Output file saved in: "+"./OutputAnalysis/ococlus/"+dataset+"/"+fileName+".txt")
    else:
        print("The output file was not generated. Method option not recognized.")

def processDocument(file):
    #print("Executing processDocument() method.")
    size_df = file.shape
    uniqueP = []
    uniqueT = []
    pois_per_cluster = []
    trajs_per_cluster = []
    change = 0
    
    for i in range(size_df[0]):#number of transations
        transationString = file.transation[i].split(' ')
#         print(transationString)
        pois = []
        trajs = []
        for j in range(len(transationString)):
            try:
                tmp = int(transationString[j])
#                 print(transationString[j])
            except:
                tmp = ''
            
            if change == 0:
                if type(tmp) is int:
                    if transationString[j] not in uniqueP:
                        uniqueP.append(transationString[j])
#                         if not pois: # empty
#                     print(transationString[j])
                    pois.append(transationString[j])
#                         print(transationString[j],end="|")
                else:
    #                 print("NO",end=" | ")
                    change = 1
#                     print("")
            else:
                if type(tmp) is int:
                    if transationString[j] not in uniqueT:
                        uniqueT.append(transationString[j])
#                     print(transationString[j])
                    trajs.append(transationString[j])
                else:
    #                 print("NO",end=" | ")
                    if change == 1:
                        change = 2
                        # entre colchetes ficam os indices das linhas que contem os atributos da parte anterior
                        tmp = re.split('\[| |\]',transationString[j])# split para separar primeiro colchete e o núm: [5
#                         print("1-tmp:"+str(tmp),end=", ")
#                         print(tmp[0])
                        if len(tmp) > 1: # tmp=['(12)'], bypass para este caso; normal: tmp= ['','343'] ou ['','343','']
                            if tmp[1] not in uniqueT:
                                uniqueT.append(tmp[1])
#                             print(tmp[1])
                            trajs.append(tmp[1])
                    else:
                        tmp = re.split('\[| |\]',transationString[j])
#                         print("2-tmp:"+str(tmp))
                        # etapa para processar entre os cholchetes e finalizar
                        if len(tmp) == 2:# tmp= ['','343'] or ['343','']
                            if tmp[0] == '':
#                                 print(tmp[1])
                                trajs.append(tmp[1])
                            else:
#                                 print(tmp[0])
                                trajs.append(tmp[0])
                        if len(tmp) == 3:
#                             print(tmp[1])
                            trajs.append(tmp[1])
#                         if tmp[0] not in uniqueT:
#                             uniqueT.append(tmp[0])
#                         trajs.append(tmp[0])
                        change = 0
        pois_per_cluster.append(pois)
        trajs_per_cluster.append(trajs)

#     print('Number of unique elements: ',len(uniqueElements))
#     print("P per Clusters: "+str(pois_per_cluster))
#     print("T per Clusters: "+str(trajs_per_cluster))
    return pois_per_cluster,trajs_per_cluster