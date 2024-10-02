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

def get_data(input_data):
    '''
    This method will assign the variables used by the algorithm.
    
    INPUT
        input_data: A panda dataframe of the input data file.
    
    OUTPUT
        D: A binary matrix from the input data.
        N: A noise binary matrix with the same size of D.
        data_dict: A dictionary to store D as a vertical representation.
        data_res_dict: A copy of data_dict used to sort the attributes of D and find unconvered elements.
        
    '''
    
    data_pd = input_data #txt file with sequence of check-ins (POI)
    frequence_per_poi_dict = {} # store the frequence of a POI as "POI": num_of_occurrences
    poi_at_trajs_dict_set = {}  # store a set with each index line (tid trajectory) that contains a given POI.
                            # "POI": set(0,1,4,...); It is the S variable
#     global data_res_dict
    uncover_poi_dict = {} # It is the s* variable
#     global D # input data as a binary matrix
#     global N # noise matrix with the same size of D
    num_of_objects = 0
    num_of_attributes = 0
    map_id_to_attribute = {} # map the 
    map_attribute_to_id = {} # map the
    trajectory_dict = {} # it stores the trajectories with its check-ins. "TID": [POI1,POI2,...]
#     max_val_att = 0 
    att_id = 0 # assign an ID to each attribute
    
    # read each line
    for index, row in data_pd.iterrows():
        num_of_objects+=1
        object_data = row[0].split(" ")
#         trajectory_dict[str(index)] = {}
#         trajectory_dict[str(index)] = object_data
        trajectory_dict[str(index)] = []
        
#         for attribute in object_data: # we look at each item of the given transaction
        for att_j in range(len(object_data)): # we look at each item of the given transaction
            attribute = object_data[att_j]
            
            if attribute != "":
#                 if int(attribute) > max_val_att:
#                     max_val_att = int(attribute)
#                 if attribute not in map_unique_attributes_dataset:
#                 if attribute not in map_attribute_to_id.keys():
    
                if attribute not in map_attribute_to_id: # mapping
#                     unique_attributes_dataset.append(attribute)
                    map_attribute_to_id[attribute] = str(att_id)
                    map_id_to_attribute[str(att_id)] = attribute
                    att_id += 1
                
                # substitute the check-in by its ID
                trajectory_dict[str(index)].append(map_attribute_to_id[attribute])
                
                # store the indeces containing a given POI
                if map_attribute_to_id[attribute] in poi_at_trajs_dict_set:
#                     data_dict[map_attribute_to_id[attribute]].append(index)
                    poi_at_trajs_dict_set[map_attribute_to_id[attribute]].add(str(index))
                else:
#                     data_dict[map_attribute_to_id[attribute]] = [index]
                    poi_at_trajs_dict_set[map_attribute_to_id[attribute]] = set([str(index)])
                
                # store the frequence for each POI
                if map_attribute_to_id[attribute] in frequence_per_poi_dict:
                    current_value = frequence_per_poi_dict[map_attribute_to_id[attribute]]
                    frequence_per_poi_dict[map_attribute_to_id[attribute]] = current_value + 1
                else:
                    frequence_per_poi_dict[map_attribute_to_id[attribute]] = 1
            
                    
    uncover_poi_dict = poi_at_trajs_dict_set.copy()
#     num_of_attributes = len(data_dict)
#     num_of_attributes = max_val_att+1
#     num_of_attributes = len(map_attribute_to_id)
    print("######################################")
    print("Number of trajectories: "+str(index+1))
    print("Number of unique check-ins: "+str(len(map_attribute_to_id)))
    print("########################################")
    if VERBOSE:
        print("Map_attribute_to_id:"+str(map_attribute_to_id))
        print("")
        print("Map_id_to_attribute:"+str(map_id_to_attribute))
        print("")
        print("Frequence_per_poi:"+str(frequence_per_poi_dict))
        print("")
        print("Trajectories: "+str(trajectory_dict))
        print("")
        print("POI occurring at trajectories: "+str(poi_at_trajs_dict_set))
        print("Get data is DONE!")
        
    
#     D = np.zeros((num_of_objects,num_of_attributes),dtype=int)
#     for key, values in poi_at_trajs_dict.items():
#         print("key:"+str(key)+" Values:"+str(values))
#         for line in values:
# #             D[line][int(key)] = 1
# #             D[line][map_unique_attributes_dataset[key]] = 1
# #             print(line,key)
# #             print(type(line),type(key))
#             D[line][int(key)] = 1
#     N = np.zeros((num_of_objects,num_of_attributes),dtype=int)
    
#     return D, N, poi_at_trajs_dict, data_res_dict, map_id_to_attribute
    return map_id_to_attribute, frequence_per_poi_dict, poi_at_trajs_dict_set, trajectory_dict

# def create_df_map_traj_user(df):
def create_df_map_traj_user(df=pd.DataFrame):
    '''
    Method to support the calculation of the quality result.
    It returns a dataframe with the users and their trajectories with its respective length.
    '''
    try:
        #     df = pd.read_csv('data/real_application/foursquare_NY/fs_ny_top_users_10.csv', sep=";")
        df_map_traj_user = pd.DataFrame(columns=['Tid','Traj_length','User'])
        tids = []
        user = ''
        traj_length = 0

        sequence = []
        past_tid = None
        curr_tid = None
        num_of_seqs = 0
        map_element_id = 0
        unique_elements = {}
        map_id_to_element = {}

        for i in range(len(df)):
            curr_tid = df.loc[i,"new_tid"]
            if curr_tid not in tids:
                tids.append(int(curr_tid))
                user = int(df.loc[i,"label"])
                traj_length = len(df[df['new_tid'] == curr_tid])
                # append rows to an empty DataFrame
                df_map_traj_user = df_map_traj_user.append({'Tid' : curr_tid, 'Traj_length' : traj_length, 'User' : user},ignore_index = True)
        df_map_traj_user['Tid'] = df_map_traj_user['Tid'].astype(int, errors='ignore')
        df_map_traj_user['Traj_length'] = df_map_traj_user['Traj_length'].astype(int, errors='ignore')
        df_map_traj_user['User'] = df_map_traj_user['User'].astype(int, errors='ignore')

    #     print(df_map_traj_user.shape)
    #     print(df_map_traj_user.head())
    #     print('Todo DataFrame (traj_length):',' mean=',df_map_traj_user['Traj_length'].mean(),
    #           ' std=',df_map_traj_user['Traj_length'].std())
    #     u_185 = df_map_traj_user[df_map_traj_user['User']==185]
    #     print(u_185)
    #     print('User 185:',' mean=',u_185['Traj_length'].mean(),' std=',u_185['Traj_length'].std())
    #     df_map_traj_user.groupby(['label']).nunique()['new_tid'].mean()
        r = (df_map_traj_user.groupby(['User'])['Traj_length']
             .agg([np.count_nonzero,np.mean,np.std])
             .rename(columns={'count_nonzero':'Count_trajs',
                              'mean':'AVG_traj_length',
                              'std':'STD_traj_length'}))
    #     print('Número médio de trajs por usuário = {:.2f} com DP = {:.2f}'.format(r['Count_trajs'].mean(),
    #                                                                        r['Count_trajs'].std()))
    #     print('Tamanho médio das trajs por usuário = {:.2f} com DP médio = {:.2f}'.format(r['AVG_traj_length'].mean(),
    #                                                                        r['STD_traj_length'].mean()))
        return df_map_traj_user
    except:
        raise('Please, check the input data format.')
        return None
    
