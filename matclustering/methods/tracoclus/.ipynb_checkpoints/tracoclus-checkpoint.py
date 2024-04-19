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

#https://github.com/bigdata-ufsc/ss-ococlus
# ------------------------------------------------------------------------------------------
#toy example dataset
# input_test2.dat and input_test2.dat

#synthetic datasets
# synthetic-1_fimi.dat, synthetic-2_fimi.dat and synthetic-3_fimi.dat

experiment = 'Toy' # Syn: run the synthetic data analysis; Real: run the real data analysis; 
                   # Toy: run the toy example.
toy_number = "2" # sufix of a given toy dataset [1,1-1,2]

find_overlap = False # True: find overlapped co-clusters; False: Find no overlapped co-clusters
VERBOSE = True # True: print results as a verbose mode; False: print the main results
num_of_sim = 1 # number of simulations to perform
k = -1 #max number of co-cluster that could be found. -1: default [driven by cost function]
e_obj = 0.4 # maximum error tolerance for object. -1: default [accept the maximum error]
e_att = .8 # maximum error tolerance for attribute. -1: default [accept the maximum error]

if VERBOSE: print('--->Verbose mode ON.<---')

print("Executing TRACOCLUS method")
# ------------------------------------------------------------------------------------------


def TRACOCLUS(input_data, avg_cc_analysis, k=-1, e_obj=-1, e_att=-1):    
    input_D = ''
    split = input_data.split('.')
    print(split)
    if (split[0] != 'sjgs') and (split[0] != 'sjgs2') and (split[0] != 'splice_data'):
        if split[-1] == 'dat':
            path = './data/real_application/foursquare_NY/preprocessed/'
            input_D = pd.read_csv(path+input_data, header=None, names=["transation"])
        else:
            path = './data/real_application/foursquare_NY/'
            input_D = pd.read_csv(path+input_data, header=None, names=["transation"])
    else:
        if split[-1] == 'dat':
            path = './data/real_application/gene_sequences/SJGS/preprocessed/'
            input_D = pd.read_csv(path+input_data, header=None, names=["transation"])
        else:
            path = './data/real_application/gene_sequences/SJGS/'
            input_D = pd.read_csv(path+input_data, header=None, names=["transation"])
    
    ### variable declaration
    if k == -1:
        k=sys.maxsize
    if e_obj == -1:
        e_obj = 1
    if e_att == -1:
        e_att = 1
    
    cost_model = sys.float_info.max # initial cost function of the model
    num_of_coclusters = 0
    D = []
    final_coclusters = [] # store the attribute and objects clusters. final_coclusters[[C1_att,C1_obj],[Ck_att,Ck_obj]]
    pattern_model = [set(),set()]# Union between the found co-clusters [list of obj,list of att]
    cost_per_cocluster = []# stores the cost to build the cocluster
    history_cost_model = []
    ###
    
#     D,N,data_dict,data_res_dict,map_id_to_attribute = get_data(input_D)
    
    # Gamma: store the found co-clusters
    overlap_coef_threshold = 0.5
    INITIAL_COST = 100.0
    final_coclusters = {}
    final_clustered_elements = set()
    final_coclustering_cost = 0
    coclustering_sizes_remove = [] # stores the cluster to be removed from final_coclustering_size
#     avg_cc_analysis = "combine" # 1. 'index_rows_set'; 2. 'cost_function'; 3. combine
#     avg_cc_analysis = cc_analysis # 1. 'index_rows_set'; 2. 'cost_function'
    final_coclustering_size = {} # stores the clusters and its num of rows
    candidates_ref_values = {}
    final_coclustering_avg_row_size = 0
    total_of_iterations = 0
#     clustering_perf = Performance(sns,plt)
#     clustering_perf = Performance()

    start_1 = timer()
    # Initialize main data structures
    map_id_to_attribute_dict, S_poi_freq_dict, poi_at_trajs_dict_set, trajs_data_dict_list = get_data(input_D)
    time_p1 = timer()-start_1
    clustering_perf.set_variables(len(trajs_data_dict_list))
#     S_poi_freq_dict = sort_attributes(S_poi_freq_dict)
    
    ### select att-values at most elements by log2 of the length of the set
#     i = num_elements_to_test('log2',len(S_poi_freq_dict))
#     print('Limit log2: ',i)
#     for key, value in S_poi_freq_dict.items():
#         if i < 0:
#             break
#         else:
#             max_list_with_log2[key]=value
#             i-=1
    
    stop_scability = False
    if SCABILITY:
        stop_scability = False
        if num_of_els <= len(S_poi_freq_dict):
            print('Number of elements to test in the scability analysis: '+str(num_of_els))
            S_poi_freq_dict = sort_attributes(S_poi_freq_dict)
            max_list_of_elems = {}
            i = 0
            for key, value in S_poi_freq_dict.items():
                if i < num_of_els:
                    max_list_of_elems[key]=value
                    i += 1
                else:
                    break
            S_poi_freq_dict = max_list_of_elems.copy()
            print('Number of the most frequent elements: ',len(S_poi_freq_dict))
        else:
            print('Current dataset is done for scability analisys.')
            stop_scability = True
    else:
        
        print('Original number of elements: ',len(S_poi_freq_dict))
        
        ## select att-values by its frequence that are higher than the average
        #if true it selects just the elements with frequency equal or bigger than the AVG; otherwise use all elements
        if element_analysis:
            max_list_of_elems = {}
            average_freq = np.mean(list(S_poi_freq_dict.values()))
            for key, value in S_poi_freq_dict.items():
                if value > average_freq:
                    max_list_of_elems[key]=value

            S_poi_freq_dict = max_list_of_elems.copy()
            print('Element analysis: True. Number of the most frequent elements: ',len(S_poi_freq_dict))
        else:
            print('Element analysis: False. We consider the original number of elements: ',len(S_poi_freq_dict))
    
    
    print('Tirando a duvida sobre o tamanho da lista de elementos: ',len(S_poi_freq_dict))
    
    start = timer()
    for iter_k in range(k):
        if stop_scability:
            break
#     for iter_k in tqdm(range(len(S_poi_freq_dict)*len(S_poi_freq_dict)), colour='blue', desc='Searching for candidates'):
        print('Searching for candidate: '+str(iter_k+1),end="\r")
#         print('')
        
#         print('S: ',S_poi_freq_dict)
        S_poi_freq_dict = sort_attributes(S_poi_freq_dict)
#         S_poi_freq_dict = sort_attributes(max_list_with_log2)
#         print('Current main list S: ',S_poi_freq_dict)
        S_uppercase_queue_list = populate_queue(S_poi_freq_dict)
        
        ### Initialize the current co-cluster 'cocluster_*' (CC) and candidate co-cluster 'cc_candidate' (CC*)
        cocluster_sequence_str = ''
        cocluster_attributes_list = ''
        cocluster_index_rows_set = set()
        cocluster_elements_set = set()
#         cocluster_cost_function = sys.maxsize
        cocluster_cost_function = INITIAL_COST
        cocluster_max_overlapped_coef = 1
        cc_candidate = {}
#         num_of_attributes = len(s_poi_freq_queue_list)

        clustering_perf.append_result(total_of_iterations,iter_k,final_coclustering_cost)
    
#         num_att_to_test_S = len(S_uppercase_queue_list)
#         while(num_att_to_test_S > 0):
#             num_att_to_test_S -= 1
#         while S_uppercase_queue_list: # loop it while queue is not empty
        limit = num_elements_to_test('length',len(S_uppercase_queue_list))
#         limit = num_elements_to_test('log2',len(S_uppercase_queue_list))
        for iter_elements_freq in range(0,limit):
#         for iter_elements_freq in tqdm(range(limit), colour='blue', desc='Testing element reference'):
            
#             if cocluster_sequence_str == '':
        
            S_poi_node_queue = S_uppercase_queue_list.popleft()
#             head_sequence_str = S_poi_node_queue[0]
#             trajectories_head_sequence_set = poi_at_trajs_dict_set[S_poi_node_queue[0]]
#             tail_sequence_str = S_poi_node_queue[0]
#             trajectories_tail_sequence_set = poi_at_trajs_dict_set[S_poi_node_queue[0]]
            S_uppercase_queue_list.append(S_poi_node_queue)
            s_lowercase_queue_list = S_uppercase_queue_list.copy()
#             sequence_cc = S_poi_node_queue[0]
            sequence_cc = {'cs_sequence_cc': S_poi_node_queue[0],
                           'cs_traj_ids_set_cc': poi_at_trajs_dict_set[S_poi_node_queue[0]],
                           'cs_elements_cc': set(),
                           'clustered_elements': final_clustered_elements}

            num_attributes_to_test_s = len(s_lowercase_queue_list)
            while(num_attributes_to_test_s > 0): # if it completes one loop the process stops

                s_poi_node_queue = s_lowercase_queue_list.popleft()
#                     poi_node_queue = s_poi_freq_queue_list[0]
                #s_lowercase_queue_list.append(s_poi_node_queue)# original: comentado para inserir apenas no update
                cc_candidate = candidate_cocluster(trajs_data_dict_list, poi_at_trajs_dict_set,
                                                   sequence_cc, s_poi_node_queue)

                if ((cc_candidate != None) and (cc_candidate['cost_function'] <= cocluster_cost_function) and 
                    (candidate_deviation(avg_cc_analysis,cc_candidate,final_coclustering_size,
                                         ('pass' if cc_type_process != 'incremental' else cc_type_process))) and 
                    (overlap_coefficient(cc_candidate['elements_set'],final_coclusters) <= overlap_coef_threshold)):
                    
                    over_coef_cc_candidate=overlap_coefficient(cc_candidate['elements_set'],final_coclusters)
#                     print('Current co-cluster CC was improved!')

                    ### update CC
                    cocluster_sequence_str = cc_candidate['sequence_str']
                    cocluster_attributes_list = cc_candidate['sequence_str'].split('-')
                    cocluster_index_rows_set = cc_candidate['index_rows_set'].copy()
                    cocluster_elements_set = cc_candidate['elements_set'].copy()
                    cocluster_cost_function = cc_candidate['cost_function']
                    cocluster_max_overlapped_coef = over_coef_cc_candidate

                    ### update sequence_cc
                    sequence_cc['cs_sequence_cc'] = cocluster_sequence_str
                    sequence_cc['cs_traj_ids_set_cc'] = cocluster_index_rows_set
                    sequence_cc['cs_elements_cc'] = cocluster_elements_set

#                     update_queue_s(cocluster_sequence_str, sequence_cc['cs_sequence_cc'],
#                                    s_lowercase_queue_list, s_poi_node_queue)
                    update_queue_s(cocluster_sequence_str, s_lowercase_queue_list, s_poi_node_queue)

                    num_attributes_to_test_s = len(s_lowercase_queue_list)# reassign the counter to restart


#                     trajectories_head_sequence_set = cocluster_index_rows_set
#                     head_sequence_str = cocluster_sequence_str
#                     trajectories_tail_sequence_set = cocluster_index_rows_set
#                     tail_sequence_str = cocluster_sequence_str


                    total_of_iterations += 1
        
    #                         clustering_perf.append_result(total_of_iterations,iter_k,cocluster_cost_function)
#                     else:# inserting back the element without update
#                         s_lowercase_queue_list.append(s_poi_node_queue)
#                         num_attributes_to_test_s -= 1
#                         total_of_iterations += 1
                else:# inserting back the element without update
                    s_lowercase_queue_list.append(s_poi_node_queue)
#                     print('Current co-cluster CC was NOT improved!')
#                     trajectories_head_sequence_set = tmp_traj_set
#                     head_sequence_str = tmp_head_sequence_str
#                     trajectories_tail_sequence_set = tmp_traj_set
#                     tail_sequence_str = tmp_tail_sequence_str
                    num_attributes_to_test_s -= 1
                    total_of_iterations += 1
#                         clustering_perf.append_result(total_of_iterations,iter_k,cocluster_cost_function)

#                 print('Queue s* AFTER to update: ',s_lowercase_queue_list)
#                 print('')

                ### Performance purpose ###
                ### Descontinuado ###
#                 if cocluster_cost_function != INITIAL_COST:
#                     clustering_perf.append_result(total_of_iterations,iter_k,
#                                                   (final_coclustering_cost+cocluster_cost_function))
#                 else:
#                     clustering_perf.append_result(total_of_iterations,iter_k,final_coclustering_cost)

            ## END while POIs_to_test (POIs_queue) ##
            #########################################

            ### check if CC was identified. If don't, it tries the next element p
            if cocluster_sequence_str == '':
                sequence_cc['cs_sequence_cc'] = ''
                sequence_cc['cs_sequence_cc'] = set()
                
            else: # co-cluster identified Step to store the found cocluster K
#                 final_coclusters.update({str(iter_k):{'cc_objs':cocluster_index_rows_set,
#                                                       'cc_atts':cocluster_sequence_str,
#                                                       'cc_elements':cocluster_elements_set,
#                                                       'cc_cost':cocluster_cost_function}})
#                 final_clustered_elements = final_clustered_elements.union(cocluster_elements_set)
#                 final_coclustering_cost += cocluster_cost_function
#                 print('Main list S BEFORE to update: ',S_poi_freq_dict)
#                 update_uppercase_S(cocluster_attributes_list, cocluster_index_rows_set, S_poi_freq_dict)
#                 print('Main list S AFTER to update: ',S_poi_freq_dict)
#                 partial = timer()
#                 print('Cluster "{}" finished at time "{}".'.format((iter_k+1),(partial-start))
                break
            
        ### END while S
        #
        
        ## into loop of iteration_k
        partial = timer()
        if VERBOSE:
            print('Cluster "{}" finished at time "{}".'.format(iter_k+1,partial-start))
        
        ### there is not any good co-cluster to identify anymore. Stop searching
        if (cocluster_cost_function >= 0) or (cocluster_max_overlapped_coef > overlap_coef_threshold):
            if VERBOSE:
                print('There is not any good co-cluster to identify anymore.')
                
            if cc_type_process == 'sample':
                candidate_ref = avg_cc_analysis
                set_of_candidates = final_coclusters.copy()
#                 candidates_ref_values = candidates_ref_values.copy()
                
                clustering_perf.store_dist(final_coclustering_size)
                candidate_deviation(avg_cc_analysis,final_coclusters,final_coclustering_size,'sample')
                end = timer()
                time_p2 = end-start
                total_time_alg = time_p1+time_p2
                clustering_perf.store_data_scability_test(candidates_ref_values, num_of_els,
                                                          format_time_minutes(total_time_alg), input_data, run_sim)
                
                print('\nTotal clustering time in minutes: ',format_time_minutes(total_time_alg))
                print('Total clustering time: ',format_time_output(end-start), end=" ")
                now = datetime.datetime.now()
                print("("+str(now.day)+"/"+str(now.month)+"/"+str(now.year)+" - "+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+")")
                print('')
                
                if run_sim < 1 and STORE_CLUS_STATS:
                    clustering_perf.compute_measures_at_once(set_of_candidates, candidates_ref_values,
                                                             map_id_to_attribute_dict,
                                                             trajs_data_dict_list, input_data,
                                                             clustering_perf.store_dist(candidates_ref_values))
            
            break
        else:
            if VERBOSE:
                    print('Co-cluster sequence "{}" present in "{}" trajectories.'.format(cocluster_sequence_str,
                                                                                          len(cocluster_index_rows_set)))
            final_coclusters.update({str(iter_k):{'cc_objs':cocluster_index_rows_set,
                                                      'cc_atts':cocluster_sequence_str,
                                                      'cc_elements':cocluster_elements_set,
                                                      'cc_cost':cocluster_cost_function,
                                                      'cc_over_coef':cocluster_max_overlapped_coef}})
            final_clustered_elements = final_clustered_elements.union(cocluster_elements_set)
            final_coclustering_cost += cocluster_cost_function
            update_uppercase_S(cocluster_attributes_list, cocluster_index_rows_set, S_poi_freq_dict)
            
            
            ### PERFORMANCE PURPOSE CODE ###
            ### storing the candidates reference values to evaluate the candidate later
            if avg_cc_analysis == "rows":
                final_coclustering_size.update({str(iter_k):len(cocluster_index_rows_set)})
                candidates_ref_values.update({str(iter_k):{'rows':len(cocluster_index_rows_set),
                                                           'cost':cocluster_cost_function}})
            elif avg_cc_analysis == "cost":
                final_coclustering_size.update({str(iter_k):cocluster_cost_function})
                candidates_ref_values.update({str(iter_k):{'rows':len(cocluster_index_rows_set),
                                                           'cost':cocluster_cost_function}})
            else:
                final_coclustering_size.update({str(iter_k):{'rows':len(cocluster_index_rows_set),
                                                             'cost':cocluster_cost_function}})
                candidates_ref_values.update({str(iter_k):{'rows':len(cocluster_index_rows_set),
                                                           'cost':cocluster_cost_function}})
            
            clustering_perf.store_data_scability_rows_cost(int(iter_k+1),
                                                           len(cocluster_index_rows_set),
                                                           int(cocluster_cost_function),
                                                           len(cocluster_sequence_str.split('-')),
                                                           format_time_minutes(timer()-start_1),
                                                           int(len(cocluster_elements_set)),
                                                           int(num_of_els),
                                                           int(run_sim),
                                                           input_data)
        
    clustering_perf.summary_clusters(final_coclusters, map_id_to_attribute_dict, trajs_data_dict_list)
    del map_id_to_attribute_dict,trajs_data_dict_list,candidates_ref_values,cocluster_index_rows_set,cocluster_elements_set
    del final_clustered_elements, S_poi_freq_dict, poi_at_trajs_dict_set
    return D,final_coclusters

def candidate_deviation(ref,value_ref,set_of_clusters,cc_type_process='incremental'):
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
                ### z-score: we consider values greater than z_thres once it is a positive distribution
                    try:
                        z = (len(value_ref['index_rows_set'])-mean)/std
                    except:
                        z = (value_ref-mean)/std
                    print('Z-score(rows): ',z)
                    return z >= cc_z_threshold_r
            elif ref == "cost":
                ### normal mean
                if cc_type_analysis == 'mean':
                    return value_ref['cost_function'] <= np.ceil(mean)
                else:
                ### z-score: we consider values smaller than z_thres once it is a negative distribution
                    try:
                        z = (value_ref['cost_function']-mean)/std
                    except:
                        z = (value_ref-mean)/std
                    print('Z-score(cost): ',z)
                    return z <= cc_z_threshold_c
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
                    return ((z_rows >= cc_z_threshold_r) or (z_cost <= cc_z_threshold_c))
                
        elif cc_type_process == 'sample':
            candidates_to_remove = []
            try:# single ref: rows OR cost
                mean
                if cc_type_analysis == 'mean':
                    print('Mean:',mean)
                    for key,value in set_of_clusters.items():
#                         print('Candidate-'+key+' Mean:',mean,' Value ref:',value,end='')
                        if ref == 'rows' and value < mean:
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        elif ref == 'cost' and value > mean:
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        else:
#                             print(' -> Keep')
                            pass
                else:#z-score
                    for key,value in set_of_clusters.items():
                        z = (value-mean)/std
#                         print('Candidate-'+key+' Z-score:',z,end='')
                        if ref == 'rows' and z < cc_z_threshold_r:
                            candidates_to_remove.append(key)
#                             print(' -> Remove')
                        elif ref == 'cost' and z > -cc_z_threshold_c:
                            candidates_to_remove.append(key)
#                             print(' -> Remove')
                        else:
#                             print(' -> Keep')
                            pass

            except:#double ref combine: rows AND cost
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
                        if (z_rows < cc_z_threshold_r) and (z_cost > cc_z_threshold_c):
#                             print(' -> Remove')
                            candidates_to_remove.append(key)
                        else:
#                             print(' -> Keep')
                            pass
                        
#             print("Remove candidates: ",candidates_to_remove)
#             print("Number of candidates to remove: ",len(candidates_to_remove))
            for candidate in candidates_to_remove:#at this point, value_ref is the set of candidates
                del value_ref[candidate]

        else: # just pass step. Stores the candidate co-clusters and analyse them with sample analysis if desirable
            return True               
                
    else:# pass step to reach a minimum number of elements to perform computation
        return True
    
def candidate_cocluster(trajs_data_dict_list, poi_at_trajs_dict_set, sequence_cc, s_poi_node_queue):
    INITIAL_COST = 100.0
    ### Current sequence
    ### The method tries to form two sequence, if the sequence is valid the method picks the best one
    head_sequence_str = sequence_cc['cs_sequence_cc']
    trajectories_head_sequence_set = sequence_cc['cs_traj_ids_set_cc']

    tail_sequence_str = sequence_cc['cs_sequence_cc']
    trajectories_tail_sequence_set = sequence_cc['cs_traj_ids_set_cc']

    ### Try to expand the candidate sequence one element at a time if it forms a frequent sequence
    ### Step to test ELEMENT at the HEAD ###
    tmp_head_sequence_str = head_sequence_str
    head_sequence_str = s_poi_node_queue[0]+'-'+head_sequence_str
#     if VERBOSE:
#         print('-> Head sequence: ',head_sequence_str)
    tmp_traj_set = trajectories_head_sequence_set
    trajectories_head_sequence_set = trajectories_head_sequence_set.intersection(poi_at_trajs_dict_set[s_poi_node_queue[0]])
    trajectories_head_sequence_set, position_poi_per_traj_head = check_sequence(trajs_data_dict_list,
                                                                                trajectories_head_sequence_set,
                                                                                head_sequence_str)

    if len(trajectories_head_sequence_set) > 0:
#         if VERBOSE:
#             print('Number of rows with this sequence: {}'.format(len(trajectories_head_sequence_set)))
        elements_head_sequence = form_elements(trajectories_head_sequence_set,
                                               head_sequence_str,
                                               position_poi_per_traj_head)    
        overlapped_elements = elements_head_sequence.intersection(sequence_cc['clustered_elements'])
        cost_head_sequence = cost_function(len(trajectories_head_sequence_set),
                                           len(head_sequence_str.split('-')),
                                           len(overlapped_elements))
#         overlap_coef_head = overlap_coefficient(elements_head_sequence,final_coclusters)
#         print('Head cost: {} and overlap_coef: {}.'.format(cost_head_sequence,
#                                                            overlap_coef_head))
    else:
#         if VERBOSE:
#             print('Tested head sequence "{}" does NOT exist!'.format(head_sequence_str))
        trajectories_head_sequence_set = tmp_traj_set
        head_sequence_str = tmp_head_sequence_str
        cost_head_sequence = INITIAL_COST #
        overlap_coef_head = 1
    #### END test HEAD sequence ####

    #### Step test ELEMENT at the TAIL ####
    tmp_tail_sequence_str = tail_sequence_str
    tail_sequence_str = tail_sequence_str+'-'+s_poi_node_queue[0]
#     if VERBOSE:
#         print('-> Tail sequence: ',tail_sequence_str)
    tmp_traj_set = trajectories_tail_sequence_set
    trajectories_tail_sequence_set = trajectories_tail_sequence_set.intersection(poi_at_trajs_dict_set[s_poi_node_queue[0]])
    trajectories_tail_sequence_set, position_poi_per_traj_tail = check_sequence(trajs_data_dict_list,
                                                                                trajectories_tail_sequence_set,
                                                                                tail_sequence_str)

    if (len(trajectories_tail_sequence_set) > 0):
#         if VERBOSE:
#             print('Number of rows with this sequence: {}'.format(len(trajectories_tail_sequence_set)))
        elements_tail_sequence = form_elements(trajectories_tail_sequence_set,
                                               tail_sequence_str,
                                               position_poi_per_traj_tail)
        overlapped_elements = elements_tail_sequence.intersection(sequence_cc['clustered_elements'])
        cost_tail_sequence = cost_function(len(trajectories_tail_sequence_set),
                                           len(tail_sequence_str.split('-')),
                                           len(overlapped_elements))
#         overlap_coef_tail = overlap_coefficient(elements_tail_sequence,final_coclusters)
#         print('Tail cost: {} and overlap_coef: {}.'.format(cost_tail_sequence,
#                                                            overlap_coef_tail))
    else:
#         if VERBOSE:
#             print('Tested tail sequence "{}" does NOT exist!'.format(tail_sequence_str))
        trajectories_tail_sequence_set = tmp_traj_set
        tail_sequence_str = tmp_tail_sequence_str
        cost_tail_sequence = INITIAL_COST
        overlap_coef_tail = 1
    #### END test TAIL sequence ####
    
#     print('Current co-cluster cost: ',cocluster_cost_function)
#     print('Queue s* BEFORE to upadate: ',s_lowercase_queue_list)

    ### Step to test the best sequence if exist a sequence
    if (cost_head_sequence < cost_tail_sequence) and (cost_head_sequence < 0):
#         print('Co-cluster improved with HEAD sequence.')

        # update the nodes of queue s.
#         update_queue_s(cocluster_sequence_str, head_sequence_str,
#                        s_lowercase_queue_list, s_poi_node_queue)
#         update_queue_s(candidate_sequence['cs_sequence_cc'], head_sequence_str,
#                        s_lowercase_queue_list, s_poi_node_queue)

#         cocluster_sequence_str = head_sequence_str
#         cocluster_attributes_list = head_sequence_str.split('-')
#         cocluster_index_rows_set = trajectories_head_sequence_set.copy()
#         cocluster_elements_set = elements_head_sequence.copy()
#         cocluster_cost_function = cost_head_sequence
#         cocluster_max_overlapped_coef = overlap_coef_head
        
        cc_candidate = {'sequence_str': head_sequence_str,
                        'attributes_list': head_sequence_str.split('-'),
                        'index_rows_set': trajectories_head_sequence_set.copy(),
                        'elements_set': elements_head_sequence.copy(),
                        'cost_function': cost_head_sequence}        
        
        return cc_candidate

    elif (cost_tail_sequence < cost_head_sequence) and (cost_tail_sequence < 0):
#         if VERBOSE:
#             print('Co-cluster improved with TAIL sequence.')

        # update the nodes of queue s.
#         update_queue_s(cocluster_sequence_str,tail_sequence_str,
#                        s_lowercase_queue_list,s_poi_node_queue)
#         update_queue_s(candidate_sequence['cs_sequence_cc'], tail_sequence_str,
#                        s_lowercase_queue_list, s_poi_node_queue)

#         cocluster_sequence_str = tail_sequence_str
#         cocluster_attributes_list = tail_sequence_str.split('-')
#         cocluster_index_rows_set = trajectories_tail_sequence_set.copy()
#         cocluster_elements_set = elements_tail_sequence.copy()
#         cocluster_cost_function = cost_tail_sequence
#         cocluster_max_overlapped_coef = overlap_coef_tail
        
        cc_candidate = {'sequence_str': tail_sequence_str,
                        'attributes_list': tail_sequence_str.split('-'),
                        'index_rows_set': trajectories_tail_sequence_set.copy(),
                        'elements_set': elements_tail_sequence.copy(),
                        'cost_function': cost_tail_sequence}        
        
        return cc_candidate
    
    else:# it does not found any sequence formed by the elements
#         cc_candidate = {'sequence_str': None}
        return None

def create_alluvial_diagram():
    df_traj_user = pd.read_csv('./data/real_application/foursquare_NY/fs_ny_top_users_10.csv', sep=';')
#     df_traj_user.drop(columns=['tid','lat_lon','time','day','type','root_type','rating','weather'],inplace=True)
    df_traj_user = df_traj_user[['new_tid','label']]
    #     print(df_traj_user.head())
    
    traj_id = 50
    user_label = df_traj_user[df_traj_user['new_tid'] == traj_id]['label'].unique()[0]
    print('Trajectory "{}" belongs to User: "{}"'.format(traj_id,
                                                         user_label))
    
# def update_queue_s(cocluster_sequence_str, tested_sequence_str, s_poi_freq_queue_list, poi_node_queue):
def update_queue_s(cocluster_sequence_str, s_poi_freq_queue_list, poi_node_queue):
    '''
    Method to update the nodes in queue s. It decrements the value of a given node in s.
    The input are:
        1. The current string sequence of a cocluster;
        2. The tested string sequence to improve a cocluster;
        3. The queue s;
        4. A single node of queue s.
    '''
    # update list s when the first sequence is identified
#     if cocluster_sequence_str == '':
    tmp_split = cocluster_sequence_str.split('-')
    if len(tmp_split) == 2:
#         tmp_split = tested_sequence_str.split('-')
        s_poi_freq_queue_list.append(poi_node_queue)
        for attribute in tmp_split:
            for node_s in s_poi_freq_queue_list:
                if attribute == node_s[0]:
                    node_s[1] -= 1
                    if node_s[1] <= 0: # all occurences were used, then remove the element from the queue
                        print('Element with 0 removed.')
                        s_poi_freq_queue_list.remove(node_s)
                    break
    else: # update a single node in case a sequence is already discovered
        poi_node_queue[1] -= 1
        if poi_node_queue[1] <= 0: # all occurences were used, then remove the element from the queue
            s_poi_freq_queue_list.remove(node_s)
        else:
            s_poi_freq_queue_list.append(poi_node_queue)

def update_uppercase_S(cc_atts, cc_objs, S_dict):
    '''
    Method to update the dictonary S. It decrements the frequency of the given attributes in S.
    S is updated regarding the frequency of an attribute times the number of objects that it appears in a
    co-cluster.
    E.g., Given a co-clsuter with sequence Home-Work-Home with 5 trajectories. Then, in S, Home is 
    decremented with value 10 (2*5) and Work with value 5 (1*5).
    
    The input are:
        1. Co-cluster attributes;
        2. Co-cluster objects;
        3. The dictionary S of attributes and its frequency.
    '''
    tmp_dict = {}
    for attribute in cc_atts: # groups repeation
#         S_poi_freq_dict[attribute] -= 1
        try:
            tmp_dict[attribute] += 1
        except:
            tmp_dict.update({attribute:1})
    for attribute, value in tmp_dict.items():
        S_dict[attribute] -= (tmp_dict[attribute]*len(cc_objs))
        if S_dict[attribute] <= 0:
            S_dict.pop(attribute)
            
def num_elements_to_test(option,number):
    
    if option == 'log2':
        return int(round(np.log2(number)))
    elif option == 'log10':
        return int(round(np.log10(number)))
    elif option == 'length':# Attention! The number for length must to be at most length of structure. e.g. array, dic.
        return int(round(number))
    else:
        return print('Choose a valid option!')
    
