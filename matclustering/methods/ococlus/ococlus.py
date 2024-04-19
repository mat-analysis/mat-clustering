def OCoClus(input_D,k=-1,e_obj=-1,e_att=-1):
    ### variable declaration
    if k == -1:
        k=sys.maxsize
    if e_obj == -1:
        e_obj = 1
    if e_att == -1:
        e_att = 1
    
    cost_model = sys.float_info.max # initial cost function of the model
    num_of_coclusters = 0
    final_coclusters = [] # store the attribute and objects clusters. final_coclusters[[C1_att,C1_obj],[Ck_att,Ck_obj]]
    pattern_model = [set(),set()]# Union between the found co-clusters [list of obj,list of att]
    cost_per_cocluster = []# stores the cost to build the cocluster
    history_cost_model = []
    ###
    
    D,N,data_dict,data_res_dict = get_data(input_D)
    for itertator in range(1,k+1):
        print("Iterator: ",itertator)
        if VERBOSE:
            print("Pattern model: ",pattern_model)
        C,E,new_cost_dense = find_dense_cocluster(data_dict, data_res_dict)
        if new_cost_dense >= 0:
            print("No relevant co-cluster can be found anymore.")
            break    
        C_expanded = expand_dense_cocluster(C, E, data_dict,e_obj,e_att)

        print("")
        new_cost = cost_function(len(pattern_model[0].union(C[1])),
                                              len(pattern_model[1].union(set(C[0]))))
        if VERBOSE:
            print("Dense co-cluster cost: ",new_cost_dense)
            print("Current model cost: ",cost_model)
            print("New model cost with the found co-cluster: ",new_cost)
            print("Attribute cluster:"+str(C[0])+", Object cluster: "+str(C[1]))
            print(" ")
        
        if new_cost < cost_model:
            final_coclusters.append(C)
            cost_model = new_cost
            num_of_coclusters += 1
            data_res_dict = update_residual_dataset(data_res_dict,C[0],C[1])
            pattern_model[0] = pattern_model[0].union(C[1])
            pattern_model[1] = pattern_model[1].union(set(C[0]))
        else:
            print("No co-cluster can be found anymore.")
            break
    
    print("Find global co-cluster is done.")
    if find_overlap:
        attributes_per_cluster = []
        objects_per_cluster = []
        for i in range(len(final_coclusters)):
            attributes_per_cluster.append(set(list(map(int,final_coclusters[i][0]))))
            objects_per_cluster.append(final_coclusters[i][1])

        print("\nFind overlapped co-clusters method")
        attribute_clusters, objects_clusters = findOverlap(attributes_per_cluster,objects_per_cluster)

        final_coclusters = []
        for i in range(len(attribute_clusters)):
            final_coclusters.append([attribute_clusters[i],objects_clusters[i]])

        print("\nNon-overlapped and overlapped co-clusters.")
        print("Number of co-clusters: ",len(final_coclusters))
        if VERBOSE:
            print("Final co-clusters: ",final_coclusters)
    else:
        print("\nNon-overlapped co-clusters.")
        print("Number of co-clusters: ",num_of_coclusters)
#         for i in range(len(final_coclusters)):
#             tmp_data = final_coclusters[i][1]
#             final_coclusters[i][1] = list(tmp_data)
        if VERBOSE:
            print("Final co-clusters: ",final_coclusters)

    return D,final_coclusters

def find_dense_cocluster(input_dataset, residual_dataset):
    '''
    Input
        input_dataset: a dictionary with the list of objects per attribute. input_dataset[att] -> [objects_in_att]
        residual_dataset: it is a copy of input_dataset used to sort the attributes
    
    Output
        C: A dense co-cluster. It is a array with a list of attributes and a set of objects. C[att_list,obj_set]
    '''
    print("Find dense cocluster method.")
    att_dense_cocluster_list = []
#     att_dense_cocluster_set = set([])
    obj_dense_cocluster_list = []
    att_extension_list = []
    cost_dense_cocluster = sys.float_info.max
    
    sads = sort_att_ds(residual_dataset)
#     print("Sorted att: ",sads)
    cc_att = sads[0]
#     att_dense_cocluster_list.append(cc_att)
    att_dense_cocluster_list.append(cc_att)
    obj_dense_cocluter_set = set(residual_dataset[cc_att]) # get the objs forthe given attribute cc_att
    count_group_att = 1
#     cost_dense_cocluster = cost_function(len(pattern_model[0].union(obj_dense_cocluter_set)),
#                                          len(pattern_model[1].union(set(cc_att))))
    new_cost_function = cost_function(len(obj_dense_cocluter_set), count_group_att)
    if VERBOSE:
        print("New cost: "+str(new_cost_function)+", Cost dense: "+str(cost_dense_cocluster))
    cost_dense_cocluster = new_cost_function
    
    for next_att in range(1,len(sads)):
        cc_att_test = sads[next_att]
#         print("Attribute: ",cc_att_test)
        curr_cc_att = set(residual_dataset[cc_att_test])
        intersection_objs = obj_dense_cocluter_set.intersection(curr_cc_att)
#         print("Intersection test: ",intersection_objs)
        tmp = att_dense_cocluster_list.copy()
        tmp.append(cc_att_test)
#         new_cost_function = cost_function(len(pattern_model[0].union(intersection_objs)),
#                                           len(pattern_model[1].union(set(tmp))))
        new_cost_function = cost_function(len(intersection_objs),count_group_att+1)
#         print(intersection_objs,curr_cc_att)
#         print(len(intersection_objs),count_group_att+1)
        if VERBOSE:
            print("New cost: "+str(new_cost_function)+", Cost dense: "+str(cost_dense_cocluster))
            
        if  new_cost_function <= cost_dense_cocluster:
            att_dense_cocluster_list.append(cc_att_test)
            obj_dense_cocluter_set = intersection_objs
            cost_dense_cocluster = new_cost_function
            count_group_att += 1
        else:
            att_extension_list.append(cc_att_test)
    
    C = [att_dense_cocluster_list, obj_dense_cocluter_set]
    
    # no good rectangle was found
    if cost_dense_cocluster >= 0:
        att_extension_list = []
        C = [[],set()]
    else:
        C = [att_dense_cocluster_list, obj_dense_cocluter_set]
    
    return C, att_extension_list, cost_dense_cocluster

def expand_dense_cocluster(C,E,att_data_dict,e_obj,e_att):
    '''
    INPUT
        C = a tuple(list_atts, set_objs); list_atts has the list of attributes and set_objs is a set of objects
        E = list of attributes not present in C_a to be tested
        att_data_dict = a dict with list of objects per att; att_data_dict[att] -> [objects]
        e_obj = maximum object error 
        e_att = maximum attribute error 
    
    OUTPUT
        
    '''
    print("Expand co-cluster method.")
    added_att = True
    
    if not C[0]: # nothing good to discover
        pass
    else:
        curr_cost = cost_function(len(C[1]), len(C[0]),0,0)
        noise_added = 0 # the quantanty of noise added in the cluster during the process
        while(added_att):
            # try to add new objects to cocluster C
            try_new_objs = set(np.arange(0,len(D))).difference(C[1]) # get the objects not present in C
            if VERBOSE:
                print("# Try to extend the list of Objects #")
            for obj in try_new_objs:
                obj_quantanty = 0
                for att in C[0]:
                    if D[obj][int(att)] == 1:
                        obj_quantanty += 1
                if not_too_noisy(obj_quantanty, C, e_obj, e_att, att_data_dict, E, "obj"):
#                     print("Ruído valor: ",not_noise_val)
                    new_cost = cost_function(len(C[1])+1,len(C[0]),0,(noise_added+(len(C[0])-obj_quantanty)))
                    if new_cost <= curr_cost:
                        C[1].add(obj)
                        curr_cost = new_cost
                        noise_added += (len(C[0])-obj_quantanty)
                else:
                    if VERBOSE:
                        print("Object too noisy to be added. (Obj: "+str(obj)+")")

            added_att = False
            # try to add new attributes to cocluster C
            if VERBOSE:
                print("# Try to extend the list of Attributes #")
            
            while(len(E) != 0):
                att = E.pop(0)
                if VERBOSE:
                    print("ATT->"+str(att), end= " | ")
                att_obj_quantanty = len(C[1].intersection(set(att_data_dict[att])))
                if not_too_noisy(att_obj_quantanty, C, e_obj, e_att, att_data_dict, E, "att"):
                    new_cost = cost_function(len(C[1]),len(C[0])+1,0,(noise_added+(len(C[1])-att_obj_quantanty)))
                    if new_cost <= curr_cost:
                        C[0].append(str(att))
                        curr_cost = new_cost
                        added_att = True
                        noise_added += (len(C[1])-att_obj_quantanty)
                        break
                else:
                    if VERBOSE:
                        print("Attribute too noisy to be added. (Att: "+str(att)+")")
                        
def findOverlap(SetsC,SetsR):
    '''
    INPUT
        SetsC: It has K sets of attributes regarding each attribute cluster
        SetsR: It has K sets of objects regarding each object cluster
    
    OUTPUT
        columnClusters: A list with the attribute clusters
        rowClusters: A list with the object clusters
    '''
    newSetsColumns = []
    newSetsRows = []
    
    # merge sets that can overlap
    merge(SetsC,SetsR,newSetsColumns,newSetsRows)

    #Removing sets with redundant information
    removeSubsets(newSetsColumns,newSetsRows)
    
    columnClusters = newSetsColumns
    rowClusters = newSetsRows
    return columnClusters,rowClusters

def merge(SetsC, SetsR, newSetsColumns, newSetsRows):
    num_of_cluster = len(SetsC)
    
    # keep the original cluster
    for set_column_i in range(len(SetsC)):
        newSetsRows.append(SetsR[set_column_i])
        newSetsColumns.append(SetsC[set_column_i])
    
    revisit = True
    while(revisit):
        revisit = False
        tmp_r = []
        tmp_c = []
        size = len(newSetsColumns)
        for i in range(size-1):
            first_r = newSetsRows[i]
            first_c = newSetsColumns[i]
            for j in range(i+1,size):
                p_intersec_r = first_r.intersection(newSetsRows[j])
                p_intersec_c = first_c.union(newSetsColumns[j])
                if len(p_intersec_r) > 0:
                    tmp_r.append(p_intersec_r)
                    tmp_c.append(p_intersec_c)
        
        for i in range(len(tmp_c)):
            change = True
            for j in range(len(newSetsColumns)):
                if tmp_c[i].issubset(newSetsColumns[j]):
                    change = False
            if change:
                newSetsRows.append(tmp_r[i])
                newSetsColumns.append(tmp_c[i])
                revisit = True
    
    # start to find and merge all possible overlapping clusters
    for set_column_i in range(num_of_cluster):
        tested_pattern = SetsR[set_column_i]
        new_cols = SetsC[set_column_i]
        overlapped = False
        
        # finds with who the tested pattern overlaps
        set_overlap_id = []
        for i in range(num_of_cluster):
            if i != set_column_i:
                if len(tested_pattern.intersection(SetsR[i])) > 0:
                    set_overlap_id.append(i)    
        
        # Discover if exist overlaps between the sets in set_overlap_id
        sub_overlap_id = []
        for i in range(len(set_overlap_id)):
            try_combine_ids = [set_overlap_id[i]]
            added = False
            for j in range(i+1,len(set_overlap_id)):
                if len(SetsR[set_overlap_id[i]].intersection(SetsR[set_overlap_id[j]])) > 0:
                    try_combine_ids.append(set_overlap_id[j])
                    added = True
            sub_overlap_id.append(try_combine_ids)
        #Check if some pattern is isolated and was not added
        for id_pattern in set_overlap_id:
            added = False
            for combined_ids in sub_overlap_id:
                if id_pattern in combined_ids:
                    added = True
            if added == False:
                sub_overlap_id.append([id_pattern])
        
        # merge the analysed pattern with the overlapped combined patterns ids
        for pattern_ids in sub_overlap_id:
            tested_pattern = SetsR[set_column_i]
            new_cols = SetsC[set_column_i]
            for pattern_id in pattern_ids:
                tmp = tested_pattern.intersection(SetsR[pattern_id])
                tested_pattern = tmp
                new_cols = new_cols.union(SetsC[pattern_id])
            # save the new patterns
            newSetsColumns.append(new_cols)
            newSetsRows.append(tested_pattern)

def removeSubsets(setsColumns,setsRows):
    finalSetsCols = []
    finalSetsRows = []
    
    again = True
    # We are out of the loop when we do not have any subset to remove
    while(again):
        again = False
        for i in range(len(setsColumns)):
            isSubset = False
            currC = setsColumns[i]
            currR = setsRows[i]
            
            for j in range(len(setsColumns)):
                if i != j:
                    nextC = setsColumns[j]
                    nextR = setsRows[j]
                    if currC.issubset(nextC) and currR.issubset(nextR):
                        isSubset = True
        
            if isSubset or len(setsRows[i]) == 0:
                setsColumns.pop(i)
                setsRows.pop(i)
                again = True
                break
    
    #converting data type back to list
    for i in range(len(setsColumns)):
        setsColumns[i] = list(setsColumns[i])
        setsRows[i] = list(setsRows[i])