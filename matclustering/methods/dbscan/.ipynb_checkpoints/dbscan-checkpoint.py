from sklearn.cluster import DBSCAN
path_method = "OutputAnalysis\dbscan"
check_path(path_method)
for ds in range(len(datasets)):
    ds_name = "Syn-"+str(ds+1)
    print("\nDataset: "+ds_name)
    res = os.mkdir(path_method+"\\"+ds_name)

    for run in range(numberOfRuns):
        print("Run-"+str(run+1))
    
        df = pd.read_csv(datasets[ds],header=None)
        df.columns = [str(i) for i in range(df.shape[1])]
        ncols = df.shape[1]
        data = df.values.copy()
        del df

        dbscan = DBSCAN(eps = 0.5, min_samples = 5) # default parameters
        dbscan.fit(data)

        ids_clus = [ele for ele in list(set(dbscan.labels_)) if ele != -1 ]

        reconstructed_matrix = np.zeros(data.shape,dtype=int)
        for i in range(len(dbscan.labels_)):
            if dbscan.labels_[i] != -1: #mark the elements if it's not noise
                reconstructed_matrix[i,:] = 1
#         print("Data cost: ",data.sum())
        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_matrix)))
        del data, reconstructed_matrix
        gc.collect()
        
        clustering = build_clustering_output_omega(ids_clus,(dbscan.labels_,ncols),trad=True)

        # XMEASURES format ground-truth C++ version
        dbscan_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(dbscan_clustering_xm)
        path = path_method+"/"+ds_name
        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_dbscan_"+ds_name+"_trad.cnl", 
                     header= False,index=False, encoding='utf8')
        del clustering, df_gt, dbscan_clustering_xm
        gc.collect()