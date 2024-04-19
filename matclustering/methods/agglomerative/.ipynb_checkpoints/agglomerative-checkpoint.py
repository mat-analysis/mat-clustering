from sklearn.cluster import AgglomerativeClustering
path_method = "OutputAnalysis\Agglomerative"
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

        HClustering = AgglomerativeClustering(n_clusters=clusters[ds], linkage='ward') #default parameters
        HClustering.fit(data)

        reconstructed_matrix = np.ones(data.shape,dtype=int)
        ids_clus = list(set(HClustering.labels_))
#         print("Data cost: ",data.sum())
        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_matrix)))
        del reconstructed_matrix, data
        gc.collect()
        
        clustering = build_clustering_output_omega(ids_clus,(HClustering.labels_,ncols),trad=True)

        # XMEASURES format ground-truth C++ version
        HClustering_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(HClustering_clustering_xm)
        path = path_method+"/"+ds_name
        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_agglomerative_"+ds_name+"_trad.cnl", 
                     header= False,index=False, encoding='utf8')
        del clustering, df_gt, HClustering_clustering_xm
        gc.collect()