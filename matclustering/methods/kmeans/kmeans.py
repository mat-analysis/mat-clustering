from sklearn.cluster import KMeans
path_method = "OutputAnalysis\kmeans"
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

#         kmeans = KMeans(n_clusters = clusters[ds], random_state = 0)
        kmeans = KMeans(n_clusters = clusters[ds])
        kmeans.fit(data)
        ids_clus = list(set(kmeans.labels_))
        reconstructed_matrix = np.ones(data.shape,dtype=int)
#         print("Data cost: ",data.sum())
        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_matrix)))
        del data, reconstructed_matrix
        gc.collect()
        
        clustering = build_clustering_output_omega(ids_clus,(kmeans.labels_,ncols),trad=True)

        # XMEASURES format ground-truth C++ version
        kmeans_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(kmeans_clustering_xm)
#         name = datasets[ds].split("/")[-1]
#         name = name.split(".")[0]
        path = path_method+"/"+ds_name
        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_kmeans_"+ds_name+"_trad.cnl", 
                     header= False,index=False, encoding='utf8')
        del clustering, df_gt, kmeans_clustering_xm
        gc.collect()