from sklearn.cluster.bicluster import SpectralBiclustering
clusters = [(6,5),(7,8),(3,3)] # number of co-clusters in the ground-truth data, respectively.
path_method = "OutputAnalysis\kluger"
check_path(path_method)
for ds in range(len(datasets)):
    ds_name = "Syn-"+str(ds+1)
    print("\nDataset: "+ds_name)
    res = os.mkdir(path_method+"\\"+ds_name)

    for run in range(numberOfRuns):
        print("Run-"+str(run+1))
        
        df = pd.read_csv(datasets[ds],header=None)
        df.columns = [i+1 for i in range(df.shape[1])]
        data = df.values.copy()
        del df

        KlugerCocluster = SpectralBiclustering(n_clusters = clusters[ds])
        KlugerCocluster.fit(data)

        reconstructed_kluger = np.zeros(data.shape,dtype=int)
        for nc in range((clusters[ds][0]*clusters[ds][1])):
            if len(KlugerCocluster.get_indices(nc)[0]) != 0 and len(KlugerCocluster.get_indices(nc)[1]) != 0:
                for i in KlugerCocluster.get_indices(nc)[0]:
                    for j in KlugerCocluster.get_indices(nc)[1]:
                        reconstructed_kluger[i][j] = 1
        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_kluger)))
        del reconstructed_kluger, data
        gc.collect()
        
        clustering = build_clustering_output_omega(clusters[ds],KlugerCocluster) #format to omega and f-score measure

        # XMEASURES format ground-truth C++ version
        KlugerCocluster_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(KlugerCocluster_clustering_xm)
        path = path_method+"/"+ds_name
        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_kluger_"+ds_name+"_co.cnl", 
                     header= False,index=False, encoding='utf8')
        del clustering, df_gt, KlugerCocluster_clustering_xm
        gc.collect()