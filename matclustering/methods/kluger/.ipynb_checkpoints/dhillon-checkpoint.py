# Co-clustering (Block diagonal) - Dhillon (2001)
from sklearn.cluster.bicluster import SpectralCoclustering
clusters = [7,10,4] # number of co-clusters in the ground-truth data, respectively.
path_method = "OutputAnalysis\dhillon"
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

        DhillonCocluster = SpectralCoclustering(n_clusters = clusters[ds])
        DhillonCocluster.fit(data)

        # Reconstruct matrix
        reconstructed_matrix = np.zeros(data.shape,dtype=int)
        for nc in range(clusters[ds]):
            if len(DhillonCocluster.get_indices(nc)[0]) != 0 and len(DhillonCocluster.get_indices(nc)[1]) != 0:
                for i in DhillonCocluster.get_indices(nc)[0]:
                    for j in DhillonCocluster.get_indices(nc)[1]:
                        reconstructed_matrix[i][j] = 1
#         print("Data cost: ",data.sum())
        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_matrix)))
        
        clustering = build_clustering_output_omega(clusters[ds],DhillonCocluster) #format to omega and f-score measure

        # XMEASURES format ground-truth C++ version
        DhillonCocluster_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(DhillonCocluster_clustering_xm)
        path = path_method+"/"+ds_name
        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_dhillon_"+ds_name+"_co.cnl", 
                     header= False,index=False, encoding='utf8')
        del reconstructed_matrix, clustering, df_gt, DhillonCocluster_clustering_xm
        gc.collect()