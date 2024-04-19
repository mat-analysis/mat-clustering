clusters = [7,10,4] # number of co-clusters in the ground-truth data, respectively.
path_method = "OutputAnalysis\panda"
# check_path(path_method)
# for ds in range(len(datasets)):
for ds in range(1):
    ds_name = "Syn-"+str(ds+1)
    print("\nDataset: "+ds_name)
#     res = os.mkdir(path_method+"\\"+ds_name)
    
#     numberOfRuns =1
    for t in range(1,4):
        print("T"+str(t)+" evaluation (",end="")
        if t == 1:
            T = "nl0_nc0"
        elif t == 2:
            T = "nl0.5_nc0.8"
        else:
            T = "nl1_nc1"
        print(T+").")
        for run in range(numberOfRuns):
            print("Run-"+str(run+1))

            df = pd.read_csv(datasets[ds],header=None)
            df.columns = [str(i) for i in range(df.shape[1])]
            ncols = df.shape[1]
            data = df.values.copy()
            del df

            pathPandaT = "./OutputAnalysis/panda/originalOutput/"
#             T1 = "/nl0_nc0/"
            file = "synthetic-1-fimi-run_"+str(run+1)+"-t"+str(t)+"_k7_"+T+".txt"
            df_fimi = pd.read_csv(pathPandaT+ds_name+"/"+T+"/"+file, header=None, names=["transation"])
            columnClusters,rowClusters = processDocument(df_fimi)

            reconstructed_panda = np.zeros(data.shape,dtype=int)
            for number_of_clusters in range(len(columnClusters)):
                for i in rowClusters[number_of_clusters]:
                    for j in columnClusters[number_of_clusters]:
                        reconstructed_panda[int(i)][int(j)] = 1
    #         print("Data cost: ",data.sum())
            print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_panda)))
            del reconstructed_panda, data
            gc.collect()

            #format to omega and f-score measure
            clustering = preprocess_clustering_output(rowClusters,columnClusters)
            # print(clustering)

            # save to XMEASURES format C++ version
            PaNDa_clustering_xm = xmeasures_format(clustering)
            df_gt = pd.DataFrame(PaNDa_clustering_xm)
            pathCNL = "./OutputAnalysis/panda/cnlFormat/"+ds_name+"/"+T+"/"
            newFile = file.split("-")
            cnlFile = "run_"+str(run+1)+"_res_panda_t"+str(t)+"_"+"k"+str(clusters[ds])+"_"+T+"_"+newFile[0]+"-"+str(ds+1)+"_co.cnl"
#             print(cnlFile)
            # df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_ococlus_"+ds_name+"_co.cnl", 
            #              header= False,index=False, encoding='utf8')
            df_gt.to_csv(pathCNL+cnlFile, header=False, index=False, encoding='utf8')
            del clustering, df_gt, PaNDa_clustering_xm
            gc.collect()
        print("")