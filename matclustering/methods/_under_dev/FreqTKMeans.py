import numpy as np
from sklearn.cluster import KMeans

from matclustering.methods.core import AbstractTrajectoryClustering
from matclustering.methods.evaluation import build_clustering_output_omega, xmeasures_format

class TKMeans(AbstractTrajectoryClustering): # Trajectory KMeans
    def __init__(self,
                 k=6,
                 random_state=1,
                 verbose=False):
        
        super().__init__('TKMeans', random_state=random_state, verbose=verbose)

        self.add_config(k=k)
    
    def prepare_input(self, data):
        data = data.copy()
        ncols = data.shape[1]
        data.columns = [str(i) for i in range(ncols)]
        return data
        
    def create(self, config=None):
        if config:
            k = config[0]
        else:
            k = self.config['k']
        
        return KMeans(n_clusters = k)
    
    def fit(self, data):
        
        if not self.model:
            self.model = self.create()
            
        ncols = data.shape[1]
        data = data.values
        
        self.model.fit(data)
        
        ids_clus = list(set(self.model.labels_))
        reconstructed_matrix = np.ones(data.shape,dtype=int)
        
        print("Reconstruction error: ", np.sum(np.bitwise_xor(data,reconstructed_matrix)))
        
        clustering = build_clustering_output_omega(ids_clus, (self.model.labels_, ncols), trad=True)
        
        # XMEASURES format ground-truth C++ version
        kmeans_clustering_xm = xmeasures_format(clustering)
        df_gt = pd.DataFrame(kmeans_clustering_xm)
        
        return df_gt, clustering

# -----------------------------------------------------------------
#from sklearn.cluster import KMeans
#path_method = "OutputAnalysis\kmeans"
#check_path(path_method)
#
#for ds in range(len(datasets)):
#    ds_name = "Syn-"+str(ds+1)
#    print("\nDataset: "+ds_name)
#    res = os.mkdir(path_method+"\\"+ds_name)
#
#    for run in range(numberOfRuns):
#        print("Run-"+str(run+1))
#        df = pd.read_csv(datasets[ds],header=None) #TODO header?
#        df.columns = [str(i) for i in range(df.shape[1])]
#        ncols = df.shape[1]
#        data = df.values.copy()
#        del df
#
##         kmeans = KMeans(n_clusters = clusters[ds], random_state = 0)
#        kmeans = KMeans(n_clusters = clusters[ds])
#        kmeans.fit(data)
#        ids_clus = list(set(kmeans.labels_))
#        reconstructed_matrix = np.ones(data.shape,dtype=int)
##         print("Data cost: ",data.sum())
#        print("Reconstruction error: ",np.sum(np.bitwise_xor(data,reconstructed_matrix)))
#        del data, reconstructed_matrix
#        gc.collect()
#        
#        clustering = build_clustering_output_omega(ids_clus,(kmeans.labels_,ncols),trad=True)
#
#        # XMEASURES format ground-truth C++ version
#        kmeans_clustering_xm = xmeasures_format(clustering)
#        df_gt = pd.DataFrame(kmeans_clustering_xm)
##         name = datasets[ds].split("/")[-1]
##         name = name.split(".")[0]
#        path = path_method+"/"+ds_name
#        df_gt.to_csv(path.replace("\\","/")+"/run_"+str(run+1)+"_res_kmeans_"+ds_name+"_trad.cnl", 
#                     header= False,index=False, encoding='utf8')
#        del clustering, df_gt, kmeans_clustering_xm
#        gc.collect()