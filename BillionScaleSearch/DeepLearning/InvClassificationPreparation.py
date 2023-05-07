from sklearn.cluster import KMeans
import utils
import numpy as np
BaseFilePath = '/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_base.fvecs'
QueryFilePath  = '/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_query.fvecs'
FolderPath = '/home/yujianfu/Desktop/Dataset/SIFT1M/DLFiles/'

NCluster = 200

if __name__ == '__main__':
    BaseSet = utils.fvecs_read(BaseFilePath)
    ClusteringResult = KMeans(n_clusters=NCluster, random_state=0, n_init="auto", verbose=True).fit(BaseSet)

    CenFilePath = FolderPath + 'Centroids_' + str(NCluster)
    BaseAssignFilePath = FolderPath + 'BaseAssignment_' + str(NCluster) + '.npy'
    np.save(CenFilePath, ClusteringResult.cluster_centers_)
    np.save(BaseAssignFilePath, ClusteringResult.labels_)

    QuerySet =utils.fvecs_read(QueryFilePath)
    QueryLabels = ClusteringResult.predict(QuerySet)
    QueryAssignFilePath = FolderPath + 'QueryAssignment_' + str(NCluster) + '.npy'
    np.save(QueryAssignFilePath, QueryLabels)
