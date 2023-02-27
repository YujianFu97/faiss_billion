#include "Index/BIndex.h"
#include "parameters/MillionScale/ParameterResults.h"
#include "float.h"

int main(){

    // We test the number of neighbors in neighbor cluster and the 
    // cost for searching the nearest neighbors
    Retrain = false;
    std::string PathRecord = PathFolder + Dataset + "/BCProp.txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    std::cout << "Load data from " << PathLearn << "\n";
    BIndex * Index = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);
    time_recorder Trecorder = time_recorder();

    float Err = Index->TrainCentroids(CTrainSize, PathLearn, PathCentroid, PathCentroidNorm, UseOptimize, Nlevel, OptSize, Lambda);
    if (Recording){RecordFile << "Centroid Training Time: " << Trecorder.getTimeConsumption() / 1000000 << " Training Error: " << Err / CTrainSize << std::endl;};

    if (Recording){RecordFile << " M: " << M << " EfConstruction: " << EfConstruction;}
    Index->BuildGraph(M, EfConstruction, PathGraphInfo, PathGraphEdges, PathCentroid);
    if (Recording){RecordFile << " Graph Construction Time: " << Trecorder.getTimeConsumption() / 1000000 << std::endl;}

    if (Recording){RecordFile << " nbatch: " << NBatches << " BatchSize: " << BatchSize << std::endl;}
    Index->AssignBaseset(NBatches, BatchSize, PathBase, PathBaseIDInv, PathBaseIDSeq);
    if (Recording) {RecordFile << "BaseSet Assignment Time: " << Trecorder.getTimeConsumption() / 1000000<< std::endl;}

    if (Recording){RecordFile << Index->ClusterFeature();}

    for(size_t i = 0; i < 10; i++){
        for (size_t j = 0; j < Dimension; j++){
            std::cout << Index->CentroidHNSW->getDataByInternalId(i)[j] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<float>(QueryInput, Query.data(), Dimension, nq, true, true);
    GTInput.close(); QueryInput.close();
    std::vector<float> BaseSet(nb * Dimension);
    std::ifstream BaseInput(PathBase, std::ios::binary);
    readXvecFvec<float>(BaseInput, BaseSet.data(), Dimension, nb, true, true);
    std::vector<uint32_t> BaseIdsSeq(nb);
    std::ifstream BaseIdSeqInput(PathBaseIDSeq, std::ios::binary);
    BaseIdSeqInput.read((char *)BaseIdsSeq.data(), nb * sizeof(uint32_t));

    std::cout << "Start Evaluate the query" << std::endl;
    size_t ef = 64;
    size_t KN = 50;

    // Larger: more nn in the neighbor clusters
    std::vector<uint32_t> NIN(KN * nq, 0);
    // New Cost that need to be searched in neighbor cluster (with distance ranking)
    std::vector<float> CND(KN * nq, 0);
    // New Cost that need to be searched in neighbor cluster (with with neighbor ranking)
    std::vector<float> CNN(KN * nq, 0);
    // New Cost that need to be searched in neighbor cluster (with with neighbor ranking and distance bound)
    std::vector<float> CNNB(KN * nq, 0);

    std::vector<size_t> FailNum(KN * nq, 0);
    std::vector<size_t> CentralClusterSize(nq, 0);

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){

        std::vector<uint32_t> NeighborID(ef);
        std::vector<float> NeighborDist(ef);
        auto result = Index->CentroidHNSW->searchBaseLayer(Query.data() + i * Dimension, ef);
        for (size_t j = 0; j < ef; j++){
            NeighborID[ef-j-1] = result.top().second;
            NeighborDist[ef-j-1] = result.top().first;
            result.pop();
        }

        CentralClusterSize[i] = Index->BaseIds[NeighborID[0]].size();
        std::vector<float> NeighborClusterDist(ef); NeighborClusterDist[0] = 0;
        std::vector<uint32_t> NeighborBoundID(ef);
        //std::cout << "Compute the query dist bound\n";

        float alpha = 0;
        std::vector<float> ClusterNeighborDist(ef);
        for (size_t j = 1; j < ef; j++){
            ClusterNeighborDist[j] = faiss::fvec_L2sqr(Index->CentroidHNSW->getDataByInternalId(NeighborID[0]), Index->CentroidHNSW->getDataByInternalId(NeighborID[j]), Dimension);
            float costheta = (NeighborDist[0] + ClusterNeighborDist[j] - NeighborDist[j]) / ( 2 * std::sqrt(ClusterNeighborDist[j] * NeighborDist[0]));

            float d = std::sqrt(ClusterNeighborDist[j]) / 2 - costheta * std::sqrt(NeighborDist[0]);
            float hsquare = NeighborDist[0]* (1 -  costheta * costheta);
            NeighborClusterDist[j] = d * d + alpha * alpha * hsquare;
        }
        
        //std::cout << "Sort the neighbor clusters based on boundary distance\n";
        std::vector<uint32_t> BoundDistIdx(ef); std::iota(BoundDistIdx.begin(), BoundDistIdx.end(), 0);
        std::sort(BoundDistIdx.begin(), BoundDistIdx.end(), [& NeighborClusterDist] (size_t i1, size_t i2){return NeighborClusterDist[i1] < NeighborClusterDist[i2];});
        for (size_t j = 0; j < ef; j++){
            //std::cout << NeighborID[BoundDistIdx[j]] << " " << Index->BaseIds[NeighborID[BoundDistIdx[j]]].size() << " ";
        }
        //std::cout << std::endl;

        std::unordered_set<uint32_t> VisitedClusterNeighbor;
        std::unordered_set<uint32_t> VisitedClusterDist;
        VisitedClusterNeighbor.insert(NeighborID[0]);
        VisitedClusterDist.insert(NeighborID[0]);

        std::vector<std::vector<float>> VectorCentralDist(ef);
        for (size_t j = 1; j < ef; j++){
            VectorCentralDist[j].resize(Index->BaseIds[NeighborID[j]].size());
            for (size_t k = 0; k < Index->BaseIds[NeighborID[j]].size(); k++){
                VectorCentralDist[j][k] = faiss::fvec_L2sqr(BaseSet.data() + Dimension * Index->BaseIds[NeighborID[j]][k], Index->CentroidHNSW->getDataByInternalId(NeighborID[0]), Dimension);
            }
        }

        std::vector<float> SearchedBoundDist(ef, 0);
        SearchedBoundDist[0] = 10^38;
        

        for (size_t j = 0; j < KN; j++){
            uint32_t GtCluster = BaseIdsSeq[GT[i * ngt + j]];
            if (GtCluster == NeighborID[0]){
                continue;
            }
            NIN[i * KN + j] += 1;

            auto ClusterIndice = std::find(NeighborID.begin(), NeighborID.end(), GtCluster); 
            if(ClusterIndice == NeighborID.end()){
                FailNum[i * KN + j] ++;
                continue;
            }

            if (VisitedClusterDist.count(GtCluster) == 0){
                for (size_t k = 0; k < ef; k++){
                    if (VisitedClusterDist.count(NeighborID[k]) == 0){
                        CND[i * KN + j] += Index->BaseIds[NeighborID[k]].size();
                        VisitedClusterDist.insert(NeighborID[k]);
                    }
                    if (NeighborID[k] == GtCluster){
                        break;
                    }
                }
            }

            float NCDist = faiss::fvec_L2sqr(BaseSet.data() + GT[i * ngt + j] * Dimension, Index->CentroidHNSW->getDataByInternalId(NeighborID[0]), Dimension);
            if (VisitedClusterNeighbor.count(GtCluster) == 0){
                for (size_t k = 0; k < ef; k++){
                    if (VisitedClusterNeighbor.count(NeighborID[BoundDistIdx[k]]) == 0){
                        CNN[i * KN + j] += Index->BaseIds[NeighborID[BoundDistIdx[k]]].size();
                        VisitedClusterNeighbor.insert(NeighborID[BoundDistIdx[k]]);
                        
                    }
                    if (NeighborID[BoundDistIdx[k]] == GtCluster){
                        break;
                    }
                }
            }

            for (size_t k = 0; k < ef; k++){

                if (NCDist > SearchedBoundDist[BoundDistIdx[k]]){
                    for (size_t m = 0; m < VectorCentralDist[BoundDistIdx[k]].size(); m++){
                        if (SearchedBoundDist[BoundDistIdx[k]]< VectorCentralDist[BoundDistIdx[k]][m] && VectorCentralDist[BoundDistIdx[k]][m] <= NCDist){
                            CNNB[i * KN + j] ++;
                        }
                    }
                    SearchedBoundDist[BoundDistIdx[k]] = NCDist;

                }
                
                if (NeighborID[BoundDistIdx[k]] == GtCluster){
                        break;
                }
            }
        }
    }

    float AVGCentralSize = 0;
    for (size_t i = 0; i < nq; i++){
        AVGCentralSize += CentralClusterSize[i];
    }
    std::cout << "The Central Size: " << float(AVGCentralSize) / nq << std::endl; RecordFile << " The Central Size: " << float(AVGCentralSize) / nq << std::endl;
    float Accu = 0;std::cout << "Prop of Neighbor Vectors in Neighbor Cluster: "; RecordFile << "Prop of Neighbor Vectors in Neighbor Cluster: ";
    for (size_t i = 0; i < KN; i++){
        for (size_t j = 0; j < nq; j++){
            Accu += NIN[j * KN + i];
        }
        std::cout << Accu / (nq) << ", "; RecordFile << Accu / (nq) << ", ";
    }
    Accu = 0; std::cout << "\nNum of vectors searched in distance: "; RecordFile << "\nNum of vectors searched in distance: ";
    for (size_t i = 0; i < KN; i++){
        for (size_t j = 0; j < nq; j++){
            Accu += CND[j * KN + i];
        }
        std::cout << Accu / (nq) << ", "; RecordFile << Accu / (nq) << ", ";
    }
    Accu = 0; std::cout << "\nNum of vectors searched in neighbor: ";RecordFile << "\nNum of vectors searched in neighbor: ";
    for (size_t i = 0; i < KN; i++){
        for (size_t j = 0; j < nq; j++){
            Accu += CNN[j * KN + i]; 
        }
        std::cout << Accu / (nq) << ", ";RecordFile << Accu / (nq) << ", ";
    }
    Accu = 0; std::cout << "\nNum of vectors searched in N+DistBound: ";RecordFile << "\nNum of vectors searched in N+DistBound: ";
    for (size_t i = 0; i < KN; i++){
        for (size_t j = 0; j < nq; j++){
            Accu += CNNB[j * KN + i]; 
        }
        std::cout << Accu / (nq) << ", "; RecordFile << Accu / (nq) << ", ";
    }
    Accu = 0; std::cout << "\nNum of vectors Failed: ";RecordFile << "\nNum of vectors Failed: ";
    for (size_t i = 0; i < KN; i++){
        for (size_t j = 0; j < nq; j++){
            Accu += FailNum[j * KN + i]; 
        }
        std::cout << Accu  / nq << ", "; RecordFile << Accu / (nq) << ", ";
    }
    return 1;
}
