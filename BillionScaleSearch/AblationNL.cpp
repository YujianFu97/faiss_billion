#include "Index/BIndex.h"
#include "parameters/MillionScale/ParameterResults.h"


int main(){

    std::string PathRecord = PathFolder + Dataset + "/RecordNL.txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    float prop = 0;
    std::vector<float> TrainSet(nt * Dimension);
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream TrainInput(PathLearn, std::ios::binary);
    readXvec<float>(TrainInput, TrainSet.data(), Dimension, nt, true, true);

    std::vector<uint32_t> TrainLabels(nt);
    std::vector<float> TrainDists(nt);
    std::vector<std::vector<uint32_t>> VectorOutIDs(nc);
    std::vector<std::vector<uint32_t>> TrainIds(nc);

    auto NeighborConflictMap = neighborkmeans(TrainSet.data(), Dimension, nt, nc, prop, Nlevel, 10, ClusterBoundSize, Centroids.data(), Verbose, UseOptimize, TrainIds, VectorOutIDs, Lambda, OptSize, UseGraph, 30, TrainLabels.data(), TrainDists.data());
    std::cout << "The total number of boundary conflict record: " << NeighborConflictMap.size() << " The total number of centroids: " << nc << "\n";

    // Select the neighbor list that save the most search cost
    // Struct: <Total cost saved, boundary dist, target cluster, NN cluster>
    auto comp = [](std::tuple<size_t, float, uint32_t, uint32_t> Element1, std::tuple<size_t, float, uint32_t, uint32_t> Element2){return std::get<0>(Element1) < std::get<0>(Element2);};
    std::priority_queue<std::tuple<size_t, float, uint32_t, uint32_t>, std::vector<std::tuple<size_t, float, uint32_t, uint32_t>>, decltype(comp)> NeighborListQueue(comp);
    for (auto it = NeighborConflictMap.begin(); it != NeighborConflictMap.end(); it++){

        size_t NeighborListSize = 0;
        NeighborListQueue.emplace(std::make_tuple(std::get<0>((*it).second)* std::get<2>((*it).second), std::get<1>((*it).second), (*it).first.first, (*it).first.second ));
    }

    size_t MaxNumNeighborList = 10000;
    MaxNumNeighborList = MaxNumNeighborList < NeighborListQueue.size() ? MaxNumNeighborList : NeighborListQueue.size();
    std::map<uint32_t, std::vector<std::pair<uint32_t, float>>> NeighborListMapNNCluster;
    for (size_t i = 0; i < MaxNumNeighborList; i++){
        auto EachList = NeighborListQueue.top();
        uint32_t TargetClusterID = std::get<2>(EachList);
        uint32_t NNClusterID = std::get<3>(EachList);

        auto result = NeighborListMapNNCluster.find(NNClusterID);
        if (result != NeighborListMapNNCluster.end()){
            NeighborListMapNNCluster[NNClusterID].emplace_back(std::make_pair(TargetClusterID, std::get<1>(EachList)));
        }
        else{
            NeighborListMapNNCluster[NNClusterID] = {std::make_pair(TargetClusterID, std::get<1>(EachList))};
        }
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> NeighborList;

    // Assignment of the base vectors: find the nearest trainset vectors
    hnswlib::HierarchicalNSW * GraphHNSW = new hnswlib::HierarchicalNSW(Dimension, nc, M, 2*M, EfConstruction); for (size_t i = 0; i < nc; i++){GraphHNSW->addPoint(Centroids.data() + i * Dimension);}
    std::vector<float> BaseSet(nb * Dimension);
    std::vector<uint32_t> BaseIDSeq(nb);
    std::vector<std::vector<uint32_t>> BaseNListSeq(nb);

    // Only search two types of train vector in assignment: Vectors in the target cluster, or vectors that are shifted out from the target cluster
    // Firstly search all vectors that are shifted, then the in the target cluster.
    for (size_t i = 0; i < nb; i++){
        auto result = GraphHNSW->searchKnn(BaseSet.data() + i * Dimension, GraphHNSW->efSearch);
        std::vector<uint32_t> VectorClusterID(GraphHNSW->efSearch);
        std::vector<float> VectorClusterDist(GraphHNSW->efSearch);
        for (size_t j = 0; j < GraphHNSW->efSearch; j++){
            VectorClusterID[GraphHNSW->efSearch - j - 1] = result.top().second;
            VectorClusterDist[GraphHNSW->efSearch - j - 1] = result.top().first;
            result.pop();
        }
        float VectorDist = std::numeric_limits<float>::max();
        uint32_t ClusterID = VectorClusterID[0];
        size_t ShiftSize = VectorOutIDs[ClusterID].size();
        // Firstly check all vectors that are shifted out from the target cluster
        for (size_t j = 0; j < ShiftSize; j++){
            uint32_t NNVectorID = VectorOutIDs[ClusterID][j];
            float TrainVectorDist = faiss::fvec_L2sqr(BaseSet.data() + i  * Dimension, TrainSet.data() + NNVectorID * Dimension, Dimension);
            if (TrainVectorDist < VectorDist){
                VectorDist = TrainVectorDist;
                BaseIDSeq[i] = TrainLabels[NNVectorID];
            }
        }

        // If there is one vector in the target cluster that smaller than all vectors in other clusters
        for (size_t j = 0; j < TrainIds[ClusterID].size(); j++){
            float TrainVectorDist = faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, TrainSet.data() + TrainIds[ClusterID][j] * Dimension, Dimension);
            if (TrainVectorDist < VectorDist){
                VectorDist = TrainVectorDist;
                BaseIDSeq[i] = ClusterID;
                break;
            }
        }

        // Incldue the neighbor list that is related to the base vector
        auto MapResult = NeighborListMapNNCluster.find(BaseIDSeq[i]);
        if (MapResult != NeighborListMapNNCluster.end()){
            size_t NListSize = NeighborListMapNNCluster[BaseIDSeq[i]].size();
            for (size_t j = 0; j < NListSize; j++){
                uint32_t TargetClusterID = NeighborListMapNNCluster[BaseIDSeq[i]][j].first;
                float TargetBoundaryDist = NeighborListMapNNCluster[BaseIDSeq[i]][j].second;

                float VectorTargetClusterDist = -1;
                for (size_t k = 0; k < GraphHNSW->efSearch; k++){
                    if (VectorClusterID[k] == TargetClusterID){
                        VectorTargetClusterDist = VectorClusterDist[k];
                        break;
                    }
                }
                if (VectorTargetClusterDist < 0){
                    VectorTargetClusterDist = faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
                }

                if (VectorTargetClusterDist < TargetBoundaryDist){
                    BaseNListSeq[i].emplace_back(TargetClusterID);
                }
            }
        }
    }

    for (uint32_t i = 0; i < nb; i++){
        size_t NumVectorList = BaseNListSeq[i].size();
        for (size_t j = 0; j < NumVectorList; j++){
            std::pair<uint32_t, uint32_t> NLIndex = std::make_pair(BaseNListSeq[i][j], BaseIDSeq[i]);
            auto result = NeighborList.find(NLIndex);
            if (result != NeighborList.end()){
                NeighborList[NLIndex].emplace_back(i);
            }
            else{
                NeighborList[NLIndex] = {i};
            }
        }
    }

    std::vector<std::vector<uint32_t>> BaseIds(nc);
    for (uint32_t i = 0; i < nb; i++){
        BaseIds[BaseIDSeq[i]].emplace_back(i);
    }

    // Search the index with NL
    std::vector<float> QuerySet(nq * Dimension);
    std::vector<uint32_t> GTSet(ngt * nq);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    std::ifstream GtInput(PathGt, std::ios::binary);
    readXvecFvec<DataType> (QueryInput, QuerySet.data(), Dimension, nq, true, true);

    for (size_t ParaIdx = 0; ParaIdx < NumPara; ParaIdx++){
        for (size_t i = 0; i < nq; i++){
            size_t VisitedGt = 0;
            std::vector<uint32_t> ClusterID(EfSearch[ParaIdx]);
            std::vector<float> ClusterDist(EfSearch[ParaIdx]);
            auto result = GraphHNSW->searchBaseLayer(QuerySet.data() + i * Dimension, EfSearch[ParaIdx]);
            for (size_t j = 0; j < EfSearch[ParaIdx]; j++){
                ClusterID[EfSearch[ParaIdx] - j - 1] = result.top().second;
                ClusterDist[EfSearch[ParaIdx] - j - 1] = result.top().first;
            }
            // Check the vectors in the neighbor cluster (not in the neighbor list) and the vectors in the neighbor list
            uint32_t TargetClusterID = ClusterID[0];
            for (size_t j = 0; j < BaseIds[TargetClusterID].size(); j++){
                
            }

        }
    }




}