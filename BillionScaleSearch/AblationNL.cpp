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

    auto NeighborConflictMap = neighborkmeans(TrainSet.data(), Dimension, nt, nc, prop, Nlevel, 10, ClusterBoundSize, Centroids.data(), Verbose, UseOptimize, Lambda, OptSize, UseGraph, 30, TrainLabels.data(), TrainDists.data());
    std::cout << "The total number of boundary conflict record: " << NeighborConflictMap.size() << " The total number of centroids: " << nc << "\n";
    
    // Assignment of the base vectors: find the nearest trainset vectors
    std::vector<std::vector<uint32_t>> TrainIds(nc);
    for (size_t i = 0; i < nt; i++){
        TrainIds[TrainLabels[i]].emplace_back(i);
    }

    size_t ClusterNum = 10; assert(ClusterNum >1);
    hnswlib::HierarchicalNSW * GraphHNSW = new hnswlib::HierarchicalNSW(Dimension, nc, M, 2*M, EfConstruction); for (size_t i = 0; i < nc; i++){GraphHNSW->addPoint(Centroids.data() + i * Dimension);}
    std::vector<float> BaseSet(nb * Dimension);
    std::vector<uint32_t> BaseIDSeq(nb);

    for (size_t i = 0; i < nb; i++){
        auto result = GraphHNSW->searchKnn(BaseSet.data() + i * Dimension, ClusterNum);
        float VectorDist = std::numeric_limits<float>::max();
        for (size_t j = 0; j < ClusterNum; j++){
            uint32_t ClusterID = result.top().second;
            size_t ClusterSize = TrainIds[ClusterID].size();
            std::vector<float> ClusterVectorDist(ClusterSize);
            for (size_t k = 0; k < ClusterSize; k++){
                uint32_t TrainVectorID = TrainIds[ClusterID][j];
                float TrainVectorDist = faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, TrainSet.data() + TrainVectorID * Dimension, Dimension);
                if (TrainVectorDist < VectorDist){
                    BaseIDSeq[i] = ClusterID;
                    VectorDist = TrainVectorDist;
                }
            }
        }
    }

    // Build the neighbor list for index
    auto comp = [](std::tuple<float, uint32_t, uint32_t> Element1, std::tuple<float, uint32_t, uint32_t> Element2){return std::get<0>(Element1) < std::get<0>(Element2);};
    std::priority_queue<std::tuple<float, uint32_t, uint32_t>, std::vector<std::tuple<float, uint32_t, uint32_t>>, decltype(comp)> NeighborListQueue(comp);
    for (auto it = NeighborConflictMap.begin(); it != NeighborConflictMap.end(); it++){
        //NeighborListQueue.emplace(std::make_tuple())
    }



}