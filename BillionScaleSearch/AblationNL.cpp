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

    auto result = neighborkmeans(TrainSet.data(), Dimension, nt, nc, prop, Nlevel, 10, ClusterBoundSize, Centroids.data(), Verbose, UseOptimize, Lambda, OptSize, UseGraph, 30, TrainLabels.data(), TrainDists.data());
    std::cout << "The total number of boundary conflict record: " << result.size() << " The total number of centroids: " << nc << "\n";
    
    // Assignment of the base vectors: find the nearest trainset vectors
    std::vector<std::vector<uint32_t>> TrainIds(nc);
    for (size_t i = 0; i < nt; i++){

    }


}