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

    auto result = neighborkmeans(TrainSet.data(), Dimension, nt, nc, prop, Nlevel, 10, ClusterBoundSize, Centroids.data(), true, false, UseOptimize, Lambda, OptSize, UseGraph);
    
}