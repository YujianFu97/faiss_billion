#include "Index/BIndex.h"
#include "parameters/MillionScale/ParameterResults.h"

int main(){
    Retrain = true;
    std::string PathRecord = PathFolder + Dataset + "/RecordPQ.txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    BIndex * Index = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);
    time_recorder Trecorder = time_recorder();

    float Err = Index->TrainCentroids(CTrainSize, PathLearn, PathCentroid, PathCentroidNorm, UseOptimize, Nlevel, OptSize, Lambda);
    if (Recording){RecordFile << "Centroid Training Time: " << Trecorder.getTimeConsumption() / 1000000 << " Training Error: " << Err / CTrainSize << std::endl;};

    if (Recording){RecordFile << " M: " << M << " EfConstruction: " << EfConstruction;}
    Index->BuildGraph(M, EfConstruction, PathGraphInfo, PathGraphEdges, PathCentroid);
    if (Recording){RecordFile << " Graph Construction Time: " << Trecorder.getTimeConsumption() / 1000000 << std::endl;}

    if (Recording){RecordFile << " PQ_M: " << M_PQ << " Codebits: " << CodeBits << " PQ Train Size: " << PQTrainSize << " Use OPQ: " << UseOPQ << std::endl;
    Index->TrainQuantizer(PQTrainSize, PathLearn, PathPQ, PathOPQ);
    if (Recording){RecordFile << "PQ Training time: " << Trecorder.getTimeConsumption() / 1000000 << std::endl;}

    if(Recording){RecordFile << Index->QuantizationFeature(PathBase, PathBaseIDSeq) << std::endl;}
    
    return 1;
}


