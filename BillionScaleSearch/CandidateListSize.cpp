#include "./Index/BIndex.h"
#include "./parameters/BillionScale/ParameterResults.h"


int main(){

    Retrain = false;
    std::string PathRecord = PathFolder + Dataset + "/CandidateListSize" + NCString + ".txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    NCString    = std::to_string(nc);
    PathCentroid     = PathFolder + Dataset + "/Centroids_" + NCString + ".fvecs";
    PathIniCentroid  = PathFolder + Dataset + "/Centroids_" + NCString + "_" + std::to_string(IniM)+"_Ini.fvecs";

    PathBaseIDInv    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Seq";
    PathBaseIDSeq    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Inv";

    PathCentroidNorm = PathFolder + Dataset + "/CentroidNorms_" + NCString;

    PathLearnCentroidNorm = PathFolder + Dataset + "/CentroidNormsLearn_" + NCString;
    PathGraphInfo    = PathFolder + Dataset + "/HNSW_" + NCString + "_Info";
    PathGraphEdges   = PathFolder + Dataset + "/HNSW_" + NCString + "_Edge";

    BIndex * Index = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);
    time_recorder Trecorder = time_recorder();

    float Err = Index->TrainCentroids(CTrainSize, PathLearn, PathCentroid, PathCentroidNorm, UseOptimize, UseGraph, Nlevel, OptSize, Lambda);
    if (Recording){RecordFile << "Centroid Training Time: " << Trecorder.getTimeConsumption() / 1000000 << " Training Error: " << Err / CTrainSize << std::endl;};

    if (Recording){RecordFile << " M: " << M << " EfConstruction: " << EfConstruction;}
    Index->BuildGraph(M, EfConstruction, PathGraphInfo, PathGraphEdges, PathCentroid);
    if (Recording){RecordFile << " Graph Construction Time: " << Trecorder.getTimeConsumption() / 1000000 << std::endl;}

    BatchSize = CTrainSize / NBatches;
    PathBaseIDInv = PathBaseIDInv + "_Train";
    PathBaseIDSeq = PathBaseIDSeq + "_Train";
    if (Recording){RecordFile << " nbatch: " << NBatches << " BatchSize: " << BatchSize << std::endl;}
    Index->AssignBaseset(NBatches, BatchSize, PathLearn, PathBaseIDInv, PathBaseIDSeq);
    if (Recording) {RecordFile << "TrainSet Assignment Time: " << Trecorder.getTimeConsumption() / 1000000<< std::endl;}

    if (Recording){RecordFile << Index->ClusterFeature();}

    /*
    size_t EfSearch = nc / 10 > 100 ? nc / 10 : 100;
    size_t Scale = nb / CTrainSize;
    if(Recording){RecordFile << Index->NeighborCost(Scale, PathQuery, PathGt, nq, ngt, EfSearch, TargetRecall);}
    */

    return 1;

}