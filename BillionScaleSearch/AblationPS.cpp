#include "Index/BIndex.h"
#include "parameters/BillionScale/ParameterResults.h"
// Basic Theory

int main(){

    Retrain = true;
    bool TestSearch = true;
    bool TestPS = true;
    bool TestNonPS = false;
    std::string PathRecord = PathFolder + Dataset + "/RecordPS.txt";
    std::ofstream RecordFile;

    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));

    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }
    time_recorder Trecorder = time_recorder();
    if (TestPS){
    if (Recording){
        RecordFile<<"\n----------------------- Start PS training process----------------------- \n";
        RecordFile << "The parameter setting: Initialization M: " << IniM << " Maximum M: " << MaxM << " CheckBatch: " << CheckBatch << " MaxCanSize: " << 
                    MaxCandidateSize << " Recall@K: " << K  << " Target Recall: " << TargetRecall << "\n\n";
        }
    BIndex * Index = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);

    std::cout << "\n----------------------- Learn Centroids, Graph, Vector Assignment and Product Quantizer--------------------------\n";
    std::cout << "Nlevel: " << Nlevel << " OptSize: " << OptSize << " Lambda: " <<Lambda << "\n";
    PathFolder = PathFolder + Dataset + "/PS_" + NCString_PS + "/";
    PrepareFolder((char *) PathFolder.c_str());
    exit(0);

    uint32_t NC = Index->LearnCentroidsINI(CTrainSize, nq, UseOptimize, IniM, MaxM,  TargetRecall, MaxCandidateSize, K, MaxFailureTimes, CheckProp, LowerBound, CheckBatch, M_PQ, CodeBits, PQ_TrainSize_PS, M, EfConstruction, Nlevel, OptSize, Lambda, ngt, PathFolder, PathGt, PathBase, PathQuery, PathLearn, RecordFile);
    if (Recording){RecordFile << "Centroid Training Time: " << Trecorder.getTimeConsumption() / 1000000 << " nc Result: " << NC << std::endl;};


    if (Recording){RecordFile << Index->ClusterFeature();}

    if (!TestSearch)
    return 0;

    RecordFile << " Graph data: efConstruction " << Index->CentroidHNSW->efConstruction_ << " efSearch: " << Index->CentroidHNSW->efSearch << " M: " << Index->CentroidHNSW->maxM_ << " Num of nodes: " << Index->CentroidHNSW->maxelements_ << "\n";
    std::cout << "\n----------------------- Index Evaluation--------------------------\n";
    if(Recording){RecordFile << Index->Eval(PathQuery, PathGt, nq, ngt, NumRecall, NumPara, RecallK, MaxItem, EfSearch);}
    }

    if (TestNonPS){

    std::cout << "\n----------------------- Base index training--------\n";
    BIndex * BaseIndex = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);
    Trecorder.reset();


    if (Recording){RecordFile << "\n----------------------- Base index training--------\n";}
    BaseIndex->TrainCentroids(CTrainSize, PathLearn, PathCentroid, PathCentroidNorm, UseOptimize, UseGraph, Nlevel, OptSize, Lambda);
    if (Recording){RecordFile << "Centroid Training Time: " << Trecorder.getTimeConsumption() / 1000000 << " nc Result: " << nc << std::endl;};
    
    BaseIndex->Retrain = true;
    BaseIndex->BuildGraph(M, EfConstruction,PathGraphInfo, PathGraphEdges, PathCentroid);
    if (Recording){RecordFile << " Graph Construction Time: " << Trecorder.getTimeConsumption() / 1000000 << std::endl;}

    BaseIndex->AssignBaseset(NBatches, BatchSize, PathBase, PathBaseIDInv, PathBaseIDSeq);
    if (Recording) {RecordFile << "BaseSet Assignment Time: " << Trecorder.getTimeConsumption() / 1000000<< std::endl;}
    if (Recording){RecordFile << BaseIndex->ClusterFeature();}

    if (!TestSearch)
    return 0;

    BaseIndex->TrainQuantizer(PQTrainSize, PathLearn, PathPQ, PathOPQ);
    if (Recording) {RecordFile << "Train Quantizer Time: " << Trecorder.getTimeConsumption() / 1000000<< std::endl;}
    
    BaseIndex->QuantizeBaseset(NBatches, BatchSize, PathBase, PathBaseIDSeq, PathBaseCode, PathBaseNorm, PathOPQCentroids);

    if (Recording) {RecordFile << "BaseSet Quantization Time: " << Trecorder.getTimeConsumption() / 1000000<< std::endl;}
    if(Recording){RecordFile << BaseIndex->Eval(PathQuery, PathGt, nq, ngt, NumRecall, NumPara, RecallK, MaxItem, EfSearch);}
    }

    return 1;
}