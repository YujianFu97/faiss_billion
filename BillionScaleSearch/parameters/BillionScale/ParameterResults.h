#include "DEEP1B.h"

//For recording
const bool SavingIndex = true;
const bool Recording = true;
bool Retrain = true;
const bool UseOPQ = false;
const bool UsePS = false;

//For construction 
size_t CTrainSize = nt;
size_t nc = 1000000;
size_t PQTrainSize = 100000; 
size_t M_PQ = 8;
size_t CodeBits = 8; 
size_t NBatches = 20; 
size_t BatchSize = nb / NBatches; 
std::string OPQState = UseOPQ ? "_OPQ_":"_";

//For search
const size_t NumRecall = 2;
const size_t NumPara = 10;
size_t RecallK[NumRecall] = {1, 10};
size_t MaxItem[NumPara] = {100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000};
size_t EfSearch[NumPara] = {10, 20, 30, 40, 50, 60, 70,80, 90, 100};
size_t EfNList[NumPara] =  {10, 20, 50, 100, 120, 150, 200, 250, 300, 350};
size_t AccustopItem[NumPara] = {100000};

// For optkmeans
size_t Nlevel = 2;
bool UseGraph = true;
bool UseOptimize = false;
bool ConstrolStart = false;
bool AddiFunction = true;

//For HNSW graph
size_t M = 32;
size_t EfConstruction = 200;

//For PS
size_t IniM = 4000000;
size_t MaxM = 20000000;
size_t CheckBatch = 200000;
size_t Verbose = true;
size_t PQ_TrainSize_PS = 50000; 
size_t K = 10;
float TargetRecall = (0.28) * K; // Target number of gt visited in candidate list, this is related to the recall
float MaxCandidateSize = 50000;    // Maximum candidate list size, note: this is the number of vectors in trainset, i.e. Trainsize * SubsetProp vectors
size_t MaxFailureTimes = 5;
float CheckProp = 0.001;
size_t LowerBound = 10;

std::string NCString_PS = std::to_string(IniM);
std::string NCString    = std::to_string(nc);

// For NClist
size_t SearchK = 10;
const bool UseQuantize = false;
size_t QuantBatchSize = 20;
size_t NLTargetK = 10;
size_t NeighborNum = 20;
size_t Assign_num_batch = 20;
size_t Graph_num_batch = 20;
size_t Assign_batch_size = nb / Assign_num_batch;
size_t ClusterDistNum = 1000; //Distance between vectors and centroids, used in search task
std::string NNDatasetName = "Train"; // Choose one from "Train" or "Base"
//std::string NNDatasetName = "Base";

// File Path
const std::string PathNLFolder = PathFolder + Dataset + "/" + "NLFiles/";
std::string PathDatasetNN = PathNLFolder + NNDatasetName + "NN_" + std::to_string(nc) + "_" + std::to_string(SearchK);
std::string PathCentroidDist = PathNLFolder + "CentroidNeighborDist_" + std::to_string(nc) + "_" + std::to_string(ClusterDistNum);
std::string PathBaseNeighborID = PathNLFolder + "BaseNeighborID_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
std::string PathBaseNeighborDist = PathNLFolder + "BaseNeighborDist_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
std::string PathSubGraphFolder = PathNLFolder + "SubGraphIndexes/";

// Data Path 

// For DEEP1B

const std::string PathLearn =     PathFolder  + Dataset + "/" + Dataset +"_base.fvecs";
const std::string PathBase =      PathFolder  + Dataset + "/" + Dataset +"_base.fvecs";
const std::string PathGt =        PathFolder  + Dataset + "/" + Dataset +"_groundtruth.ivecs";
const std::string PathQuery =     PathFolder  + Dataset + "/" + Dataset +"_query.fvecs";


// For SIFT1B
/*
const std::string PathLearn =     PathFolder  + Dataset + "/" + Dataset +"_learn.bvecs";
const std::string PathBase =      PathFolder  + Dataset + "/" + Dataset +"_base.bvecs";
const std::string PathGt =        PathFolder  + Dataset + "/" + Dataset +"_groundtruth.ivecs";
const std::string PathQuery =     PathFolder  + Dataset + "/" + Dataset +"_query.bvecs";
*/

const std::string PathTrainGt =   PathFolder  + Dataset + "/" + Dataset +"_Train_groundtruth.ivecs";
const std::string PathTrainDist = PathFolder  + Dataset + "/" + Dataset +"_Train_Dist.fvecs";

std::string PathCentroid     = PathFolder + Dataset + "/Centroids_" + NCString + ".fvecs";
std::string PathIniCentroid  = PathFolder + Dataset + "/Centroids_" + NCString + "_" + std::to_string(IniM)+"_Ini.fvecs";

std::string PathBaseIDInv    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Seq";
std::string PathBaseIDSeq    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Inv";

std::string PathCentroidNorm = PathFolder + Dataset + "/CentroidNorms_" + NCString;

std::string PathLearnCentroidNorm = PathFolder + Dataset + "/CentroidNormsLearn_" + NCString;
std::string PathGraphInfo    = PathFolder + Dataset + "/HNSW_" + NCString + "_Info";
std::string PathGraphEdges   = PathFolder + Dataset + "/HNSW_" + NCString + "_Edge";

std::string PathCentroidNeighborID = PathFolder + Dataset + "/CenNeighborID_" + NCString;

std::string PathCentroidNeighborDist = PathFolder + Dataset + "/CenNeighborDist_" + NCString;

std::string PathNeighborList = PathFolder + Dataset + "/NeighborList_" + NCString;


std::string PathPQ           = PathFolder + Dataset + "/PQ_" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathOPQ          = PathFolder + Dataset + "/OPQ_" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString;
std::string PathBaseNorm     = PathFolder + Dataset + "/BaseNorm_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathBaseCode     = PathFolder + Dataset + "/BaseCode_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathOPQCentroids = PathFolder + Dataset + "/OPQCentroids_" + "_OPQ_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString +".fvecs";


std::string PathCentroid_PS     = PathFolder + Dataset + "/Centroids_" + NCString_PS + ".fvecs";
std::string PathIniCentroid_PS  = PathFolder + Dataset + "/Centroids_" + NCString_PS + "_" + std::to_string(IniM)+"_Ini.fvecs";

std::string PathBaseIDInv_PS    = PathFolder + Dataset + "/BaseID_nc" + NCString_PS + "_Seq";
std::string PathBaseIDSeq_PS    = PathFolder + Dataset + "/BaseID_nc" + NCString_PS + "_Inv";

std::string PathCentroidNorm_PS = PathFolder + Dataset + "/CentroidNorms_" + NCString_PS;

std::string PathLearnCentroidNorm_PS = PathFolder + Dataset + "/CentroidNormsLearn_" + NCString_PS;
std::string PathGraphInfo_PS    = PathFolder + Dataset + "/HNSW_" + NCString_PS + "_Info";
std::string PathGraphEdges_PS   = PathFolder + Dataset + "/HNSW_" + NCString_PS + "_Edge";

std::string PathCentroidNeighborID_PS = PathFolder + Dataset + "/CenNeighborID_" + NCString_PS;

std::string PathCentroidNeighborDist_PS = PathFolder + Dataset + "/CenNeighborDist_" + NCString_PS;

std::string PathNeighborList_PS = PathFolder + Dataset + "/NeighborList_" + NCString_PS;


std::string OPQState_PS         = UseOPQ ? "_OPQ_":"_";
std::string PathPQ_PS           = PathFolder + Dataset + "/PQ_" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString_PS;
std::string PathOPQ_PS          = PathFolder + Dataset + "/OPQ_" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString_PS;
std::string PathBaseNorm_PS     = PathFolder + Dataset + "/BaseNorm_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString_PS;
std::string PathBaseCode_PS     = PathFolder + Dataset + "/BaseCode_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString_PS;
std::string PathOPQCentroids_PS = PathFolder + Dataset + "/OPQCentroids_" + "_OPQ_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString_PS +".fvecs";
