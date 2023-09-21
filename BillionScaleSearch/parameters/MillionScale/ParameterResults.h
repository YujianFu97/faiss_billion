#include "./SIFT1M.h"

//For recording
const bool SavingIndex = true;
const bool Recording = true;
bool Retrain = false;
const bool UseOPQ = false;
const bool UsePS = false;

//For index construction
const size_t CTrainSize = nt;     // Number of vectors used for initialization centroid training
const size_t nc = 1000;
size_t PQTrainSize = 20000;
size_t M_PQ =32;
size_t CodeBits = 8;
size_t NBatches = 1;
std::string OPQState = UseOPQ ? "_OPQ_":"_";

//For search
const size_t NumRecall = 2;
size_t RecallK[NumRecall] = {1, 5};
const size_t NumPara = 10;
size_t MaxItem[NumPara] =           {10000, 1000, 5000, 8000, 10000, 15000, 20000, 25000, 30000, 35000};
size_t EfSearch[NumPara] =          {10, 50, 50, 50, 50, 50, 50, 50, 50, 50}; 
size_t EfNList[NumPara] =           {200, 50, 100, 500, 600, 1000, 1300, 1500, 2000, 2300};
//This is the number of checked vectors but not update 
//size_t MaxNoUpdate[NumPara] = {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000};


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
size_t IniM = 30000;
size_t MaxM = 100000;
size_t CheckBatch = 1000;
size_t Verbose = true;
size_t PQ_TrainSize_PS = 10000; 
size_t K = 1;
float TargetRecall = (0.36) * K; // Target number of gt visited in candidate list, this is related to the recall
float MaxCandidateSize = 10000;    // Maximum candidate list size, note: this is the number of vectors in trainset, i.e. Trainsize * SubsetProp vectors
size_t MaxFailureTimes = 5;
float CheckProp = 0.01;
size_t LowerBound = 20;

std::string NCString_PS = std::to_string(IniM);
std::string NCString    = std::to_string(nc);

// For NClist
const bool UseQuantize = false; // UseQuantize = True is not implemented yet
size_t QuantBatchSize = 20; 

size_t SearchK = 10; // The number of NN in search, use SearchK > TargetK
size_t NLTargetK = 10; 
size_t NeighborNum = 20; // The number of neighbor clusters to be computed
size_t Assign_num_batch = 2; // Assignment batch num
size_t Graph_num_batch = 2;  // Batch num of 
size_t Assign_batch_size = nb / Assign_num_batch;
size_t ClusterDistNum = 1000; //Distance between vectors and centroids, used in search task
//std::string NNDatasetName = "Train"; // Choose one from "Train" or "Base"
std::string NNDatasetName = "Train";

// File Path
const std::string PathNLFolder = PathFolder + Dataset + "/" + "NLFiles/";
std::string PathDatasetNN = PathNLFolder + NNDatasetName + "NN_" + std::to_string(nt) + "_" + std::to_string(SearchK);
std::string PathCentroidNLNorm = PathNLFolder + "CentroidNeighborNorm_" + std::to_string(nc);
std::string PathBaseNeighborID = PathNLFolder + "BaseNeighborID_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
std::string PathBaseNeighborDist = PathNLFolder + "BaseNeighborDist_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
std::string PathBaseAssignComp = PathNLFolder + "BaseAssignmentComp_" + std::to_string(nc);
std::string PathSubGraphFolder = PathNLFolder + "GraphIndex/";
std::string PathCenGraphInfo = PathNLFolder + "CenGraph_" + std::to_string(nc) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
std::string PathCenGraphEdge = PathNLFolder + "CenGraph_" + std::to_string(nc) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";
std::string PathCenGraphIndex = PathNLFolder + "CenGraph_" + std::to_string(nc) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".index";

// Data Path 
const std::string PathLearn =     PathFolder  + Dataset + "/" + Dataset +"_learn.fvecs";
const std::string PathBase =      PathFolder  + Dataset + "/" + Dataset +"_base.fvecs";
const std::string PathGt =        PathFolder  + Dataset + "/" + Dataset +"_groundtruth.ivecs";
const std::string PathQuery =     PathFolder  + Dataset + "/" + Dataset +"_query.fvecs";
const std::string PathTrainGt =   PathFolder  + Dataset + "/" + Dataset +"_Train_groundtruth.ivecs";
const std::string PathTrainDist = PathFolder  + Dataset + "/" + Dataset +"_Train_Dist.fvecs";

std::string PathCentroid     = PathFolder + Dataset + "/Centroids_" + NCString + ".fvecs";
std::string PathIniCentroid  = PathFolder + Dataset + "/Centroids_" + NCString + "_" + std::to_string(IniM)+"_Ini.fvecs";

std::string PathBaseIDInv    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Inv";
std::string PathBaseIDSeq    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Seq";

std::string PathCentroidNorm = PathFolder + Dataset + "/CentroidNorms_" + NCString;

std::string PathLearnCentroidNorm = PathFolder + Dataset + "/CentroidNormsLearn_" + NCString;
std::string PathGraphInfo    = PathFolder + Dataset + "/HNSW_" + NCString + "_Info";
std::string PathGraphEdges   = PathFolder + Dataset + "/HNSW_" + NCString + "_Edge";

std::string PathCentroidNeighborID = PathFolder + Dataset + "/CenNeighborID_" + NCString;

std::string PathCentroidNeighborDist = PathFolder + Dataset + "/CenNeighborDist_" + NCString;

std::string PathNeighborList = PathFolder + Dataset + "/NeighborList_" + NCString;


std::string PathPQ           = PathFolder + Dataset + "/PQ_" + "M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathOPQ          = PathFolder + Dataset + "/OPQ_" + "M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString;
std::string PathBaseNorm     = PathFolder + Dataset + "/BaseNorm_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathBaseCode     = PathFolder + Dataset + "/BaseCode_nc" + "_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + OPQState + NCString;
std::string PathOPQCentroids = PathFolder + Dataset + "/OPQCentroids_" + "_OPQ_M_" + std::to_string(M_PQ) + "_NBits_" + std::to_string(CodeBits) + "_" + NCString +".fvecs";


std::string PathCentroid_PS     = PathFolder + Dataset + "/Centroids_" + NCString_PS + ".fvecs";
std::string PathIniCentroid_PS  = PathFolder + Dataset + "/Centroids_"  + std::to_string(IniM)+"_PS.fvecs";
std::string PathTrainsetLabel_PS = PathFolder + Dataset + "/TrainsetLabel_"  + std::to_string(IniM)+ "_" + std::to_string(nt) + "_PS.ivecs";

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
