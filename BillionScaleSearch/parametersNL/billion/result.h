#include "./DEEP1B.h"


// For search
const size_t NumRecall = 1;
const size_t RecallK[NumRecall] = {1};

const size_t NumPara = 10;
const size_t MaxItem[NumPara] =           {50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000};
const size_t EfSearch[NumPara] =          {190, 200, 210, 220, 250, 300, 350, 400, 450, 470}; 
const std::string SearchMode = "NonParallel";

// For kmeans clustering
const size_t nc = 1000000;
const size_t nt_c = 20000000 ;

//For HNSW graph
const size_t M_kmeans = 32;
const size_t efConstruction_kmeans = 200;
const size_t efSearch_kmeans = 200;

const size_t M = 32;
const size_t EfConstruction = 100;
const size_t EfAssignment = 100;
const size_t EfNN = 100;
const size_t EfQueryEvalaution = 100;

// For NClist
const bool prunevectors = false;
const size_t SearchK = 10;    // The number of NN in search, use SearchK > TargetK
const size_t NLTargetK = 10;
const size_t MaxNeSTSize = 500;
const size_t MaxNeSTNeighbor = 50;
const size_t NeighborNum = 1; // The number of neighbor clusters to be computed
const size_t Assign_num_batch = 100; // Assignment batch num
const size_t Assign_batch_size = nb / Assign_num_batch;
const size_t Graph_num_batch = 20;
const size_t Graph_batch_size = nb / Graph_num_batch;

// File path
const std::string PathBase = PathFolder  + Dataset + "/" + Dataset +"_base.vecs";
const std::string PathTrain = PathFolder  + Dataset + "/" + Dataset +"_learn.vecs";
const std::string PathGt =        PathFolder  + Dataset + "/" + Dataset +"_groundtruth.ivecs";
const std::string PathQuery =     PathFolder  + Dataset + "/" + Dataset +"_query.vecs";

const std::string PathNLFolder = PathFolder + Dataset + "/" + "NLFiles/";
const std::string PathSubGraphFolder = PathNLFolder + "GraphIndex/";

const std::string PathCentroid = PathNLFolder + "Centroids_" + std::to_string(nc) + "-" + std::to_string(nt_c) + ".fvecs";

const std::string PathTrainsetNN = PathNLFolder +  + "TrainNN_" + std::to_string(nt) + "_" + std::to_string(SearchK) + ".NN";
const std::string PathCenGraphIndex = PathNLFolder + "CenGraph_" + std::to_string(nc) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".index";

const std::string PathBaseNeighborID = PathNLFolder + "BaseNeighborID_" + std::to_string(nc) + "_" + std::to_string(NeighborNum)  + ".ID";
const std::string PathBaseNeighborDist = PathNLFolder + "BaseNeighborDist_" + std::to_string(nc) + "_" + std::to_string(NeighborNum) + ".Dist";
const std::string PathTrainNeighborID = PathNLFolder + "TrainNeighborID_" +   "_" + std::to_string(nc) + "_" + std::to_string(NeighborNum) + ".ID";
const std::string PathTrainNeighborDist = PathNLFolder + "TrainNeighborDist_" +   "_" + std::to_string(nc) + "_" + std::to_string(NeighborNum) + ".Dist";

const std::string PathNeSTClustering =   PathNLFolder + "NeSTClustering_" + std::to_string(nc) + "_" + std::to_string(NLTargetK) + "_" + std::to_string(MaxNeSTSize);
const std::string PathNeST1DProjection = PathNLFolder + "NeST1DProjection_" + std::to_string(nc) + "_" + std::to_string(NLTargetK) + "_" + std::to_string(MaxNeSTSize);
const std::string PathNeSTNonList =   PathNLFolder + "NeSTNonList_" + std::to_string(nc) + "_" + std::to_string(NLTargetK) + "_" + std::to_string(MaxNeSTSize);



