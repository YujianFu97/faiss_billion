#ifndef _NCSEARCH_H
#define _NCSEARCH_H

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "../../faiss/utils/distances.h"
#include "../../faiss/utils/utils.h"
#include "../../faiss/utils/random.h"
#include "../utils/utils.h"
#include "../Optkmeans/Optkmeans.h"
#include "../hnswlib/hnswalg.h"
#include "../../faiss/impl/ProductQuantizer.h"
#include "../../faiss/utils/Heap.h"
#include <math.h>
#define STOP 1.0e-8
#define TINY 1.0e-30

uint32_t HeuristicNCList(size_t nb, float alpha, float DistBound, float * TrainSet, float * ClusterSize, float * Centroids, size_t Dimension, size_t TrainSize, 
                        bool verbose, bool optimize, size_t BoundSize, size_t CheckBatch, size_t MaxM, size_t NLevel, size_t OptSize, float Lambda);

uint32_t HeuristicINI(std::ofstream & RecordFile, size_t nb, size_t nq, size_t RecallK, float SubsetProp, size_t PQ_M, size_t nbits, size_t PQ_TrainSize, float * TrainSet, float * QuerySet, float * ResultCentroids, std::string PathCentroid,
std::string PathTrainsetLabel, size_t Dimension, size_t TrainSize, float TargetRecall, float MaxCandidateSize, bool verbose, bool optimize, size_t ClusterBoundSize, size_t CheckBatch, size_t MaxM, size_t IniM, size_t NLevel, size_t GraphM, size_t GraphEf, size_t OptSize, float Lambda);


std::tuple<bool, size_t, float, float, float, float> BillionUpdateRecall(
    size_t nb, size_t nq, size_t Dimension, size_t nc, size_t RecallK, float TargetRecall, float MaxCandidateSize, size_t ngt, size_t Assignment_num_batch, size_t NumInMemoryBatches,
    float * QuerySet, uint32_t * QueryGtLabel, float * CenNorms, uint32_t * Base_ID_seq, DataType * InMemoryBatches,
    std::string Path_base, 
    std::ofstream & RecordFile, hnswlib::HierarchicalNSW * CentroidHNSW, faiss::ProductQuantizer * PQ, std::vector<std::vector<uint32_t>> & BaseIds);


void BillionUpdateCost(
    size_t ClusterNum, size_t NC, float CheckProp, size_t LowerBound, size_t Dimension,
    float * ClusterVectorCost,
    std::string Path_base,
    std::vector<std::vector<uint32_t>> & BaseIds, hnswlib::HierarchicalNSW * Graph
);

void BillionUpdateCentroids(
    size_t Dimension, size_t NCBatch, float AvgVectorCost, bool optimize, float Lambda, size_t OptSize, size_t & NC, size_t nb, size_t Assignment_num_batch, size_t NumInMemoryBatches, 
    float * ClusterCostBatch, uint32_t * ClusterIDBatch, float * Centroids, uint32_t * Base_ID_seq, DataType * InMemoryBatches,
    std::string Path_base,
    std::vector<std::vector<uint32_t>> & BaseIds
);



#endif