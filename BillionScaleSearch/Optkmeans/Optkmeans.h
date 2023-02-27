#ifndef _OPTKMEANS_H
#define _OPTKMEANS_H

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "../../faiss/utils/distances.h"
#include "../../faiss/utils/utils.h"
#include "../../faiss/utils/random.h"
#include"../utils/utils.h"
#include  "../../faiss/Clustering.h"
#include "../hnswlib/hnswalg.h"
#include <omp.h>

typedef float DataType;
//typedef uint8_t DataType;

struct ClusterCostQueue
{
    std::unordered_set<uint32_t> ClusterIDs;
    std::map<uint32_t, int> VecNumInCluster;
};


/**
 * Kmeans++ algorithm for intialization in Kmeans
 * 
**/
void kmeansplusplus(float * trainset, size_t dimension, size_t trainsize, size_t nc, float * centroids);

/**
 * The function for optimized Kmeans with the cluster size in consideration
 * Parameter:                             Meaning:
 * trainset                               The vectors for training
 * dimension
 * trainsize                              Number of training vectors
 * nc                                     NUmber of clusters
 * centroids                              Output centroids
 * iter                                   Training iterations
**/

void OptimizeCluster(size_t nc, size_t TrainSize, size_t NeighborSize, float * ClusterSize, uint32_t * Labels, float * Distances, float Lambda = 50, bool AddiFunc = true, bool ControlStart = false);

float optkmeans(float * Trainset, size_t Dimension, size_t Trainsize, size_t nc, 
            float * Centroids, bool Verbose, bool Initialized, bool Optimize, 
            float lambda = 50, size_t OptSize = 10, bool UseGraph = false, bool  addi_func = true, 
            bool  control_start = false, size_t iterations = 30, bool keeptrainlabels= false, 
            uint32_t * trainlabels = nullptr, float * traindists = nullptr );


void randinit(float * trainset, size_t dimension, size_t train_size, size_t nc, float * centroids);

float hierarkmeans(float * trainset, size_t dimension, size_t trainsize, size_t nc, 
                float * centroids, size_t level, bool optimize = true, bool UseGraph = false, size_t OptSize = 10, float Lambda = 50, 
                size_t iterations = 30, bool keeptrainlabels =false, uint32_t * trainlabels = nullptr, 
                float * traindists = nullptr);


void GraphSearch(uint32_t * ID, float * Dist, float * Query, float * BaseSet, size_t nq, size_t nb, size_t k, size_t Dimension, size_t M = 32, size_t EfCons = 500);

//float sc_eval(idx_t * assign_id, idx_t * neighbor_id, size_t * cluster_size, size_t neighbor_size, size_t neighbor_test_size, size_t nb, size_t nc);

float neighborkmeans(float * Trainset, size_t Dimension, size_t Trainsize, size_t nc, 
            float * Centroids, bool Verbose, bool Initialized, bool Optimize, 
            float lambda = 50, size_t OptSize = 10, bool UseGraph = false, bool  addi_func = true, 
            bool  control_start = false, size_t iterations = 30, bool keeptrainlabels= false, 
            uint32_t * trainlabels = nullptr, float * traindists = nullptr);


#endif