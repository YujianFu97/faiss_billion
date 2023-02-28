#include<string>
#include "./Optkmeans.h"
#include <random>
#include <set>

// The function for optimizing Kmeans centroids with cluster size
#define EPS (1 / 1024.)

void OptimizeCluster(size_t nc, size_t TrainSize, size_t NeighborSize, float * ClusterLists, uint32_t * Labels, float * Distances, float Lambda, bool AddiFunc, bool ControlStart){

    /*
    for (size_t j = 0; j < nc; j++){
        printf("%f ", cluster_lists[j]);
    }
    printf("\n");
    */
    // Vector assignment: If the cost function is: sum[nb](d^2 + lambda * cs)
    // sum[nb](d^2) + sum[nc](lambda * cs^2)
    // the difference is: d2^2 - d1^2 + 2*lambda*(cs2 - cs1 + 1)
    // Start the reassignment from the largest / smallest cluster 
    // The larger lambda means larger weight on the search cost
    if (AddiFunc && ControlStart){
        // Initialize the indice
        std::vector<uint32_t> ClusterIdx(nc);
        std::iota(ClusterIdx.begin(), ClusterIdx.end(), 0);
        // Sort the cluster size and start from the largest cluster
        std::sort(ClusterIdx.begin(), ClusterIdx.end(), [&ClusterLists](uint32_t i1, uint32_t i2) {return ClusterLists[i1] > ClusterLists[i2];});

        // To avoid double check: When a vector is assigned to another cluster, it will be re-checked
        std::vector<bool> CheckFlag(TrainSize, false);

        for (size_t j = 0; j < nc; j++){
            uint32_t CurrentID = ClusterIdx[j];
            for (size_t k = 0; k < TrainSize; k++){
                if (Labels[k * NeighborSize] == CurrentID && !CheckFlag[k]){
                    CheckFlag[k] = true;
                    for (size_t m = 1; m < NeighborSize; m++){

                        // Only transfer vectors from large clusters to small clusters
                        if (ClusterLists[Labels[k * NeighborSize + m]] > ClusterLists[Labels[k * NeighborSize]])
                            continue;
                        float Difference = Distances[k * NeighborSize + m] - Distances[k * NeighborSize] + 
                                            2 * Lambda * (ClusterLists[Labels[k * NeighborSize + m]] - ClusterLists[Labels[k * NeighborSize]] + 1);
                        if (Difference < 0){
                            // std::cout << k << " " << labels[k * NeighborSize] << " " << cluster_lists[labels[k * NeighborSize]]  << " " <<
                            //       labels[k * NeighborSize + m] << " " << cluster_lists[labels[k * NeighborSize + m]] << std::endl; 
                            float DistTemp = Distances[k * NeighborSize + m]; 
                            Distances[k * NeighborSize + m] = Distances[k*NeighborSize]; 
                            Distances[k * NeighborSize] = DistTemp; 
                            uint32_t LabelTemp = Labels[k * NeighborSize + m]; 
                            Labels[k * NeighborSize + m] = Labels[k * NeighborSize]; 
                            Labels[k * NeighborSize] = LabelTemp; 
                            ClusterLists[Labels[k * NeighborSize]] ++; 
                            ClusterLists[Labels[k * NeighborSize + m]] --;
                            //break;
                        }
                    }
                }
                else{
                    continue;
                }
            }
        }
    }
    else if (AddiFunc){
        for (size_t j = 0; j < TrainSize; j++){
            for (size_t k = 1; k < NeighborSize; k++){
                float difference = Distances[j * NeighborSize + k] - Distances[j * NeighborSize] + 
                                    2 * Lambda * (ClusterLists[Labels[j * NeighborSize + k]] - ClusterLists[Labels[j * NeighborSize]] + 1);
                if (difference < 0){
                    //std::cout << labels[j * NeighborSize] << " " << cluster_lists[labels[j * NeighborSize]]  << " " << 
                        //       labels[j * NeighborSize + k] << " " << cluster_lists[labels[j * NeighborSize + k]] << std::endl; 
                    float dist_temp = Distances[j * NeighborSize + k];
                    Distances[j * NeighborSize + k] = Distances[j * NeighborSize];
                    Distances[j * NeighborSize] = dist_temp;
                    uint32_t LabelTemp = Labels[j * NeighborSize + k];
                    Labels[j * NeighborSize + k] = Labels[j * NeighborSize];
                    Labels[j * NeighborSize] = LabelTemp;
                    ClusterLists[Labels[j * NeighborSize]] ++;
                    ClusterLists[Labels[j * NeighborSize + k]] --;
                    //break;
                }
            }
        }
    }
    // If we change the target function
    // Vector assignment: If the cost function is: sum[nb](d^2 * cs)
    // sum[nc](sum(d^2)cs)
    // the difference is: d22^2 * (cs2 +1) + d2^2 - d1^2 - d11^2 * cs1
    else if (ControlStart){
        std::vector<float> ClusterDistances(nc, 0);

        for (size_t j = 0; j < TrainSize; j++){
            ClusterDistances[Labels[j*NeighborSize]] += Distances[j*NeighborSize];
        }

        // Initialize the indice
        std::vector<uint32_t> ClusterIdx(nc);
        std::iota(ClusterIdx.begin(), ClusterIdx.end(), 0);
        // Start from the largest cluster
        std::sort(ClusterIdx.begin(), ClusterIdx.end(), [&ClusterLists](uint32_t i1, uint32_t i2) {return ClusterLists[i1] < ClusterLists[i2];});
        for (size_t j = 0; j < nc; j++){
            uint32_t current_id = ClusterIdx[j];
            for (size_t k = 0; k < TrainSize; k++){
                if (Labels[k * NeighborSize] == current_id){
                    for (size_t m = 1; m < NeighborSize; m++){
                        uint32_t base_idx = Labels[k * NeighborSize];
                        uint32_t target_idx = Labels[k * NeighborSize + m];
                        if (ClusterLists[Labels[k * NeighborSize + m]] > ClusterLists[Labels[k * NeighborSize]])
                            continue;
                        float difference = Distances[k * NeighborSize + m] * (ClusterLists[target_idx] + 1) +
                                            ClusterDistances[target_idx] - ClusterDistances[base_idx] - Distances[k * NeighborSize] * ClusterLists[base_idx];
                        if (difference < 0){
                            // std::cout << k << " " << labels[k * NeighborSize] << " " << cluster_lists[labels[k * NeighborSize]]  << " " <<
                            //       labels[k * NeighborSize + m] << " " << cluster_lists[labels[k * NeighborSize + m]] << std::endl; 
                            float DistTemp = Distances[k * NeighborSize + m]; 
                            Distances[k * NeighborSize + m] = Distances[k*NeighborSize];
                            Distances[k * NeighborSize] = DistTemp;
                            uint32_t label_temp = Labels[k * NeighborSize + m];
                            Labels[k * NeighborSize + m] = Labels[k * NeighborSize];
                            Labels[k * NeighborSize] = label_temp;
                            ClusterLists[Labels[k * NeighborSize]] ++;
                            ClusterLists[Labels[k * NeighborSize + m]] --;
                            ClusterDistances[target_idx] += Distances[k * NeighborSize + m];
                            ClusterDistances[base_idx] -= Distances[k * NeighborSize];
                            //break;
                        }
                    }
                }
                else{
                    continue;
                }
            }
        }
    }
    else{
        std::vector<float> ClusterDistances(nc, 0);

        for (size_t j = 0; j < TrainSize; j++){
            ClusterDistances[Labels[j*NeighborSize]] += Distances[j*NeighborSize];
        }
        for (size_t j = 0; j < TrainSize; j++){
            for (size_t k = 1; k < NeighborSize; k++){
                uint32_t base_idx = Labels[j * NeighborSize];
                uint32_t target_idx = Labels[j * NeighborSize + k];
                if (ClusterLists[target_idx] > ClusterLists[base_idx])
                    continue;
                float difference = Distances[j * NeighborSize + k] * (ClusterLists[target_idx] + 1) +
                                    ClusterDistances[target_idx] - ClusterDistances[base_idx] - Distances[j * NeighborSize] * ClusterLists[base_idx];
                if (difference < 0){
                    //std::cout << j << " " << labels[j * NeighborSize] << " " << cluster_lists[labels[j * NeighborSize]]  << " " <<
                    //    labels[j * NeighborSize + k] << " " << cluster_lists[labels[j * NeighborSize + k]] << std::endl; 
                    float dist_temp = Distances[j * NeighborSize + k];
                    Distances[j * NeighborSize + k] = Distances[j * NeighborSize];
                    Distances[j * NeighborSize] = dist_temp;
                    uint32_t label_temp = Labels[j * NeighborSize + k];
                    Labels[j * NeighborSize + k] = Labels[j * NeighborSize];
                    Labels[j * NeighborSize] = label_temp;
                    ClusterLists[Labels[j * NeighborSize]] ++;
                    ClusterLists[Labels[j * NeighborSize + k]] --;
                    ClusterDistances[Labels[j * NeighborSize]] += Distances[j * NeighborSize + k];
                    ClusterDistances[Labels[j * NeighborSize + k]] -= Distances[j * NeighborSize];
                }
            }
        }
    }
}

// Kmeans training with neighbor info for optimization
float neighborkmeans(float * TrainSet, size_t Dimension, size_t TrainSize, size_t nc, 
            float * Centroids, bool verbose, bool Initialized, bool Optimize, 
            float lambda, size_t OptSize, bool UseGraph, bool  addi_func, 
            bool  control_start, size_t iterations, bool keeptrainlabels, 
            uint32_t * trainlabels, float * traindists){
    
    // This is the number of clusters to be considered in groundtruth
    size_t NeighborNum = 10;
    // The number of groundtruth to be considered
    size_t RecallK = 10;

    // Corner cases
    if (TrainSize < nc){
        printf("Number of training points (%ld) should be at least "
             "as large as number of clusters (%ld)", TrainSize, nc);
        exit(0);}
    if (TrainSize == nc) {
        if (verbose) { printf("Number of training points (%ld) same as number of "
                       "clusters, just copying\n", TrainSize);}
        memcpy(Centroids, TrainSet, Dimension * nc * sizeof(float));
        if (keeptrainlabels){
            for (size_t i = 0; i < nc; i++){
                trainlabels[i] = i;
                traindists[i] = EPS;
            }
        }
        return 0;}

    if (verbose) printf("Clustering %ld points in %ldD to %ld clusters, nt / nc = %ld, redo  %ld iterations\n", 
                        size_t(TrainSize), Dimension, nc, size_t(TrainSize / nc), iterations);

    //time_recorder Trecorder = time_recorder();

    // Firstly train the index with original kmeans and then compute the neighbor info
    
    // Prepare the neighbor info
    std::vector<uint32_t> NeighborID(TrainSize * NeighborNum);
    std::vector<float> NeighborDist(TrainSize * NeighborNum);
    std::vector<std::vector<uint32_t>> TrainIDs(nc);
    std::vector<std::vector<uint32_t>> TrainBeNNs(TrainSize); // The vectors that takes the target vectors as NN
    
    optkmeans(TrainSet, Dimension, TrainSize, nc, Centroids, verbose, Initialized, Optimize, lambda, OptSize, UseGraph, addi_func, control_start, iterations);
    GraphSearch(NeighborID.data(), NeighborDist.data(), TrainSet, Centroids, TrainSize, nc, NeighborNum, Dimension);

    for (size_t i = 0; i < TrainSize; i++){
        TrainIDs[NeighborID[i * NeighborNum]].emplace_back(i);
    }

    std::vector<size_t> ClusterSize(nc);
    for (size_t i = 0; i < nc; i++){
        ClusterSize[i] = TrainIDs[i].size();
    }

    std::vector<int64_t> VectorGt(TrainSize * RecallK);
    std::vector<float> VectorDist(TrainSize * RecallK);

#pragma omp parallel for
    for (size_t i = 0; i < TrainSize; i++){
        faiss::maxheap_heapify(RecallK, VectorDist.data() + i * RecallK, VectorGt.data() + i * RecallK);
        for (size_t j = 0; j < NeighborNum; j++){
            uint32_t ClusterID = NeighborID[i * NeighborNum + j];
            size_t ClusterSize = TrainIDs[ClusterID].size();
            for (size_t temp = 0; temp < ClusterSize; temp++){
                uint32_t VectorID = TrainIDs[ClusterID][temp];
                float Dist = faiss::fvec_L2sqr(TrainSet + i * Dimension, TrainSet + VectorID * Dimension, Dimension);
                if (Dist < VectorDist[i * RecallK]){
                    faiss::maxheap_pop(RecallK, VectorDist.data() + i * RecallK, VectorGt.data() + i * RecallK);
                    faiss::maxheap_push(RecallK, VectorDist.data() + i * RecallK, VectorGt.data() + i * RecallK, Dist, VectorID);
                }
            }
        }
    }

    for (size_t i = 0; i < TrainSize; i++){
        for (size_t j = 0; j < RecallK; j++){
            TrainBeNNs[VectorGt[i * RecallK + j]].emplace_back(i);
        }
    }

    // Check the neighbor cluster num of all vectors
    // Result: The value in vectorcostsource: the number of NNs in the target cluster
    std::vector<ClusterCostQueue> VectorCostSource(TrainSize);
    for (size_t i = 0; i < TrainSize; i++){
        for (size_t j = 0; j < RecallK; j++){
            uint32_t NNID = VectorGt[i * RecallK + j];
            uint32_t NNClusterID = NeighborID[NNID * NeighborNum];
            for (size_t temp = 0; temp < NeighborNum; temp++){
                if (VectorCostSource[i].ClusterIDs.count(NeighborID[i * NeighborNum + temp] == 0)){
                    VectorCostSource[i].VecNumInCluster.insert(std::make_pair(NeighborID[i * NeighborNum + temp], 0));
                }
                if (NeighborID[i * NeighborNum + temp] == NNClusterID){
                    VectorCostSource[i].VecNumInCluster[NNClusterID] ++;
                    break;
                }
            }
        }
    }
    // Update the cluster until the search cost is reduced
    // Check the NNs of each target vector, whether the NN should be placed into the cluster with target vector
    // Update the vector ID based on the neighbor search cost
    // This is the cost that is related to the target vector
    // The cluster search cost 
    for (size_t i = 0; i < TrainSize; i++){
        for (size_t j = 0; j < RecallK; j++){

            uint32_t NN = VectorGt[i * RecallK + j];
            if (NeighborID[NN * NeighborNum] == NeighborID[i * NeighborNum]){
                continue;
            }
            // Check the origin cost
            uint32_t OriginNNID = NeighborID[NN * NeighborNum];
            size_t NNRNNNum = TrainBeNNs[NN].size();
            size_t OriginalClusterCost = 0;
            for (size_t temp1 = 0; temp1 < NNRNNNum; temp1++){
                for(auto it = VectorCostSource[TrainBeNNs[NN][temp1]].ClusterIDs.begin(); it!=VectorCostSource[TrainBeNNs[NN][temp1]].ClusterIDs.end(); it++){
                    OriginalClusterCost += ClusterSize[*it];
                }
            }

            uint32_t OriginID = NeighborID[NN * NeighborNum];
            NeighborID[NN * NeighborNum] = NeighborID[i * NeighborNum];
            ClusterSize[i * NeighborNum]--; ClusterSize[NN * NeighborNum]++;
            // Check the shift cost
            std::vector<ClusterCostQueue> VectorShiftQueue(NNRNNNum);
            for (size_t temp1 = 0; temp1 < NNRNNNum; temp1++){
                uint32_t NNRNN = TrainBeNNs[NN][temp1];
                for (size_t temp2 = 0; temp2 < RecallK; temp2++){
                    uint32_t NNRNNNN = VectorGt[NNRNN * RecallK + temp2];
                    for (size_t temp3 = 0; temp3 < NeighborNum; temp3++){
                        if(VectorShiftQueue[temp1].ClusterIDs.count(NeighborID[NNRNN * NeighborNum + temp3]) == 0){
                            VectorShiftQueue[temp1].VecNumInCluster.insert(std::make_pair(NeighborID[NNRNN * NeighborNum + temp3], 0));
                        }

                        if (NeighborID[NNRNN * NeighborNum + temp3] == NeighborID[NNRNNNN * NeighborNum]){
                            VectorShiftQueue[temp1].VecNumInCluster[NeighborID[NNRNN * NeighborNum + temp3]] ++;
                            break;
                        }
                    }
                }
            }
            size_t ShiftVectorCost = 0;
            for (size_t temp1 = 0; temp1 < NNRNNNum; temp1++){
                for(auto it = VectorShiftQueue[temp1].ClusterIDs.begin(); it!=VectorShiftQueue[temp1].ClusterIDs.end(); it++){
                    ShiftVectorCost += ClusterSize[*it];
                }
            }

            if (OriginalClusterCost < ShiftVectorCost){
                NeighborID[NN * NeighborNum] = OriginID;
                ClusterSize[i * NeighborNum]++; ClusterSize[NN * NeighborNum]--;
            }
        }
    }

    // Update the cost
}


// Do the kmeans training with the cluster size optimization

float optkmeans(float * TrainSet, size_t Dimension, size_t TrainSize, size_t nc, 
            float * Centroids, bool verbose, bool initialized, bool optimize,
            float Lambda, size_t OptSize, bool UseGraph, bool  AddiFunc, 
            bool  ControlStart, size_t Iterations, bool keeptrainlabels, 
            uint32_t * trainlabels, float * traindists){

    // Corner cases
    if (TrainSize < nc){
        printf("Number of training points (%ld) should be at least "
             "as large as number of clusters (%ld)", TrainSize, nc);
        exit(0);}
    if (TrainSize == nc) {
        if (verbose) { printf("Number of training points (%ld) same as number of "
                       "clusters, just copying\n", TrainSize);}
        memcpy(Centroids, TrainSet, Dimension * nc * sizeof(float));
        if (keeptrainlabels){
            for (size_t i = 0; i < nc; i++){
                trainlabels[i] = i;
                traindists[i] = EPS;
            }
        }
        return 0;}

    if (verbose) printf("Clustering %ld points in %ldD to %ld clusters, nt / nc = %ld, redo  %ld iterations\n", 
                        size_t(TrainSize), Dimension, nc, size_t(TrainSize / nc), Iterations);

    time_recorder Trecorder = time_recorder();

    if (! initialized){
        randinit(TrainSet, Dimension, TrainSize, nc, Centroids);
        if (verbose) printf("Initialization with random in %.2f ms \n", Trecorder.getTimeConsumption()/1000);
    }

    // The neighbor cluster to be considered for each vector for reassignment
    // If no optimize, then neighborsize should be 1
    size_t NeighborSize = !optimize ? 1 : (nc < OptSize ? nc : OptSize) ;
    std::vector<uint32_t> Labels(TrainSize * NeighborSize, 0);
    std::vector<float> Distances(TrainSize * NeighborSize, 0);

    float sse = 0;
    Iterations = keeptrainlabels ? Iterations + 1 : Iterations;

    for (size_t i = 0; i < Iterations; i++){
        // Search the closest centroid
        //std::cout << "Assign to train vectors with neighbor size: " << NeighborSize << "\n";
        if (UseGraph){
            GraphSearch(Labels.data(), Distances.data(), TrainSet, Centroids, TrainSize, nc, NeighborSize, Dimension);
        }
        else{
            std::vector<int64_t> TempLabels(TrainSize * NeighborSize, 0);
            faiss::float_maxheap_array_t res = {size_t(TrainSize), size_t(NeighborSize), TempLabels.data(), Distances.data()};
            faiss::knn_L2sqr (TrainSet, Centroids, Dimension, TrainSize, nc, &res);
            for (size_t j = 0; j < TrainSize * NeighborSize; j++){Labels[j] = TempLabels[j];}
        }

        /*
        printf("The distances for searched vectors \n");
        for (size_t j = 0; j < 10; j++){
            for (size_t k = 0; k < NeighborSize; k++){
                printf("%.2f ", distances[j * NeighborSize + k]);
            }
            printf("\n");
        }
        */
        //std::cout << "Update the error\n";
        for(size_t j = 0; j < TrainSize; j++){
            if (keeptrainlabels){
                trainlabels[j] = Labels[j * NeighborSize];
                traindists[j] = Distances[j * NeighborSize];
            }
            sse += Distances[j * NeighborSize];
        }

        if (keeptrainlabels && i == Iterations - 1){break;}

        //std::cout << "Update the cluster size\n";
        std::vector<float> cluster_size(nc, 0);
        for (size_t j = 0; j < TrainSize; j++){
            uint32_t id = Labels[j * NeighborSize];
            cluster_size[id] += 1;
        }

        if(optimize){
            //std::cout << "Optimize the neighbor info\n";
            OptimizeCluster(nc, TrainSize, NeighborSize, cluster_size.data(), Labels.data(), Distances.data(), Lambda, AddiFunc, ControlStart);
        }

        // Update the centroids
        
        std::fill(Centroids, Centroids + nc*Dimension, 0);

        //std::cout << "Update the cluster centroids\n";
        // Get the sum of all vectors in one cluster
        for (size_t j = 0; j < TrainSize; j++){
            uint32_t id = Labels[j * NeighborSize];
            for (size_t k = 0; k < Dimension; k++){
                Centroids[id * Dimension + k] += TrainSet[j * Dimension + k];
            }
        }

        // Get the mean of all vectors in one cluster
        for (size_t j = 0; j < nc; j++){
            float cz = (float)cluster_size[j];
            if (cz != 0){
                for (size_t k = 0; k < Dimension; k++){
                    Centroids[j * Dimension + k] /= cz;
                }
            }
        }
        //std::cout << "Update the void centroids\n";
        // Handle the void clusters
        size_t n_void = 0;
        faiss::RandomGenerator rng (1234);
        size_t Sum = 0;
        for (size_t ci = 0; ci < nc; ci++){
            Sum += cluster_size[ci];
            if (cluster_size[ci] == 0){
                size_t cj;
                for (cj = 0; 1; cj = (cj+1) % nc){
                    float p = (cluster_size[cj] - 1.0) / (float) (TrainSize - nc);
                    float r = rng.rand_float();
                    if (r< p){break;}
                }
                memcpy (Centroids + ci *Dimension, Centroids + cj * Dimension, sizeof(float) * Dimension);
                /* Introduce small pertubation */
                for (size_t j = 0; j < Dimension; j++){
                    if (j%2 == 0){
                        Centroids[ci * Dimension + j] *= 1 + EPS;
                        Centroids[cj * Dimension + j] *= 1 - EPS;
                    }
                    else{
                        Centroids[ci * Dimension + j] *= 1 - EPS;
                        Centroids[cj * Dimension + j] *= 1 + EPS;
                    }
                }
                cluster_size[ci] = cluster_size[cj] / 2;
                cluster_size[cj] -= cluster_size[ci];
                n_void++;
            }
        }

        if (verbose){

        //Compute the STD of cluster size
        double sum = std::accumulate(cluster_size.begin(), cluster_size.end(), 0.0);
        if (sum != TrainSize){std::cout << "Cluster Size Sum: " << sum <<  " is different from the trainsize: " << TrainSize << " " << Sum << std::endl; exit(0);}
        double m =  sum / cluster_size.size();
        double accum = 0.0;
        double diffprop = 0.0;
        size_t MaxSize = 0;
        size_t MinSize = TrainSize;
        std::for_each (std::begin(cluster_size), std::end(cluster_size), [&](const double d) {
            accum += (d - m) * (d - m);
            diffprop += std::abs(d - m);
            if (d < MinSize){MinSize = d;}
            if(d > MaxSize){MaxSize = d;}
        });

        double stdev = sqrt(accum / cluster_size.size());

        // Report the training result
        
            printf("Iteration (%ld), time (%.2f) s, Dist (%.2f), cluster split (%ld), Cluster size STD: (%.2lf), Diff Prop: (%.1lf) / (%.0lf), Max:  %ld, Min: %ld \n", i, Trecorder.get_time_usage(), sse / TrainSize, n_void, stdev, diffprop/nc, m, MaxSize, MinSize);
        }

        /*
        if (verbose){
            printf("The cluster size: ");
            for(size_t j = 0; j < nc; j++){printf("%ld, ", cluster_size[j]);}
            std::cout << std::endl;
        }
        */

        /*
        if (best_sse - EPS < sse){if (verbose) printf("\nsse gets saturated, kmeans training stop"); break;}
        */
        sse = 0;
    }
    if (verbose) printf("\n");
    return 0;
}
#undef EPS




void randinit(float * trainset, size_t dimension, size_t train_size, size_t nc, float * centroids){
    RandomSubset<float>(trainset, centroids, dimension, train_size, nc);
}

void GraphSearch(uint32_t * ID, float * Dist, float * Query, float * BaseSet, size_t nq, size_t nb, size_t k, size_t Dimension, size_t M, size_t EfCons){
    time_recorder TRecorder = time_recorder();
    hnswlib::HierarchicalNSW * Graph = new hnswlib::HierarchicalNSW(Dimension, nb, M, 2 * M, EfCons);
    for (size_t i = 0; i < nb; i++){
        Graph->addPoint(BaseSet + i * Dimension);
    }
    TRecorder.print_time_usage("Graph for GraphSearch function Constructed");
#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        auto Result = Graph->searchKnn(Query + i * Dimension, k);
        for (size_t j = 0; j < k; j++){
            Dist[i * k + k - 1 - j] = Result.top().first;
            ID[i * k + k - 1 - j] = Result.top().second;
            Result.pop();
        }
    }
    delete(Graph);
    Graph = nullptr;
    TRecorder.print_time_usage("GraphSearch completed");
}

// Do hierarchical local training and then do global training for refinement

float hierarkmeans(float * trainset, size_t dimension, size_t trainsize, size_t nc, 
                float * centroids, size_t level, bool optimize, bool UseGraph, size_t OptSize, float Lambda, 
                size_t iterations, bool keeptrainlabels, uint32_t * trainlabels, 
                float * traindists){

    time_recorder Trecorder = time_recorder();

    printf("Hierarchical Kmeans training with %ld layers for %ld centroids\n", level, nc);

    // Initialization
    size_t nc_layer = std::floor(pow(nc, 1.0/level));

    size_t nc_completed = 1;

    std::vector<uint32_t> labels(trainsize);
    std::vector<float> distances(trainsize);
    std::vector<std::vector<float>> trainsets(0);
    std::vector<size_t> trainsizes(0);
    // We use this to locate the new centroid
    std::vector<size_t> accumulated_nc(0);

    size_t ShrinkSize = 100000;
    if (trainsize > ShrinkSize && level > 1){
        std::vector<float> FirstLayerSet(ShrinkSize * dimension);
        RandomSubset<float>(trainset, FirstLayerSet.data(), dimension, trainsize, ShrinkSize);
        // 1 layer Kmeans training
        optkmeans(FirstLayerSet.data(), dimension, ShrinkSize, nc_layer, centroids, true, false, optimize, Lambda, OptSize, UseGraph, false, iterations);
    }
    else{
    // 1 layer Kmeans training, do not keep the train labels as we will search the label later
        optkmeans(trainset, dimension, trainsize, nc_layer, centroids, true, false, optimize, Lambda, OptSize, UseGraph,
                  true, false, iterations);
    }
    nc_completed = nc_layer;

    printf("Completed first layer training with time consumption %.2f s \n", Trecorder.get_time_usage());

    for (size_t each_level = 1; each_level < level; each_level++){
        // Assign the trainset
        std::cout << "Assign the training vectors" << std::endl;
        GraphSearch(labels.data(), distances.data(), trainset, centroids, trainsize, nc_completed, 1, dimension);

        //faiss::float_maxheap_array_t res = 
        //{size_t(trainsize), size_t(1), labels.data(), distances.data()};
        //faiss::knn_L2sqr(trainset, centroids, dimension, trainsize, nc_completed, &res);

        // Re-orgnize the trainset
        trainsizes.resize(nc_completed, 0);
        for (size_t i = 0; i < trainsize; i++){
            trainsizes[labels[i]]++;
        }

        trainsets.resize(nc_completed);
        size_t MaxSize = 0; size_t MinSize = trainsize;
        for (size_t i = 0; i < nc_completed; i++){
            trainsets[i].resize(trainsizes[i] * dimension);
            if (trainsizes[i] < MinSize){MinSize = trainsizes[i];}
            if (trainsizes[i] > MaxSize){MaxSize = trainsizes[i];}
        }

        std::vector<size_t> temp_trainsizes(nc_completed, 0);
        for (size_t i = 0; i < trainsize; i++){
            uint32_t label = labels[i];
            memcpy(trainsets[label].data() + temp_trainsizes[label] * dimension, trainset + i * dimension, dimension * sizeof(float));
            temp_trainsizes[label]++;
        }
        printf("Completed trainset re-orgnization with time consumption %.2f s \n", Trecorder.get_time_usage());
        std::cout << "The max cluster size: " << MaxSize << " Min cluster size: " << MinSize << "\n";

        // Update the nc setting for next layer training
        // We assign the nc based on the cluster size -- as we want a balanced distribution on the cluster size
        // We add the gap for keeping expected nc trained

        assert(nc_completed > 0);
        size_t expected_nc_next = each_level == level - 1 ? nc : nc_completed * nc_layer;
        size_t nc_assigned = 0;
        accumulated_nc.resize(nc_completed, 0);
        for (size_t i = 0; i < nc_completed; i++){
            // Decide the nc based on the cluster size
            accumulated_nc[i] += std::floor(float(trainsizes[i] * expected_nc_next) / trainsize);
            nc_assigned += accumulated_nc[i];
        }

        // nc_gap: nc required to be further assigned
        size_t nc_gap = expected_nc_next - nc_assigned;
        assert(expected_nc_next >= nc_assigned);

        if (nc_gap > 0){
            std::cout << "Updating the nc assignment" << std::endl;
            // If we have larger gap than completed nc
            size_t nc_selected = nc_gap < nc_completed ? nc_gap : nc_completed;
            size_t nc_gap_per = std::floor(float(nc_gap) / nc_selected);
            size_t nc_gap_gap = nc_gap - nc_gap_per * nc_selected;

            std::vector<float> sizes_cover_gap(nc_selected, 0);
            std::vector<int64_t> labels_cover_gap(nc_selected, 0);
            faiss::minheap_heapify(nc_selected, sizes_cover_gap.data(), labels_cover_gap.data());
            // Select the largest clusters in nc_selected 
            for (size_t i = 0; i < nc_completed; i++){
                if (trainsizes[i] > sizes_cover_gap[0]){
                    faiss::minheap_pop(nc_selected, sizes_cover_gap.data(), labels_cover_gap.data());
                    faiss::minheap_push(nc_selected, sizes_cover_gap.data(), labels_cover_gap.data(), float(trainsizes[i]), i);
                }
            }
            for (size_t i = 0; i < nc_selected; i++){
                accumulated_nc[labels_cover_gap[i]] += i < nc_gap_gap ? nc_gap_per + 1: nc_gap_per;
            }
        }

        for (size_t i = 1; i < nc_completed; i++){
            accumulated_nc[i] += accumulated_nc[i-1];
        }

        assert(accumulated_nc[nc_completed -1] == expected_nc_next);
        printf("Update cluster nc for the last layer with time consumption %.2f s \n", Trecorder.get_time_usage());

        // Do local kmeans training in parallel
        int nt = omp_get_max_threads();
        // We do loop with the number of threads
        size_t StartIndice = 0;
        size_t EndIndice = StartIndice + (nt < nc_completed ? nt : nc_completed);
        bool FlagContinue = true;
        
        while(FlagContinue){
#pragma omp parallel for
            for (size_t i = StartIndice; i < EndIndice; i++){
                //std::cout << i << " " << temp_trainsizes[i] << " " << accumulated_nc[i] - accumulated_nc[i-1] <<  "\n";
                std::vector<uint32_t> TrainsetLabel;
                std::vector<float> TrainsetDist;
                if(keeptrainlabels){TrainsetLabel.resize(trainsizes[i], 0); TrainsetDist.resize(trainsizes[i], 0);}

                size_t cluster_nc = i == 0? accumulated_nc[0] : accumulated_nc[i] - accumulated_nc[i-1];
                //std::cout << "Number of cluster in the " << i << " th cluster " << accumulated_nc[i] << " " << cluster_nc << "\n";
                optkmeans(trainsets[i].data(), dimension, trainsizes[i], cluster_nc, 
                        centroids + (accumulated_nc[i]-cluster_nc) * dimension, false, false, optimize, Lambda, OptSize, false, true, false, 
                        iterations, keeptrainlabels, TrainsetLabel.data(), TrainsetDist.data());
                if (keeptrainlabels){
                    size_t scaneditems = 0;
                    for (size_t j = 0; j < trainsize; j++){
                        if (labels[j] == i){
                            trainlabels[j] = TrainsetLabel[scaneditems] + (accumulated_nc[i]-cluster_nc);
                            traindists[j] = TrainsetDist[scaneditems];
                            scaneditems ++;
                        }
                    }
                    assert(scaneditems == trainsizes[i]);
                }
            }
            std::cout << "Parallel Completed cluster split in the " << EndIndice << " constructed index " << " with " << nc_completed << " in total in " <<Trecorder.get_time_usage() << " s\n";
            if (EndIndice == nc_completed){FlagContinue = false;}
            StartIndice = EndIndice;
            EndIndice = EndIndice + nt <= nc_completed ? EndIndice + nt : nc_completed;
        }

        printf("\nCompleted level %ld training with time consumption %.2f s \n", each_level+1, Trecorder.get_time_usage());
        nc_completed = expected_nc_next;
    }

    // Run global kmeans for refinement
    //float sse = optkmeans(trainset, dimension, trainsize, nc, centroids, true, true, optimize, Lambda, OptSize, true, false, 1);
    std::cout << "Hierarchical Training Completed" << std::endl;
    return 0;
}


float sc_eval(uint32_t * assign_id, uint32_t * neighbor_id, size_t * cluster_size, size_t NeighborSize, size_t neighbor_test_size, size_t nb){
    float actual_sc = 0;
    float neighbor_sc = 0;
    float conflict = 0;
    float setsize = 0;
    for (size_t i = 0; i < nb; i++){
        std::set<uint32_t> visited_clusters;
        visited_clusters.insert(assign_id[i]);
        //std::cout << assign_id[i] << " ";
        actual_sc += cluster_size[assign_id[i]];
        
        //if(neighbor_id[i * NeighborSize] != i){
        //    std::cout << "Neighbor Error " << i << " " << neighbor_id[i * NeighborSize];
        //    exit(0);
        //}
        
        for (size_t j = 1; j < neighbor_test_size; j++){
            if (visited_clusters.find(assign_id[neighbor_id[i * NeighborSize + j]]) != visited_clusters.end()){
                continue;
            }
            else{
                //std::cout << assign_id[neighbor_id[i * NeighborSize + j]] << " ";
                conflict ++;
                actual_sc += cluster_size[assign_id[neighbor_id[i * NeighborSize + j]]];
                neighbor_sc += cluster_size[assign_id[neighbor_id[i * NeighborSize + j]]];
                visited_clusters.insert(assign_id[neighbor_id[i * NeighborSize + j]]);
            }
        }
        setsize += visited_clusters.size();
        //std::cout << std::endl;
    }
    std::cout << "The overall search cost for " << neighbor_test_size-1 << " neighbors " << " is " << actual_sc << " with " << 
    neighbor_sc << " neighbor search cost and " <<  conflict / (nb * (neighbor_test_size - 1)) << " conflicts and " << setsize / nb << " setsize, neighbor cost percent: " << 
    neighbor_sc / actual_sc <<  std::endl;
    return actual_sc;
}


void kmeansplusplus(float * trainset, size_t dimension, size_t trainsize, size_t nc, float * centroids){
    printf("Initialize with Kmeans++ \n");
    time_recorder Trecorder = time_recorder();
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, trainsize);
    uint32_t id = distribution(gen);
    memcpy(centroids, trainset + id * dimension, dimension * sizeof(float));
    Trecorder.print_time_usage("Initialization");


    std::vector<float> minDistances(trainsize, 0);
    std::vector<int64_t> minLabels(trainsize, 0);
    float sum_distance = 0;

    std::uniform_real_distribution<float> real_distribution(0.0, 1.0);
    for (size_t i = 1; i < nc; i++){
        
        faiss::float_maxheap_array_t res = {size_t(trainsize), size_t(1), minLabels.data(), minDistances.data()};
        faiss::knn_L2sqr(trainset, centroids, dimension, trainsize, i, &res);
        sum_distance = std::accumulate(minDistances.begin(), minDistances.end(), 0);
    
        float probability = real_distribution(gen);
        sum_distance = sum_distance * probability;

        for (size_t j = 0; j < trainsize; j++){
            sum_distance -= minDistances[j];
            if (sum_distance > 0){continue;}
            memcpy(centroids + i * dimension, trainset + j * dimension, dimension);
            break;
        }
        Trecorder.print_time_usage("Iteration "+std::to_string(i) + " / " + std::to_string(nc));
    }
}