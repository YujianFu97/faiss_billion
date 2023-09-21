#pragma once

#include <string>
#include <unistd.h>
#include <set>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include "parametersNL/billion/result.h"
#include "utils/utils.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib_old/hnswalg.h"
//#include "FastPFor/headers/codecfactory.h"
//#include "complib/include/codecfactory.h"

constexpr float UNDEFINED_TARGET_CLUSTER_DIST = -1.0f;
constexpr size_t EXTRA_COMP_SIZE = 1024;
size_t NumComp = 0;
float EPS = 1e-5;

void loadNeighborAssignments(const std::string& path, size_t dataSize, size_t neighborNum, std::vector<uint32_t>& assignmentVec) {
    assert(exists(path));

    std::vector<uint32_t> vecNeighborID(dataSize * neighborNum);
    std::ifstream inputStream(path, std::ios::binary);
    if (!inputStream) {std::cerr << "Error opening path file: " << path << "\n";exit(1);}
    inputStream.read(reinterpret_cast<char*>(vecNeighborID.data()), dataSize * neighborNum * sizeof(uint32_t));
    inputStream.close();

    for (size_t i = 0; i < dataSize; i++) {
        assignmentVec[i] = vecNeighborID[i * neighborNum];
    }
}

void faisskmeansClustering(const float* training_data, float* centroids) {
    omp_set_num_threads(128);
    faiss::IndexFlatL2 index(Dimension); // Index with L2 distance
    faiss::ClusteringParameters cp; // Default clustering parameters
    cp.niter = 10;
    cp.verbose = true;
    faiss::Clustering clustering(Dimension, nc, cp);
    clustering.train(nt_c, training_data, index);
    memcpy(centroids, clustering.centroids.data(), Dimension * nc * sizeof(float));
}

// Function to update the Boundary Conflict Distance Map
void updateBoundaryConflictMapDist(
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>> &BoundaryConflictDistMap,
    const std::pair<uint32_t, uint32_t>& pair,
    float NNClusterDist,
    float QueryClusterDist,
    float DistNLBoundary
) {
    auto &ConflictDist = BoundaryConflictDistMap[pair];
    ConflictDist.first++;
    std::get<0>(ConflictDist.second) += DistNLBoundary;
    std::get<1>(ConflictDist.second) += QueryClusterDist;
    std::get<2>(ConflictDist.second) += NNClusterDist;
    std::get<3>(ConflictDist.second) += (NNClusterDist / QueryClusterDist);
}


float computeQueryClusterDist(
    const std::vector<float> &BaseNeighborDistBatch,
    const std::vector<uint32_t> &BaseNeighborIDBatch,
    const std::vector<float> &BaseBatch,
    const std::vector<float> &Centroids,
    uint32_t j,
    uint32_t NNClusterID,
    uint32_t QueryClusterID
){
    float QueryClusterDist = UNDEFINED_TARGET_CLUSTER_DIST;

    for (size_t temp = 1; temp < NeighborNum; temp++) {
        if (BaseNeighborIDBatch[j * NeighborNum + temp] == QueryClusterID) {
            QueryClusterDist = BaseNeighborDistBatch[j * NeighborNum + temp];
            break;
        }
    }
    if (QueryClusterDist < 0) {
        QueryClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);
    }

    return QueryClusterDist;
}

// The dist to centroids are squared distance, the boundary distance is sqrt distance.
std::pair<float, float> computeDistNLBoundary(
    const std::vector<float> &BaseNeighborDistBatch,
    const std::vector<uint32_t> &BaseNeighborIDBatch,
    const std::vector<float> &BaseBatch,
    const std::vector<float> &Centroids,
    uint32_t j,
    uint32_t NNClusterID,
    uint32_t QueryClusterID
) {
    float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
    float QueryClusterDist = UNDEFINED_TARGET_CLUSTER_DIST;

    for (size_t temp = 1; temp < NeighborNum; temp++) {
        if (BaseNeighborIDBatch[j * NeighborNum + temp] == QueryClusterID) {
            QueryClusterDist = BaseNeighborDistBatch[j * NeighborNum + temp];
            break;
        }
    }

    if (QueryClusterDist < 0) {
        QueryClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);
    }

    float CentroidDist = faiss::fvec_L2sqr(Centroids.data() + NNClusterID * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);
    float CosVecNNTarget = (NNClusterDist + CentroidDist - QueryClusterDist) / (2 * sqrt(NNClusterDist * CentroidDist));
    float DistNLBoundary = sqrt(CentroidDist) / 2 - sqrt(NNClusterDist) * CosVecNNTarget;

    return std::make_pair(QueryClusterDist, DistNLBoundary);
}

void pruneBaseSet(std::vector<float>& BaseBatch, std::vector<uint32_t> & BaseAssignment,
std::vector<float> Centroids, std::vector<std::unordered_set<uint32_t>> & BCCluster,
std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> BCClusterPrune)
{
    // Compute the distance to the centroid of cluster pairs
    // Firstly compute the centroids with the conflict vectors
    std::vector<uint32_t> BaseNeighborIDBatch(Assign_batch_size * NeighborNum);
    std::vector<float> BaseNeighborDistBatch(Assign_batch_size * NeighborNum);
    std::ifstream BaseNeighborIDInputStream(PathBaseNeighborID, std::ios::binary);
    if (!BaseNeighborIDInputStream) {std::cerr << "Error opening PathBaseNeighborID file: " << PathBaseNeighborID << "\n";exit(1);}
    std::ifstream BaseNeighborDistInputStream(PathBaseNeighborDist, std::ios::binary);
    if (!BaseNeighborDistInputStream) {std::cerr << "Error opening PathBaseNeighborDist file: " << PathBaseNeighborDist << "\n";exit(1);}

    // Compute the centroids based on the conflict
    std::vector<std::unordered_map<uint32_t, std::vector<float>>> BCCentroids(nc);
    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    for (uint32_t i = 0; i < Assign_num_batch; i++){
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, nc, true, true);
        for (uint32_t j = 0; j < Assign_batch_size; j++){
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[NNVecID];
            for (auto ClusterIt = BCClusterPrune[NNClusterID].begin(); ClusterIt !=  BCClusterPrune[NNClusterID].end(); ClusterIt++){
                uint32_t QueryClusterID = ClusterIt->first;
                if (ClusterIt->second.find(NNVecID) != ClusterIt->second.end()){
                    // Add the vectors to compute the centroids
                    if (BCCentroids[NNClusterID].find(QueryClusterID) != BCCentroids[NNClusterID].end()){
                        faiss::fvec_add(Dimension, BaseBatch.data() + j * Dimension, BCCentroids[NNClusterID][QueryClusterID].data(), BCCentroids[NNClusterID][QueryClusterID].data());
                    }
                    else{
                        BCCentroids[NNClusterID][QueryClusterID].resize(Dimension);
                        memcpy(BCCentroids[NNClusterID][QueryClusterID].data(), BaseBatch.data() + j * Dimension, Dimension * sizeof(float));
                    }
                }
            }
        }
    }

    // Compute the average centroid for each conflict pair
    for (uint32_t i = 0; i < nc; i++){
        for (auto ClusterIt = BCClusterPrune[i].begin(); ClusterIt != BCClusterPrune[i].end(); ClusterIt++){
            uint32_t QueryClusterID = ClusterIt->first;
            for (uint32_t j = 0; j < Dimension; j++){
                BCCentroids[i][QueryClusterID][j] /= ClusterIt->second.size();
            }
        }
    }
    // Compute the distance bound based on the distance to centroids, boundary and centroid
    std::map<std::pair<uint32_t, uint32_t>, std::vector<float>> BCDist;
    BaseInputStream.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < Assign_num_batch; i++){
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, false);
        BaseNeighborIDInputStream.read(reinterpret_cast<char*>(BaseNeighborIDBatch.data()), Assign_batch_size * NeighborNum * sizeof(uint32_t));
        BaseNeighborDistInputStream.read(reinterpret_cast<char*>(BaseNeighborDistBatch.data()), Assign_batch_size * NeighborNum * sizeof(float));

        #pragma omp parallel for
        for (uint32_t j = 0; j < Assign_batch_size; j++){
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            auto ClusterIt = BCClusterPrune[NNClusterID];
            for (auto QueryClusterIt = ClusterIt.begin(); QueryClusterIt != ClusterIt.end(); QueryClusterIt++){
                if (QueryClusterIt->second.find(NNVecID) != QueryClusterIt->second.end()){
                    auto QueryClusterID = QueryClusterIt->first;
                    auto ClusterIDPair = std::make_pair(NNClusterID, QueryClusterID);

                    auto Distances = computeDistNLBoundary(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, QueryClusterID);
                    float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                    float QueryClusterDist = Distances.first;
                    float DistNLBoundary = Distances.second;
                    float BCCentroidDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, BCCentroids[NNClusterID][QueryClusterID].data(), Dimension);

                    #pragma omp critical
                    {
                        auto BCDistCluster = BCDist.find(ClusterIDPair);

                        if (BCDistCluster == BCDist.end()){
                            BCDist[ClusterIDPair].resize(4, 0);
                            BCDistCluster = BCDist.find(ClusterIDPair);
                        }
                        BCDistCluster->second[0] += NNClusterDist;
                        BCDistCluster->second[1] += QueryClusterDist;
                        BCDistCluster->second[2] += DistNLBoundary;
                        BCDistCluster->second[3] += BCCentroidDist;
                    }
                }
            }
        }
    }
    // Compute the average distance
    #pragma omp parallel for
    for (uint32_t i = 0; i < nc; i++){
        for (auto QueryClusterIt = BCClusterPrune[i].begin(); QueryClusterIt != BCClusterPrune[i].end(); QueryClusterIt++){
            auto ClusterIDPair = std::make_pair(i, QueryClusterIt->first);
            auto BCDistCluster = BCDist.find(ClusterIDPair);
            assert(BCDistCluster != BCDist.end());
            BCDistCluster->second[0] /= QueryClusterIt->second.size();
            BCDistCluster->second[1] /= QueryClusterIt->second.size();
            BCDistCluster->second[2] /= QueryClusterIt->second.size();
            BCDistCluster->second[3] /= QueryClusterIt->second.size();
        }
    }
    // Prune the vectors not in list, check whether to add them or not
    BaseInputStream.seekg(0, std::ios::beg);
    BaseNeighborIDInputStream.seekg(0, std::ios::beg);
    BaseNeighborDistInputStream.seekg(0, std::ios::beg);
    for (size_t i = 0; i < Assign_num_batch; i++){
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, true);
        BaseNeighborIDInputStream.read((char *)BaseNeighborIDBatch.data(), Assign_batch_size * NeighborNum * sizeof(uint32_t));
        BaseNeighborDistInputStream.read((char *) BaseNeighborDistBatch.data(), Assign_batch_size * NeighborNum * sizeof(float));

        #pragma omp parallel for
        for (size_t j = 0; j < Assign_batch_size; j++){
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseNeighborIDBatch[j * NeighborNum];

            // This vector already exists as conflict vector in list
            if (std::find(BCCluster[NNClusterID].begin(), BCCluster[NNClusterID].end(), NNVecID) != BCCluster[NNClusterID].end()){
                continue;
            }

            // Insert the vector to BCCluster list
            for (auto ClusterIt = BCClusterPrune[NNClusterID].begin(); ClusterIt != BCClusterPrune[NNClusterID].end(); ClusterIt++){
                uint32_t QueryClusterID = ClusterIt->first;
                auto Distances = computeDistNLBoundary(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, QueryClusterID);
                float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                float QueryClusterDist = Distances.first;
                float DistNLBoundary = Distances.second;
                float BCCentroidDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, BCCentroids[NNClusterID][QueryClusterID].data(), Dimension);

                // Check the condition for pruning
                auto BCDistCluster = BCDist.find(std::make_pair(NNClusterID, QueryClusterID));
                if (NNClusterDist > BCDistCluster->second[0] && QueryClusterDist < BCDistCluster->second[1]
                    && DistNLBoundary < BCDistCluster->second[2] && BCCentroidDist < BCDistCluster->second[3]){
                    BCCluster[NNClusterID].insert(NNVecID);
                    break;
                }
            }
        }
    }
}


void simplekmeans(std::vector<float>& TrainSet, size_t Dimension, size_t TrainSize, size_t nc, std::vector<float>& Centroids, size_t Iteration=10) {
    omp_set_num_threads(128);
    std::cout << "Clustering " << TrainSize << " points in " << Dimension << "D  to " <<  nc << " clusters, redo " << Iteration << " iterations\n";
    performance_recorder recorder("simplekmeans with graph search");
    const float EPS = 1e-5;
    if (TrainSize < nc) {std::cout << "Error in nc and trainsize: " << nc << " " << TrainSize << "\n"; exit(0);}
    if (TrainSize == nc) {memcpy(Centroids.data(), TrainSet.data(), TrainSize * Dimension * sizeof(float));return;}

    long RandomSeed = 1234;
    std::vector<int> RandomId(TrainSize);
    faiss::rand_perm(RandomId.data(), TrainSize, RandomSeed+1);
    for (size_t i = 0; i < nc; i++){
        memcpy(Centroids.data() + i * Dimension, TrainSet.data() + RandomId[i] * Dimension, sizeof(float) * Dimension);
    }

    std::vector<int64_t> Labels(TrainSize, 0);
    std::vector<float> Distances(TrainSize, 0);
    hnswlib::L2Space l2space(Dimension);
    for (size_t i = 0; i < Iteration; i++) {
        hnswlib::HierarchicalNSW<float> * Graph = new hnswlib::HierarchicalNSW<float>(&l2space, nc, M_kmeans, efConstruction_kmeans);
        #pragma omp parallel for
        for (size_t j = 0; j < nc; j++){
            Graph->addPoint(Centroids.data() + j * Dimension, j);
        }
        Graph->ef_ = efSearch_kmeans;
        recorder.print_performance("Graph constructed, start assignment in parallel with ef: " + std::to_string(Graph->ef_) + "\n");

        #pragma omp parallel for
        for (size_t j = 0; j < TrainSize; j++){
            size_t NumComp_ = 0;
            auto Result = Graph->searchKnn(TrainSet.data() + j * Dimension, 1, NumComp_);
            Distances[j] = Result.top().first;
            Labels[j] = Result.top().second;
        }
        delete(Graph);
        Graph = nullptr;
        recorder.print_performance("Graph searched in parallel \n");

        float SumError = std::accumulate(Distances.begin(), Distances.end(), 0.0f);
        //faiss::float_maxheap_array_t res = {TrainSize, 1, Labels.data(), Distances.data()};
        //faiss::knn_L2sqr(TrainSet.data(), Centroids.data(), Dimension, TrainSize, nc, &res);
        std::vector<float> cluster_size(nc, 0);
        for (size_t j = 0; j < TrainSize; j++) {cluster_size[Labels[j]] += 1;}
        std::fill(Centroids.data(), Centroids.data() + nc * Dimension, 0);

        for (size_t j = 0; j < TrainSize; j++) {
            faiss::fvec_add(Dimension, Centroids.data() + Labels[j] * Dimension, TrainSet.data() + j * Dimension, Centroids.data() +Labels[j] * Dimension);
        }

        for (size_t j = 0; j < nc; j++) {
            float cz = (float)cluster_size[j];
            if (cz != 0) {
                for (size_t k = 0; k < Dimension; k++) {
                    Centroids[j * Dimension + k] /= cz;
                }
            }
        }
        // Handle the void clusters
        faiss::RandomGenerator rng(1234);
        size_t NumEmpty = 0;
        for (size_t ci = 0; ci < nc; ci++) {
            if (cluster_size[ci] == 0) {
            NumEmpty ++;
            size_t cj;
            float r = rng.rand_float();
            float p = 0;
            for (cj = 0; ; cj = (cj + 1) % nc) {
                p += (cluster_size[cj] - 1.0) / (float) (TrainSize - nc);
                if (r < p) break;
            }
            memcpy(Centroids.data() + ci * Dimension, Centroids.data() + cj * Dimension, sizeof(float) * Dimension);
            for (size_t j = 0; j < Dimension; j++) {
                Centroids[ci * Dimension + j] *= 1 + EPS;
                Centroids[cj * Dimension + j] *= 1 - EPS;
            }
            cluster_size[ci] = cluster_size[cj] / 2;
            cluster_size[cj] -= cluster_size[ci];
            }
        }
        recorder.print_performance("Complete " + std::to_string(i + 1) + " th iteration with sum error: " + std::to_string(size_t(SumError)) + " and num of empty: " + std::to_string(NumEmpty));
    }
}


void inline ErrorComputation(const std::vector<float> & VecCentralClusterDist, const std::vector<float> & NeiClusterDist,
const std::vector<float> & VecNeighborClusterDist, const std::vector<float> & AlphaValue,
 uint32_t VecIdx, uint32_t AlphaIdx, uint32_t LineIdx, std::vector<float> & ErrorResult, const size_t NumNC = MaxNeSTSize
 ){
    ErrorResult.resize(3);
    float a = VecCentralClusterDist[VecIdx];
    float b = NeiClusterDist[LineIdx];
    float c = VecNeighborClusterDist[VecIdx * NumNC + LineIdx];

    float Delta = a + b - c;
    float ProjError = a - (Delta * Delta) / (4 * b);
    float LineAlpha = (Delta) / (2 * sqrt(b));
    float LineError = (LineAlpha - AlphaValue[AlphaIdx]) * (LineAlpha - AlphaValue[AlphaIdx]);

    ErrorResult[0] = ProjError;
    ErrorResult[1] = LineAlpha;
    ErrorResult[2] = ProjError + LineError;
}


float NeighborKmeans(hnswlib::HierarchicalNSW<float>* CenGraph, const std::vector<float> & TrainSet, std::vector<uint32_t> & AlphaLineID, 
std::vector<float> & AlphaLineValue, std::vector<float> & AlphaLineNorm,
std::vector<std::vector<uint32_t>> & VecInvAssignment,
uint32_t ClusterID, const uint32_t NumAlpha, const uint32_t NumNC, const size_t Iterations = 10){
    //performance_recorder recorder("NeighborKmeans");
    float SumAlphaAssignError = 0;

    auto Result = CenGraph->searchKnn(CenGraph->getDataByLabel<float>(ClusterID).data(), NumNC + 1, NumComp);

    std::vector<uint32_t> NeiClusterID(NumNC); std::vector<float> NeiClusterDist(NumNC);
    for (size_t i = 0; i < NumNC; i++){
        assert(Result.top().second != ClusterID);
        NeiClusterID[i] = Result.top().second;
        NeiClusterDist[i] = Result.top().first;
        Result.pop();
    }
    size_t TrainSize = TrainSet.size() / Dimension;
    std::vector<float> VecNeighborClusterDist(NumNC * TrainSize); std::vector<float> VecCentralClusterDist(TrainSize);
    for (size_t i = 0; i < TrainSize; i++){
        VecCentralClusterDist[i] = CenGraph->fstdistfunc_(CenGraph->getDataByLabel<float>(ClusterID).data(), TrainSet.data() + i * Dimension, CenGraph->dist_func_param_);
        for (size_t j = 0; j < NumNC; j++){
            VecNeighborClusterDist[i * NumNC + j] = CenGraph->fstdistfunc_(TrainSet.data() + i * Dimension, CenGraph->getDataByLabel<float>(NeiClusterID[j]).data(), CenGraph->dist_func_param_);
        }
    }

    VecInvAssignment.resize(NumAlpha);
    AlphaLineID.resize(NumAlpha, 0);
    AlphaLineValue.resize(NumAlpha, 0);
    std::vector<float> AlphaAssignError(NumAlpha, 0);
    std::vector<uint32_t> VecAssignment(TrainSize);

    // Initialization with randomly selected vectors, project it to centroid line
    long RandomSeed = 1234;
    std::vector<int> RandomId(TrainSize);
    faiss::rand_perm(RandomId.data(), TrainSize, RandomSeed+1);
    std::vector<float> ErrorResult;
    for (size_t AlphaIdx = 0; AlphaIdx < NumAlpha; AlphaIdx++){
        uint32_t VecIdx = RandomId[AlphaIdx];
        float MinProjError = std::numeric_limits<float>::max();
        float ResultAlpha = 0;
        float ResultLineIdx = 0;

        for (size_t LineIdx = 0; LineIdx < NumNC; LineIdx++){
            ErrorComputation(VecCentralClusterDist, NeiClusterDist, VecNeighborClusterDist, AlphaLineValue, VecIdx, AlphaIdx, LineIdx, ErrorResult, NumNC);
            if (ErrorResult[0] < MinProjError){
                MinProjError = ErrorResult[0];
                ResultLineIdx = LineIdx;
                ResultAlpha = ErrorResult[1];
            }
        }
        AlphaLineID[AlphaIdx] = ResultLineIdx;
        AlphaLineValue[AlphaIdx] = ResultAlpha;
    }

    bool ContinueFlag = true;
    float PreviousSumAssignError = std::numeric_limits<float>::max();

    for (size_t Iter = 0; Iter < Iterations && ContinueFlag; Iter++){
        // Assign the vectors to the closest alpha
        for (size_t i = 0; i < TrainSize; i++){
            float MinError = std::numeric_limits<float>::max();
            uint32_t ResultIdx = NumAlpha;

            // Iterate over all alphas, find the closest alpha
            for (size_t AlphaIdx = 0; AlphaIdx < NumAlpha; AlphaIdx++){

                uint32_t LineIdx = AlphaLineID[AlphaIdx];
                uint32_t VecIdx = i;
                ErrorComputation(VecCentralClusterDist, NeiClusterDist, VecNeighborClusterDist, AlphaLineValue, VecIdx, AlphaIdx, LineIdx, ErrorResult, NumNC);

                if (ErrorResult[2] < MinError){
                    ResultIdx = AlphaIdx;
                    MinError = ErrorResult[2];
                }
            }
            assert(ResultIdx < NumAlpha);
            VecAssignment[i] = ResultIdx;
        }
        for (uint32_t i = 0; i < NumAlpha; i++){VecInvAssignment[i].clear();}
        for (uint32_t i = 0; i < TrainSize; i++){VecInvAssignment[VecAssignment[i]].emplace_back(i);}

        // Update the alpha for each cluster, check its minimum error on all lines
        std::vector<uint32_t> EmptyAlpha;
        std::fill(AlphaAssignError.begin(), AlphaAssignError.end(), 0);
        // Recompute the alpha
        for (size_t AlphaIdx = 0;  AlphaIdx < NumAlpha; AlphaIdx++){
            // compute the result on i-th line
            if (VecInvAssignment[AlphaIdx].size() > 0){
                uint32_t ResultLineIdx = NumNC;
                float ResultLineAlpha = 0;
                float MinAlphaAssignError = std::numeric_limits<float>::max();

                for (size_t LineIdx = 0; LineIdx < NumNC; LineIdx++){
                    float SumAssignError = 0;
                    std::vector<float> VecLineAlpha;
                    for (size_t VecIdx : VecInvAssignment[AlphaIdx]){
                        ErrorComputation(VecCentralClusterDist, NeiClusterDist, VecNeighborClusterDist, AlphaLineValue, VecIdx, AlphaIdx, LineIdx, ErrorResult, NumNC);
                        SumAssignError += ErrorResult[0];
                        VecLineAlpha.emplace_back(ErrorResult[1]);
                    }
                    float TestAlpha = std::accumulate(VecLineAlpha.begin(), VecLineAlpha.end(), 0.0f) / VecLineAlpha.size();
                    for (size_t i = 0; i < VecLineAlpha.size(); i++){
                        SumAssignError += (TestAlpha - VecLineAlpha[i]) * (TestAlpha - VecLineAlpha[i]);
                    }
                    if (SumAssignError < MinAlphaAssignError){
                        MinAlphaAssignError = SumAssignError;
                        ResultLineAlpha = TestAlpha;
                        ResultLineIdx = LineIdx;
                    }
                }
                assert(ResultLineIdx < NumNC);
                AlphaLineValue[AlphaIdx] = ResultLineAlpha;
                AlphaLineID[AlphaIdx] = ResultLineIdx;
                AlphaAssignError[AlphaIdx] = MinAlphaAssignError;
            }
            else{
                EmptyAlpha.emplace_back(AlphaIdx);
            }
        }

        // Handle the void clusters
        if (!EmptyAlpha.empty()){
            faiss::RandomGenerator rng(1234);

            // Update the error of vectors, assign the alpha to the one with largest error
            for (auto EmptyAlphaIdx : EmptyAlpha){
                float NonOneElementError = 0;
                for (size_t i = 0; i < NumAlpha; i++){
                    if (VecInvAssignment[i].size() > 1){
                        NonOneElementError += AlphaAssignError[i];
                    }
                }

                // We only assign the centroid to non-empty cluster
                float Threshold = rng.rand_float() * NonOneElementError;
                uint32_t IterIndex = 0;
                float Prop = 0;
                for (; IterIndex < NumAlpha; IterIndex++){
                    if (VecInvAssignment[IterIndex].size() > 1){
                        Prop += AlphaAssignError[IterIndex];
                    }
                    if (Prop >= Threshold){
                        break;
                    }
                }

                size_t midPoint = VecInvAssignment[IterIndex].size() / 2;
                VecInvAssignment[EmptyAlphaIdx].insert(VecInvAssignment[EmptyAlphaIdx].end(),std::make_move_iterator(VecInvAssignment[IterIndex].begin()), std::make_move_iterator(VecInvAssignment[IterIndex].begin() + midPoint));
                VecInvAssignment[IterIndex].erase(VecInvAssignment[IterIndex].begin(), VecInvAssignment[IterIndex].begin() + midPoint);

                AlphaLineID[EmptyAlphaIdx] = AlphaLineID[IterIndex];
                AlphaLineValue[EmptyAlphaIdx] = (1 + EPS) * AlphaLineValue[IterIndex];
                AlphaLineValue[IterIndex] *= (1 - EPS);
                AlphaAssignError[EmptyAlphaIdx] = AlphaAssignError[IterIndex] / 2;
                AlphaAssignError[IterIndex] = AlphaAssignError[IterIndex] - AlphaAssignError[EmptyAlphaIdx];
            }
        }

        SumAlphaAssignError = std::accumulate(AlphaAssignError.begin(), AlphaAssignError.end(), 0.0f);
        if (SumAlphaAssignError < PreviousSumAssignError){
            PreviousSumAssignError = SumAlphaAssignError;
        }
        else if(SumAlphaAssignError > PreviousSumAssignError + EPS){
            std::cout << "Error in the " << Iter << " th training with error from " << PreviousSumAssignError << " to " << SumAlphaAssignError << "\n";
            exit(0);
        }
        else{
            ContinueFlag = false;
        }
        //std::cout << "TrainSize: " << TrainSize << " NumCluster: " << NumAlpha <<  " Sum of error in " << Iter << " th iteration: " << SumAlphaAssignError << " with num of empty: " << EmptyAlpha.size() << "\n";
    }

    for (uint32_t AlphaIdx = 0; AlphaIdx < NumAlpha; AlphaIdx++){
        uint32_t LineID = AlphaLineID[AlphaIdx];
        AlphaLineValue[AlphaIdx] /= sqrt(NeiClusterDist[LineID]);
        AlphaLineNorm.emplace_back(AlphaLineValue[AlphaIdx] * (AlphaLineValue[AlphaIdx] - 1) * NeiClusterDist[LineID]);
        AlphaLineID[AlphaIdx] = NeiClusterID[LineID];
    }

    return SumAlphaAssignError;
}

void trainAndClusterVectors(
    std::vector<float>& BaseBatch, 
    std::vector<uint32_t>& BaseAssignment,
    std::vector<std::unordered_set<uint32_t>>& BCCluster,
    std::vector<std::unordered_set<uint32_t>>& BCClusterConflict,
    std::vector<std::vector<uint32_t>> & AssignmentNum,
    std::vector<std::vector<uint32_t>> & AssignmentID,
    std::vector<std::vector<uint32_t>> & AlphaLineIDs,
    std::vector<std::vector<float>> & AlphaLineValues,
    std::vector<std::vector<float>> & AlphaLineNorms
){

    performance_recorder recorder("trainAndClusterVectors");
    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    // Prepare the training vectors for clusters inside clusters
    std::vector<std::vector<float>> BaseTrainVectors(nc);
    std::vector<std::vector<uint32_t>> BaseTrainVectorID(nc);
    for (size_t i = 0; i < nc; i++) {
        BaseTrainVectors[i].resize(BCCluster[i].size() * Dimension);
        BaseTrainVectorID[i].resize(BCCluster[i].size());
    }
    
    for (size_t i = 0; i < Assign_num_batch; i++) {
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, true);
        #pragma omp parallel for
        for (size_t j = 0; j < Assign_batch_size; j++) {
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            // Critical section to protect shared resource
            #pragma omp critical
            {
                auto it = std::find(BCCluster[NNClusterID].begin(), BCCluster[NNClusterID].end(), NNVecID);
                if (it != BCCluster[NNClusterID].end()){
                    int position = std::distance(BCCluster[NNClusterID].begin(), it);
                    memcpy(BaseTrainVectors[NNClusterID].data() + position * Dimension, BaseBatch.data() + j * Dimension, Dimension * sizeof(float));
                    BaseTrainVectorID[NNClusterID][position] = NNVecID;
                }
            }
        }
    }
    std::cout << "Start the NeighborKmeans training\n";
    for (size_t i = 0; i < nc; i++){assert(BaseTrainVectors[i].size() == BCCluster[i].size() * Dimension);}
    // Train the vector for clustering and assignment
    AlphaLineIDs.resize(nc);
    AlphaLineValues.resize(nc);
    AlphaLineNorms.resize(nc);

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    std::vector<float> RecordAssignError(nc, 0);
    #pragma omp parallel for
    for (uint32_t ClusterID = 0; ClusterID < nc; ClusterID++){
        std::vector<std::vector<uint32_t>> VecInvAssignment;
        std::vector<float> VecError(BCCluster[ClusterID].size());
        uint32_t NumCluster = std::min(BCClusterConflict[ClusterID].size(), BCCluster[ClusterID].size());
        if (NumCluster > MaxNeSTSize){NumCluster = MaxNeSTSize;}
        RecordAssignError[ClusterID] = NeighborKmeans(CenGraph, BaseTrainVectors[ClusterID], AlphaLineIDs[ClusterID], AlphaLineValues[ClusterID], AlphaLineNorms[ClusterID], VecInvAssignment, ClusterID, NumCluster, MaxNeSTNeighbor, 10);

        uint32_t EndIndice = 0;
        for (size_t i = 0; i < NumCluster; i++){
            EndIndice += VecInvAssignment[i].size();
            AssignmentNum[ClusterID].emplace_back(EndIndice);
            for (auto ID : VecInvAssignment[i]){
                uint32_t ActualID = BaseTrainVectorID[ClusterID][ID];
                AssignmentID[ClusterID].emplace_back(ActualID);
            }
        }
    }
    delete(CenGraph);
    std::cout << "The sum of trainig error: " << std::accumulate(RecordAssignError.begin(), RecordAssignError.end(), 0.0f) << "\n";
}

void perform1DProjection(
    const std::vector<float>& Centroids,
    const std::vector<std::vector<uint32_t>>& AssignmentNum,
    const std::vector<std::vector<float>>& CentroidsinCluster,
    std::vector<std::vector<uint32_t>>& ProjectionNNClusterID,
    std::vector<std::vector<float>> & ProjectionAlpha,
    std::vector<std::vector<float>> & ProjectionAlphaNorm
) {
    performance_recorder recorder("perform1DProjection");
    hnswlib::L2Space l2space(Dimension);
    auto CenGraph = std::make_unique<hnswlib::HierarchicalNSW<float>>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        auto Result = CenGraph->searchKnn(Centroids.data() + i* Dimension, MaxNeSTSize + 1, NumComp);

        std::vector<uint32_t> NNClusterIDChoice(MaxNeSTSize); // The projection choice
        std::vector<float> NNClusterDistChoice(MaxNeSTSize);

        // Projection to the close clusters (with high probability to be re-used)
        for (size_t j = 0; j < MaxNeSTSize; ++j) {
            NNClusterIDChoice[MaxNeSTSize - j - 1] = Result.top().second;
            NNClusterDistChoice[MaxNeSTSize - j - 1] = Result.top().first;
            Result.pop();
        }

        for (size_t j = 0; j < AssignmentNum[i].size(); ++j) {
            float MinProjectionError = std::numeric_limits<float>::max();
            float ResultAlpha = 0.0f;
            float ResultAlphaNorm = 0.0f;
            uint32_t ResultProjID = nc;

            float CentralDist = faiss::fvec_L2sqr(CentroidsinCluster[i].data() + j * Dimension, Centroids.data() + i * Dimension, Dimension);

            if (CentralDist < 1e-5){
                ProjectionNNClusterID[i].emplace_back(i);
                ProjectionAlpha[i].emplace_back(0);
                ProjectionAlphaNorm[i].emplace_back(0);
                continue;
            }
            for (size_t k = 0; k < MaxNeSTSize; ++k) {
                float NNDist = faiss::fvec_L2sqr(CentroidsinCluster[i].data() + j * Dimension, Centroids.data() + NNClusterIDChoice[k] * Dimension, Dimension);
                assert(CentralDist <= NNDist);
                float CenNNDist = NNClusterDistChoice[k];

                float CosNLNNTarget = (CentralDist + CenNNDist - NNDist) / (2 * sqrt(CentralDist * CenNNDist));
                float SqrError = CentralDist * (1 - CosNLNNTarget * CosNLNNTarget);

                if (SqrError < MinProjectionError) {
                    ResultProjID = NNClusterIDChoice[k];
                    ResultAlpha = CosNLNNTarget * sqrt(CentralDist / CenNNDist);
                    ResultAlphaNorm = ResultAlpha * (ResultAlpha - 1) * CenNNDist;
                    MinProjectionError = SqrError;
                }
            }

            if (ResultProjID < nc) {
                ProjectionNNClusterID[i].emplace_back(ResultProjID);
                ProjectionAlpha[i].emplace_back(ResultAlpha);
                ProjectionAlphaNorm[i].emplace_back(ResultAlphaNorm);
            }
            else{
                std::cout << "Projection Error\n";
                exit(0);
            }
        }
    }
}

template<typename T>
void writeVector(std::ofstream& outFile, const std::vector<std::vector<T>>& vec) {
    size_t outerSize = vec.size();
    outFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(size_t));
    for (const auto& innerVec : vec) {
        size_t innerSize = innerVec.size();
        outFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(size_t));
        outFile.write(reinterpret_cast<const char*>(innerVec.data()), innerSize * sizeof(T));
    }
}

template<typename T>
void readVector(std::ifstream& inFile, std::vector<std::vector<T>>& vec) {
    size_t outerSize;
    inFile.read(reinterpret_cast<char*>(&outerSize), sizeof(size_t));
    vec.resize(outerSize);
    for (auto& innerVec : vec) {
        size_t innerSize;
        inFile.read(reinterpret_cast<char*>(&innerSize), sizeof(size_t));
        innerVec.resize(innerSize);
        inFile.read(reinterpret_cast<char*>(innerVec.data()), innerSize * sizeof(T));
    }
}


std::vector<int> rankFloats(const std::vector<float>& nums) {
    std::vector<std::pair<float, int>> indexed_nums;
    indexed_nums.reserve(nums.size());
    for (int i = 0; i < nums.size(); ++i) {
        indexed_nums.push_back({nums[i], i});
    }

    std::sort(indexed_nums.begin(), indexed_nums.end());

    std::vector<int> ranks(nums.size());
    for (int i = 0; i < indexed_nums.size(); ++i) {
        ranks[indexed_nums[i].second] = i;
    }

    return ranks;
}