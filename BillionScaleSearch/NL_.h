#pragma once

#include <string>
#include <unistd.h>
#include <set>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include "parametersNL/million/result.h"
#include "utils/utils.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib_old/hnswalg.h"
#include "FastPFor/headers/codecfactory.h"
//#include "complib/include/codecfactory.h"

constexpr float UNDEFINED_TARGET_CLUSTER_DIST = -1.0f;
constexpr size_t EXTRA_COMP_SIZE = 1024;

void loadBoundaryConflictCentroidsAlpha(
    std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>>& BoundaryConflictCentroidAlpha,
    const std::string& path = PathBoundaryConflictMapCentroid) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Error opening path file: " << path << "\n"; exit(1); }

    uint32_t vectorSize, innerSize, innerKey;
    float alpha, alphadist;

    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(uint32_t)); // Reading the size of the vector

    BoundaryConflictCentroidAlpha.resize(vectorSize);

    for (auto& innerMap : BoundaryConflictCentroidAlpha) {
        file.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t)); // Reading the size of the inner map

        for (uint32_t i = 0; i < innerSize; ++i) {
            file.read(reinterpret_cast<char*>(&innerKey), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&alpha), sizeof(float));
            file.read(reinterpret_cast<char*>(&alphadist), sizeof(float));

            uint32_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(uint32_t)); // Reading the size of the float vector

            std::vector<float> values(vectorSize);
            for (uint32_t j = 0; j < vectorSize; ++j) {
                file.read(reinterpret_cast<char*>(&values[j]), sizeof(float)); // Reading the float values
            }

            innerMap[innerKey] = std::make_pair(std::make_pair(alpha, alphadist), std::move(values));
        }
    }
    file.close();
}

void writeBoundaryConflictCentroidsAlpha(
    const std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>>& BoundaryConflictCentroidAlpha,
    const std::string& path = PathBoundaryConflictMapCentroid) {
    std::ofstream CentroidFile(path, std::ios::binary);
    if (!CentroidFile) { std::cerr << "Error opening path file: " << path << "\n"; exit(1); }

    uint32_t vectorSize = BoundaryConflictCentroidAlpha.size();
    CentroidFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(uint32_t)); // Writing the size of the vector

    for (const auto& innerMap : BoundaryConflictCentroidAlpha) {
        uint32_t innerSize = innerMap.size();
        CentroidFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t)); // Writing the size of the inner map

        for (const auto& innerPair : innerMap) {
            uint32_t innerKey = innerPair.first;
            CentroidFile.write(reinterpret_cast<const char*>(&innerKey), sizeof(uint32_t));

            float alpha = innerPair.second.first.first;
            CentroidFile.write(reinterpret_cast<const char*>(&alpha), sizeof(float)); // Writing the alpha

            float alphadist = innerPair.second.first.second;
            CentroidFile.write(reinterpret_cast<const char*>(&alphadist), sizeof(float)); // Writing the alpha distance

            uint32_t vectorSize = innerPair.second.second.size();
            CentroidFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(uint32_t)); // Writing the size of the float vector
            
            for (const auto& value : innerPair.second.second) {
                CentroidFile.write(reinterpret_cast<const char*>(&value), sizeof(float)); // Writing the float values
            }
        }
    }
    CentroidFile.close();
}

void loadBoundaryConflictMapDist(std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>>& map,
const std::string& path = PathBoundaryConflictMapDist
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {std::cerr << "Error opening path file: " << path << "\n";exit(1);}
    while (file.peek() != EOF) {
        uint32_t key1, key2, pair_value;
        float tuple_val0, tuple_val1, tuple_val2, tuple_val3;
        file.read(reinterpret_cast<char*>(&key1), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&key2), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&pair_value), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&tuple_val0), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val1), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val2), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val3), sizeof(float));
        map[{key1, key2}] = {pair_value, {tuple_val0, tuple_val1, tuple_val2, tuple_val3}};
    }
    file.close();
}

void writeBoundaryConflictMapDist(
    const std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>>& BoundaryConflictMapDist,
    const std::string& path = PathBoundaryConflictMapDist) {
    std::ofstream DistFile(path, std::ios::binary);
    if (!DistFile) {std::cerr << "Error opening path file: " << path << "\n";exit(1);}
    for (const auto& pair : BoundaryConflictMapDist) {
        DistFile.write(reinterpret_cast<const char*>(&pair.first.first), sizeof(uint32_t));
        DistFile.write(reinterpret_cast<const char*>(&pair.first.second), sizeof(uint32_t));
        DistFile.write(reinterpret_cast<const char*>(&pair.second.first), sizeof(uint32_t));
        // Write tuple
        auto tuple_values = pair.second.second;
        DistFile.write(reinterpret_cast<const char*>(&std::get<0>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<1>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<2>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<3>(tuple_values)), sizeof(float));
    }
    DistFile.close();
}

void loadBoundaryConflictMapVec(
std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>>& map,
const std::string& path=PathBoundaryConflictMapVec) {

    std::ifstream VecFile(path, std::ios::binary);
    if (!VecFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    uint32_t outerSize;
    uint32_t key;
    uint32_t innerSize;
    uint32_t innerKey;
    size_t innerValue;
    // Read the size of the outer map
    VecFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    for (uint32_t j = 0; j < outerSize; ++j) {
        VecFile.read(reinterpret_cast<char*>(&key), sizeof(uint32_t));
        VecFile.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t));
        
        std::unordered_map<uint32_t, size_t> innerMap;

        for (uint32_t i = 0; i < innerSize; ++i) {
            VecFile.read(reinterpret_cast<char*>(&innerKey), sizeof(uint32_t));
            VecFile.read(reinterpret_cast<char*>(&innerValue), sizeof(size_t));
            innerMap[innerKey] = innerValue;
        }

        map[key] = innerMap;
    }
    VecFile.close();
}

void writeBoundaryConflictMapVec(const std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>>& BoundaryConflictMapVec, 
const std::string& path=PathBoundaryConflictMapVec) {
    std::ofstream VecFile(path, std::ios::binary);
    if (!VecFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    
    // Write the size of the outer map first
    uint32_t outerSize = static_cast<uint32_t>(BoundaryConflictMapVec.size());
    VecFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    for (const auto& outerElem : BoundaryConflictMapVec) {
        VecFile.write(reinterpret_cast<const char*>(&outerElem.first), sizeof(uint32_t));

        // Write the size of the inner map for this entry
        uint32_t innerSize = static_cast<uint32_t>(outerElem.second.size());
        VecFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t));
        
        for (const auto& innerElem : outerElem.second) {
            VecFile.write(reinterpret_cast<const char*>(&innerElem.first), sizeof(uint32_t));
            VecFile.write(reinterpret_cast<const char*>(&innerElem.second), sizeof(size_t));
        }
    }
    VecFile.close();
}


void loadBoundaryConflictClusterResult(
    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictClusterResult,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    uint32_t outerSize;
    uint32_t middleSize;
    uint32_t innerSize;
    uint32_t key;
    uint32_t value;

    ClusterFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    for (uint32_t i = 0; i < outerSize; ++i) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> middleMap;
        ClusterFile.read(reinterpret_cast<char*>(&middleSize), sizeof(uint32_t));

        for (uint32_t j = 0; j < middleSize; ++j) {
            ClusterFile.read(reinterpret_cast<char*>(&key), sizeof(uint32_t));
            ClusterFile.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t));

            std::vector<uint32_t> innerVector;
            for (uint32_t k = 0; k < innerSize; ++k) {
                ClusterFile.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
                innerVector.push_back(value);
            }

            middleMap[key] = innerVector;
        }

        BoundaryConflictClusterResult.push_back(middleMap);
    }

    ClusterFile.close();
}

void writeBoundaryConflictClusterResult(
    const std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictClusterResult,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    uint32_t outerSize = static_cast<uint32_t>(BoundaryConflictClusterResult.size());
    ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    for (const auto& outerElem : BoundaryConflictClusterResult) {
        uint32_t middleSize = static_cast<uint32_t>(outerElem.size());
        ClusterFile.write(reinterpret_cast<const char*>(&middleSize), sizeof(uint32_t));

        for (const auto& middleElem : outerElem) {
            ClusterFile.write(reinterpret_cast<const char*>(&middleElem.first), sizeof(uint32_t));
            uint32_t innerSize = static_cast<uint32_t>(middleElem.second.size());
            ClusterFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t));

            for (const auto& innerElem : middleElem.second) {
                ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
            }
        }
    }

    ClusterFile.close();
}

void loadBoundaryConflictMapCluster(
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>& map,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    uint32_t key1;
    uint32_t size1;
    uint32_t key2;
    uint32_t size2;
    uint32_t innerValue;

    while (ClusterFile.read(reinterpret_cast<char*>(&key1), sizeof(uint32_t))) {
        ClusterFile.read(reinterpret_cast<char*>(&size1), sizeof(uint32_t));

        for (uint32_t i = 0; i < size1; ++i) {
            ClusterFile.read(reinterpret_cast<char*>(&key2), sizeof(uint32_t));
            ClusterFile.read(reinterpret_cast<char*>(&size2), sizeof(uint32_t));

            std::vector<uint32_t> innerVector;
            for (uint32_t j = 0; j < size2; ++j) {
                ClusterFile.read(reinterpret_cast<char*>(&innerValue), sizeof(uint32_t));
                innerVector.push_back(innerValue);
            }

            map[key1][key2] = innerVector;
        }
    }
    ClusterFile.close();
}

void writeBoundaryConflictMapCluster(
    const std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictMapCluster, 
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    for (const auto& outerElem : BoundaryConflictMapCluster) {
        ClusterFile.write(reinterpret_cast<const char*>(&outerElem.first), sizeof(uint32_t));
        uint32_t outerSize = static_cast<uint32_t>(outerElem.second.size());
        ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

        for (const auto& middleElem : outerElem.second) {
            ClusterFile.write(reinterpret_cast<const char*>(&middleElem.first), sizeof(uint32_t));
            uint32_t middleSize = static_cast<uint32_t>(middleElem.second.size());
            ClusterFile.write(reinterpret_cast<const char*>(&middleSize), sizeof(uint32_t));

            for (const auto& innerElem : middleElem.second) {
                ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
            }
        }
    }
    ClusterFile.close();
}

void loadBoundaryConflictVecResult(
    std::vector<uint32_t>& BoundaryConflictVecResult,
    const std::string& path = PathBoundaryConflictVecResult)
{
    std::ifstream VecResultFile(path, std::ios::binary);
    if (!VecResultFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    BoundaryConflictVecResult.reserve(nb);
    uint32_t value;
    while (VecResultFile.read(reinterpret_cast<char*>(&value), sizeof(uint32_t))) {
        BoundaryConflictVecResult.push_back(value);
    }

    VecResultFile.close();
}

void writeBoundaryConflictVecResult(
    const std::vector<uint32_t>& BoundaryConflictVecResult,
    const std::string& path = PathBoundaryConflictVecResult)
{
    std::ofstream VecResultFile(path, std::ios::binary);
    if (!VecResultFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    for (const auto& elem : BoundaryConflictVecResult) {
        VecResultFile.write(reinterpret_cast<const char*>(&elem), sizeof(uint32_t));
    }
    VecResultFile.close();
}

void loadNonBCLists(
    std::vector<std::vector<uint32_t>>& NonBCLists,
    const std::string& path = PathNonBCLists)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    // Read the number of lists (outer vector size)
    uint32_t outerSize;
    ClusterFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    // Iterate through the outer vector
    for (uint32_t i = 0; i < outerSize; ++i) {
        // Read the size of the inner vector
        uint32_t size;
        ClusterFile.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

        // Read the inner vector
        std::vector<uint32_t> values(size); // Pre-allocate the inner vector with the correct size
        for (uint32_t j = 0; j < size; ++j) {
            ClusterFile.read(reinterpret_cast<char*>(&values[j]), sizeof(uint32_t));
        }

        // Add the inner vector to the outer vector
        NonBCLists.push_back(values);
    }

    ClusterFile.close();
}

void writeNonBCLists(
    const std::vector<std::vector<uint32_t>>& NonBCLists, 
    const std::string& path = PathNonBCLists)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    // Write the number of lists (outer vector size)
    uint32_t outerSize = static_cast<uint32_t>(NonBCLists.size());
    ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    // Iterate through the outer vector
    for (const auto& outerElem : NonBCLists) {
        // Write the size of the inner vector
        uint32_t size = static_cast<uint32_t>(outerElem.size());
        ClusterFile.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

        // Iterate through the inner vector
        for (const auto& innerElem : outerElem) {
            ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
        }
    }

    ClusterFile.close();
}


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
    cp.niter = 100;
    cp.verbose = true;
    faiss::Clustering clustering(Dimension, nc, cp);
    clustering.train(nt, training_data, index);
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

void writeNeighborList(
    const std::vector<std::vector<uint32_t>>& queryClusterIDs,
    const std::vector<std::vector<uint32_t>>& queryClusterIDIdx,
    const std::vector<std::vector<uint32_t>>& allVectorIDs,
    const std::vector<std::vector<float>>& AlphaCentroidDists,
    const std::vector<std::vector<float>>& AlphaCentroid,
    const std::string& path = PathNeighborList) {

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << "\n";
        return;
    }

    uint32_t nc = queryClusterIDs.size();
    file.write(reinterpret_cast<const char*>(&nc), sizeof(uint32_t));

    for (uint32_t i = 0; i < nc; i++) {
        uint32_t size = queryClusterIDs[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(queryClusterIDs[i].data()), size * sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(queryClusterIDIdx[i].data()), size * sizeof(uint32_t));

        uint32_t allVectorIDSize = allVectorIDs[i].size();
        if (!queryClusterIDIdx[i].empty()) {
            assert(queryClusterIDIdx[i].back() == allVectorIDSize);
        } else {
            assert(allVectorIDSize == 0);
        }
        file.write(reinterpret_cast<const char*>(&allVectorIDSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(allVectorIDs[i].data()), allVectorIDSize * sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(AlphaCentroidDists[i].data()), size * sizeof(float));
        file.write(reinterpret_cast<const char*>(AlphaCentroid[i].data()), size * sizeof(float));
    }
    file.close();
}


void loadNeighborList(
    std::vector<std::vector<uint32_t>>& queryClusterIDs,
    std::vector<std::vector<uint32_t>>& queryClusterIDIdx,
    std::vector<std::vector<uint32_t>>& allVectorIDs,
    std::vector<std::vector<float>>& AlphaCentroidDists,
    std::vector<std::vector<float>>& AlphaCentroid,
    const std::string& path = PathNeighborList) {

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << "\n";
        return;
    }

    uint32_t nc;
    file.read(reinterpret_cast<char*>(&nc), sizeof(uint32_t));

    queryClusterIDs.resize(nc);
    queryClusterIDIdx.resize(nc);
    allVectorIDs.resize(nc);
    AlphaCentroidDists.resize(nc);
    AlphaCentroid.resize(nc);

    for (uint32_t i = 0; i < nc; i++) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

        queryClusterIDs[i].resize(size);
        queryClusterIDIdx[i].resize(size);
        AlphaCentroidDists[i].resize(size);
        AlphaCentroid[i].resize(size);

        file.read(reinterpret_cast<char*>(queryClusterIDs[i].data()), size * sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(queryClusterIDIdx[i].data()), size * sizeof(uint32_t));

        uint32_t allVectorIDSize;
        file.read(reinterpret_cast<char*>(&allVectorIDSize), sizeof(uint32_t));
        allVectorIDs[i].resize(allVectorIDSize);
        file.read(reinterpret_cast<char*>(allVectorIDs[i].data()), allVectorIDSize * sizeof(uint32_t));

        if (!queryClusterIDIdx[i].empty()) {
            assert(queryClusterIDIdx[i].back() == allVectorIDSize);
        } else {
            assert(allVectorIDSize == 0);
        }
        file.read(reinterpret_cast<char*>(AlphaCentroidDists[i].data()), size * sizeof(float));
        file.read(reinterpret_cast<char*>(AlphaCentroid[i].data()), size * sizeof(float));
    }

    file.close();
}


void computeCentroidNorm() {

    performance_recorder recorder("computeCentroidsNorm");
    std::vector<float> CentroidsNorm(nc);
    std::vector<float> Centroids(nc * Dimension);
    std::cout << "Computing the distance between centroid vectors \n";
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {
        std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";
        exit(1);
    }
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    faiss::fvec_norms_L2sqr(CentroidsNorm.data(), Centroids.data(), Dimension, nc);
    std::ofstream CentroidDistOutputStream(PathCentroidNorm, std::ios::binary);
    if (!CentroidDistOutputStream) {
        std::cerr << "Error opening PathCentroidNorm file: " << PathCentroidNorm << "\n";
        exit(1);
    }
    CentroidDistOutputStream.write((char*) CentroidsNorm.data(), nc * sizeof(float));
    CentroidDistOutputStream.close();
    recorder.print_performance("computeCentroidsNorm");
}


struct ResultProcessor {
    std::unordered_map<uint32_t, float> idDistanceMap;
    std::vector<uint32_t> IndexStoreNNClusterID;
    std::vector<uint32_t> IndexStoreNNListID;

    uint32_t CentralClusterID = 0;
    uint32_t pairIdOffset;

    ResultProcessor(std::priority_queue<std::pair<float, size_t>> ResultQueue, uint32_t nc, uint32_t ef) {
        idDistanceMap.reserve(ef);
        pairIdOffset = nc; // Ensure pair IDs don't overlap with original IDs
        while (!ResultQueue.empty()) {
            uint32_t id = ResultQueue.top().second;
            float distance = ResultQueue.top().first;

            ResultQueue.pop();
            if (ResultQueue.empty()){
                CentralClusterID = id;
                break;
            }
            idDistanceMap[id] = distance;
        }
    }

    uint32_t createPairID(uint32_t id1, uint32_t index) {
        IndexStoreNNClusterID.emplace_back(id1);
        IndexStoreNNListID.emplace_back(index);
        return pairIdOffset + IndexStoreNNClusterID.size() - 1;
    }

    void processIDs(std::priority_queue<std::pair<float, size_t>> ResultQueue, size_t ef, std::unordered_set<uint32_t>& originalIDs, std::unordered_map<uint32_t, std::vector<uint32_t>> & NListID) {
        std::vector<uint32_t>  IDIndex;
        IDIndex.reserve(ef);

        while (!ResultQueue.empty()) {

            auto element = ResultQueue.top();
            ResultQueue.pop();

            auto id = element.second;
            if (id < pairIdOffset) {
                if (id != CentralClusterID){
                    originalIDs.insert(id);
                }
            }
            else {
                int idx = id - pairIdOffset;
                assert(idx < IndexStoreNNClusterID.size());
                IDIndex.emplace_back(idx);
            }
        }

        for (const auto& idx : IDIndex){
            if (originalIDs.find(IndexStoreNNClusterID[idx]) == originalIDs.end()){
                NListID[IndexStoreNNClusterID[idx]].emplace_back(IndexStoreNNListID[idx]);
            }
        }
    }
};


void pruneBaseSet(std::vector<float>& BaseBatch, std::vector<uint32_t> & BaseAssignment,
std::vector<float> Centroids, std::vector<std::vector<uint32_t>> BCCluster,
std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> BCClusterPrune
){
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

    // Compute the average centroid
    for (uint32_t i = 0; i < nc; i++){
        for (auto ClusterIt = BCClusterPrune[i].begin(); ClusterIt != BCClusterPrune[i].end(); ClusterIt++){
            uint32_t QueryClusterID = ClusterIt->first;
            for (uint32_t j = 0; j < Dimension; j++){
                BCCentroids[i][QueryClusterID][j] /= ClusterIt->second.size();
            }
        }
    }
    // Compute the distance bound
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
            if (std::find(BCCluster[NNClusterID].begin(), BCCluster[NNClusterID].end(), NNVecID) != BCCluster[NNClusterID].end());

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
                    BCCluster[NNClusterID].emplace_back(NNVecID);
                    break;
                }
            }
        }
    }
}


void trainAndClusterVectors(
    std::vector<float>& BaseBatch, 
    std::vector<uint32_t>& BaseAssignment,
    std::vector<std::vector<uint32_t>>& BCCluster,
    std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> & BCClusterPrune,
    std::vector<std::unordered_set<uint32_t>>& BCClusterConflict,
    std::vector<std::vector<float>> CentroidsinCluster,
    std::vector<std::vector<uint32_t>> AssignmentNum
) {

    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    // Prepare the training vectors for clusters inside cluster
    size_t nc = BCCluster.size();
    std::vector<std::vector<float>> BaseTrainVectors(nc);
    for (size_t i = 0; i < nc; i++) {BaseTrainVectors[i].resize(BCCluster[i].size() * Dimension);}
    for (size_t i = 0; i < Assign_num_batch; i++) {
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, true);
        for (size_t j = 0; j < Assign_batch_size; j++) {
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            auto it = std::find(BCCluster[NNClusterID].begin(), BCCluster[NNClusterID].end(), NNVecID);
            if (it != BCCluster[NNClusterID].end()) {
                int position = std::distance(BCCluster[NNClusterID].begin(), it);
                memcpy(BaseTrainVectors[NNClusterID].data() + position * Dimension, BaseBatch.data() + j * Dimension, Dimension * sizeof(float));
            }
        }
    }

    for (size_t i = 0; i < nc; i++) {assert(BaseTrainVectors[i].size() == BCCluster[i].size() * Dimension);}

    // Train the vector for clustering and assignment
    std::vector<std::vector<uint32_t>> AssignmentID(nc);
   #pragma omp parallel for
    for (size_t i = 0; i < nc; i++) {
        uint32_t NumCluster = prunevectors ? BCClusterPrune[i].size() : BCClusterConflict[i].size();
        
        if (NumCluster > MaxNeSTSize) {NumCluster = MaxNeSTSize;}
        if (NumCluster < MinNeSTSize) {NumCluster = MinNeSTSize;}

        std::vector<std::vector<uint32_t>> InvID(NumCluster);
        CentroidsinCluster[i].resize(NumCluster * Dimension);

        faiss::kmeans_clustering(Dimension, BCCluster[i].size(), NumCluster, BaseTrainVectors[i].data(), CentroidsinCluster[i].data());

        faiss::IndexFlatL2 index(Dimension);
        index.add(NumCluster, CentroidsinCluster[i].data());

        std::vector<float> Distance(BCCluster[i].size());
        std::vector<int64_t> ID(BCCluster[i].size());

        index.search(BCCluster[i].size(), BaseTrainVectors[i].data(), 1, Distance.data(), ID.data());

        uint32_t EndIndice = 0;
        for (size_t j = 0; j < BCCluster[i].size(); j++) {
            InvID[ID[j]].emplace_back(BCCluster[i][j]);
        }

        for (size_t j = 0; j < NumCluster; j++) {
            EndIndice += InvID[j].size();
            AssignmentNum[i].emplace_back(EndIndice);
            for (const auto& id : InvID[j]) {
                AssignmentID[i].emplace_back(id);
            }
        }
    }
}


void perform1DProjection(
    const std::vector<float>& Centroids,
    const std::vector<std::vector<uint32_t>>& AssignmentNum,
    const std::vector<std::vector<float>>& CentroidsinCluster
) {
    hnswlib::L2Space l2space(Dimension);
    auto CenGraph = std::make_unique<hnswlib::HierarchicalNSW<float>>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    std::vector<std::vector<uint32_t>> ProjectionNNClusterID(nc);
    std::vector<std::vector<float>> ProjectionAlpha(nc);
    std::vector<std::vector<float>> ProjectionAlphaNorm(nc);

    #pragma omp parallel for
    for (size_t i = 0; i < nc; ++i) {
        auto Result = CenGraph->searchKnn(Centroids.data(), MaxNeSTSize + 1);

        std::vector<uint32_t> NNClusterIDChoice(MaxNeSTSize);
        std::vector<float> NNClusterDistChoice(MaxNeSTSize);

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

            for (size_t k = 0; k < MaxNeSTSize; ++k) {
                float CentralDist = faiss::fvec_L2sqr(CentroidsinCluster[i].data() + j * Dimension, Centroids.data() + i * Dimension, Dimension);
                float NNDist = faiss::fvec_L2sqr(CentroidsinCluster[i].data() + j * Dimension, Centroids.data() + NNClusterIDChoice[k] * Dimension, Dimension);
                float CenNNDist = NNClusterDistChoice[k];

                float CosNLNNTarget = (CentralDist + CenNNDist - NNDist) / (2 * sqrt(CentralDist * CenNNDist));
                float SqrError = CenNNDist * (1 - CosNLNNTarget * CosNLNNTarget);

                if (SqrError < MinProjectionError) {
                    ResultProjID = NNClusterIDChoice[k];
                    ResultAlpha = CosNLNNTarget * sqrt(NNDist / CenNNDist);
                    ResultAlphaNorm = ResultAlpha * (ResultAlpha - 1) * CenNNDist;
                    MinProjectionError = SqrError;
                }
            }

            if (ResultProjID < nc) {
                ProjectionNNClusterID[i].emplace_back(ResultProjID);
                ProjectionAlpha[i].emplace_back(ResultAlpha);
                ProjectionAlphaNorm[i].emplace_back(ResultAlphaNorm);
            }
        }
    }
}


void computeNNConflictListAndSave_(){
    // Determine the number of list size based on the conflict gt
    // Clustering the conflict gt and pruned gt
    // Decide the pruning process

    if (!rerunComputeNL && exists(PathBoundaryConflictMapVec))
        return;

    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    std::vector<float> BaseBatch(Assign_batch_size * Dimension);
    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}

    std::vector<uint32_t> TrainsetNN(nt * SearchK);
    std::ifstream TrainsetNNInputStream(PathTrainsetNN, std::ios::binary);
    if (!TrainsetNNInputStream) {std::cerr << "Error opening PathBase file: " << PathTrainsetNN << "\n";exit(1);}
    TrainsetNNInputStream.read(reinterpret_cast<char*>(TrainsetNN.data()), nt * SearchK * sizeof(uint32_t));
    TrainsetNNInputStream.close();
    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);
    std::vector<uint32_t> TrainsetAssignment(nt);
    loadNeighborAssignments(PathTrainNeighborID, nt, NeighborNum, TrainsetAssignment);


    /*******************************************************************************************/
    // If we need to prune the vector, we need to store the cluster ID pair
    // Other wise, we just need to keep the central cluster ID
    std::vector<std::vector<uint32_t>> BCCluster(nc);
    std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> BCClusterPrune;
    std::vector<std::unordered_set<uint32_t>> BCClusterConflict;

    if (prunevectors){
        BCClusterPrune.resize(nc);
    }
    else{
        BCClusterConflict.resize(nc);
    }

    for (size_t i = 0; i < nt; i++){
        uint32_t QueryClusterID = TrainsetAssignment[i];
        for (size_t j = 0; j < NLClusterTargetK; j++){
            uint32_t NNVecID = TrainsetNN[i * SearchK + j];
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            if (NNClusterID != QueryClusterID){
                BCCluster[NNClusterID].emplace_back(NNVecID);
                if (prunevectors){
                    BCClusterPrune[NNClusterID][QueryClusterID].insert(NNVecID);
                }
                else{
                    BCClusterConflict[NNClusterID].insert(QueryClusterID);
                }
            }
        }
    }
    /*******************************************************************************************/

    if (prunevectors){
        // Compute the distance to the centroid of cluster pairs
        // Firstly compute the centroids with the conflict vectors
        pruneBaseSet(BaseBatch, BaseAssignment, Centroids, BCCluster, BCClusterPrune);
    }

    /*******************************************************************************************/
    std::vector<std::vector<float>> CentroidsinCluster(nc);    
    std::vector<std::vector<uint32_t>> AssignmentNum(nc);
    trainAndClusterVectors(BaseBatch, BaseAssignment, BCCluster, BCClusterPrune, BCClusterConflict, CentroidsinCluster, AssignmentNum);
    /*****************************************************************************************************/
    // 1D projection for distance compuatation
    perform1DProjection(Centroids, AssignmentNum, CentroidsinCluster);
}



void loadBoundaryConflictCentroidsAlpha(
    std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>>& BoundaryConflictCentroidAlpha,
    const std::string& path = PathBoundaryConflictMapCentroid) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Error opening path file: " << path << "\n"; exit(1); }

    uint32_t vectorSize, innerSize, innerKey;
    float alpha, alphadist;

    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(uint32_t)); // Reading the size of the vector

    BoundaryConflictCentroidAlpha.resize(vectorSize);

    for (auto& innerMap : BoundaryConflictCentroidAlpha) {
        file.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t)); // Reading the size of the inner map

        for (uint32_t i = 0; i < innerSize; ++i) {
            file.read(reinterpret_cast<char*>(&innerKey), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&alpha), sizeof(float));
            file.read(reinterpret_cast<char*>(&alphadist), sizeof(float));

            uint32_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(uint32_t)); // Reading the size of the float vector

            std::vector<float> values(vectorSize);
            for (uint32_t j = 0; j < vectorSize; ++j) {
                file.read(reinterpret_cast<char*>(&values[j]), sizeof(float)); // Reading the float values
            }

            innerMap[innerKey] = std::make_pair(std::make_pair(alpha, alphadist), std::move(values));
        }
    }
    file.close();
}



void writeBoundaryConflictCentroidsAlpha(
    const std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>>& BoundaryConflictCentroidAlpha,
    const std::string& path = PathBoundaryConflictMapCentroid) {
    std::ofstream CentroidFile(path, std::ios::binary);
    if (!CentroidFile) { std::cerr << "Error opening path file: " << path << "\n"; exit(1); }

    uint32_t vectorSize = BoundaryConflictCentroidAlpha.size();
    CentroidFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(uint32_t)); // Writing the size of the vector

    for (const auto& innerMap : BoundaryConflictCentroidAlpha) {
        uint32_t innerSize = innerMap.size();
        CentroidFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t)); // Writing the size of the inner map

        for (const auto& innerPair : innerMap) {
            uint32_t innerKey = innerPair.first;
            CentroidFile.write(reinterpret_cast<const char*>(&innerKey), sizeof(uint32_t));

            float alpha = innerPair.second.first.first;
            CentroidFile.write(reinterpret_cast<const char*>(&alpha), sizeof(float)); // Writing the alpha

            float alphadist = innerPair.second.first.second;
            CentroidFile.write(reinterpret_cast<const char*>(&alphadist), sizeof(float)); // Writing the alpha distance

            uint32_t vectorSize = innerPair.second.second.size();
            CentroidFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(uint32_t)); // Writing the size of the float vector
            
            for (const auto& value : innerPair.second.second) {
                CentroidFile.write(reinterpret_cast<const char*>(&value), sizeof(float)); // Writing the float values
            }
        }
    }
    CentroidFile.close();
}

void loadBoundaryConflictMapDist(std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>>& map,
const std::string& path = PathBoundaryConflictMapDist
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {std::cerr << "Error opening path file: " << path << "\n";exit(1);}
    while (file.peek() != EOF) {
        uint32_t key1, key2, pair_value;
        float tuple_val0, tuple_val1, tuple_val2, tuple_val3;
        file.read(reinterpret_cast<char*>(&key1), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&key2), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&pair_value), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&tuple_val0), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val1), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val2), sizeof(float));
        file.read(reinterpret_cast<char*>(&tuple_val3), sizeof(float));
        map[{key1, key2}] = {pair_value, {tuple_val0, tuple_val1, tuple_val2, tuple_val3}};
    }
    file.close();
}

void writeBoundaryConflictMapDist(
    const std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>>& BoundaryConflictMapDist,
    const std::string& path = PathBoundaryConflictMapDist) {
    std::ofstream DistFile(path, std::ios::binary);
    if (!DistFile) {std::cerr << "Error opening path file: " << path << "\n";exit(1);}
    for (const auto& pair : BoundaryConflictMapDist) {
        DistFile.write(reinterpret_cast<const char*>(&pair.first.first), sizeof(uint32_t));
        DistFile.write(reinterpret_cast<const char*>(&pair.first.second), sizeof(uint32_t));
        DistFile.write(reinterpret_cast<const char*>(&pair.second.first), sizeof(uint32_t));
        // Write tuple
        auto tuple_values = pair.second.second;
        DistFile.write(reinterpret_cast<const char*>(&std::get<0>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<1>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<2>(tuple_values)), sizeof(float));
        DistFile.write(reinterpret_cast<const char*>(&std::get<3>(tuple_values)), sizeof(float));
    }
    DistFile.close();
}

void loadBoundaryConflictMapVec(
std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>>& map,
const std::string& path=PathBoundaryConflictMapVec) {

    std::ifstream VecFile(path, std::ios::binary);
    if (!VecFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    uint32_t outerSize;
    uint32_t key;
    uint32_t innerSize;
    uint32_t innerKey;
    size_t innerValue;
    // Read the size of the outer map
    VecFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    for (uint32_t j = 0; j < outerSize; ++j) {
        VecFile.read(reinterpret_cast<char*>(&key), sizeof(uint32_t));
        VecFile.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t));
        
        std::unordered_map<uint32_t, size_t> innerMap;

        for (uint32_t i = 0; i < innerSize; ++i) {
            VecFile.read(reinterpret_cast<char*>(&innerKey), sizeof(uint32_t));
            VecFile.read(reinterpret_cast<char*>(&innerValue), sizeof(size_t));
            innerMap[innerKey] = innerValue;
        }

        map[key] = innerMap;
    }
    VecFile.close();
}

void writeBoundaryConflictMapVec(const std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>>& BoundaryConflictMapVec, 
const std::string& path=PathBoundaryConflictMapVec) {
    std::ofstream VecFile(path, std::ios::binary);
    if (!VecFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    
    // Write the size of the outer map first
    uint32_t outerSize = static_cast<uint32_t>(BoundaryConflictMapVec.size());
    VecFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    for (const auto& outerElem : BoundaryConflictMapVec) {
        VecFile.write(reinterpret_cast<const char*>(&outerElem.first), sizeof(uint32_t));

        // Write the size of the inner map for this entry
        uint32_t innerSize = static_cast<uint32_t>(outerElem.second.size());
        VecFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t));
        
        for (const auto& innerElem : outerElem.second) {
            VecFile.write(reinterpret_cast<const char*>(&innerElem.first), sizeof(uint32_t));
            VecFile.write(reinterpret_cast<const char*>(&innerElem.second), sizeof(size_t));
        }
    }
    VecFile.close();
}


void loadBoundaryConflictClusterResult(
    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictClusterResult,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    uint32_t outerSize;
    uint32_t middleSize;
    uint32_t innerSize;
    uint32_t key;
    uint32_t value;

    ClusterFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    for (uint32_t i = 0; i < outerSize; ++i) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> middleMap;
        ClusterFile.read(reinterpret_cast<char*>(&middleSize), sizeof(uint32_t));

        for (uint32_t j = 0; j < middleSize; ++j) {
            ClusterFile.read(reinterpret_cast<char*>(&key), sizeof(uint32_t));
            ClusterFile.read(reinterpret_cast<char*>(&innerSize), sizeof(uint32_t));

            std::vector<uint32_t> innerVector;
            for (uint32_t k = 0; k < innerSize; ++k) {
                ClusterFile.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
                innerVector.push_back(value);
            }

            middleMap[key] = innerVector;
        }

        BoundaryConflictClusterResult.push_back(middleMap);
    }

    ClusterFile.close();
}

void writeBoundaryConflictClusterResult(
    const std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictClusterResult,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    uint32_t outerSize = static_cast<uint32_t>(BoundaryConflictClusterResult.size());
    ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    for (const auto& outerElem : BoundaryConflictClusterResult) {
        uint32_t middleSize = static_cast<uint32_t>(outerElem.size());
        ClusterFile.write(reinterpret_cast<const char*>(&middleSize), sizeof(uint32_t));

        for (const auto& middleElem : outerElem) {
            ClusterFile.write(reinterpret_cast<const char*>(&middleElem.first), sizeof(uint32_t));
            uint32_t innerSize = static_cast<uint32_t>(middleElem.second.size());
            ClusterFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(uint32_t));

            for (const auto& innerElem : middleElem.second) {
                ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
            }
        }
    }

    ClusterFile.close();
}

void loadBoundaryConflictMapCluster(
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>& map,
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    uint32_t key1;
    uint32_t size1;
    uint32_t key2;
    uint32_t size2;
    uint32_t innerValue;

    while (ClusterFile.read(reinterpret_cast<char*>(&key1), sizeof(uint32_t))) {
        ClusterFile.read(reinterpret_cast<char*>(&size1), sizeof(uint32_t));

        for (uint32_t i = 0; i < size1; ++i) {
            ClusterFile.read(reinterpret_cast<char*>(&key2), sizeof(uint32_t));
            ClusterFile.read(reinterpret_cast<char*>(&size2), sizeof(uint32_t));

            std::vector<uint32_t> innerVector;
            for (uint32_t j = 0; j < size2; ++j) {
                ClusterFile.read(reinterpret_cast<char*>(&innerValue), sizeof(uint32_t));
                innerVector.push_back(innerValue);
            }

            map[key1][key2] = innerVector;
        }
    }
    ClusterFile.close();
}

void writeBoundaryConflictMapCluster(
    const std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>& BoundaryConflictMapCluster, 
    const std::string& path = PathBoundaryConflictMapCluster)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    for (const auto& outerElem : BoundaryConflictMapCluster) {
        ClusterFile.write(reinterpret_cast<const char*>(&outerElem.first), sizeof(uint32_t));
        uint32_t outerSize = static_cast<uint32_t>(outerElem.second.size());
        ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

        for (const auto& middleElem : outerElem.second) {
            ClusterFile.write(reinterpret_cast<const char*>(&middleElem.first), sizeof(uint32_t));
            uint32_t middleSize = static_cast<uint32_t>(middleElem.second.size());
            ClusterFile.write(reinterpret_cast<const char*>(&middleSize), sizeof(uint32_t));

            for (const auto& innerElem : middleElem.second) {
                ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
            }
        }
    }
    ClusterFile.close();
}

void loadBoundaryConflictVecResult(
    std::vector<uint32_t>& BoundaryConflictVecResult,
    const std::string& path = PathBoundaryConflictVecResult)
{
    std::ifstream VecResultFile(path, std::ios::binary);
    if (!VecResultFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }
    BoundaryConflictVecResult.reserve(nb);
    uint32_t value;
    while (VecResultFile.read(reinterpret_cast<char*>(&value), sizeof(uint32_t))) {
        BoundaryConflictVecResult.push_back(value);
    }

    VecResultFile.close();
}

void writeBoundaryConflictVecResult(
    const std::vector<uint32_t>& BoundaryConflictVecResult,
    const std::string& path = PathBoundaryConflictVecResult)
{
    std::ofstream VecResultFile(path, std::ios::binary);
    if (!VecResultFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    for (const auto& elem : BoundaryConflictVecResult) {
        VecResultFile.write(reinterpret_cast<const char*>(&elem), sizeof(uint32_t));
    }
    VecResultFile.close();
}

void loadNonBCLists(
    std::vector<std::vector<uint32_t>>& NonBCLists,
    const std::string& path = PathNonBCLists)
{
    std::ifstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    // Read the number of lists (outer vector size)
    uint32_t outerSize;
    ClusterFile.read(reinterpret_cast<char*>(&outerSize), sizeof(uint32_t));

    // Iterate through the outer vector
    for (uint32_t i = 0; i < outerSize; ++i) {
        // Read the size of the inner vector
        uint32_t size;
        ClusterFile.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

        // Read the inner vector
        std::vector<uint32_t> values(size); // Pre-allocate the inner vector with the correct size
        for (uint32_t j = 0; j < size; ++j) {
            ClusterFile.read(reinterpret_cast<char*>(&values[j]), sizeof(uint32_t));
        }

        // Add the inner vector to the outer vector
        NonBCLists.push_back(values);
    }

    ClusterFile.close();
}

void writeNonBCLists(
    const std::vector<std::vector<uint32_t>>& NonBCLists, 
    const std::string& path = PathNonBCLists)
{
    std::ofstream ClusterFile(path, std::ios::binary);
    if (!ClusterFile) {
        std::cerr << "Error opening path file: " << path << "\n";
        exit(1);
    }

    // Write the number of lists (outer vector size)
    uint32_t outerSize = static_cast<uint32_t>(NonBCLists.size());
    ClusterFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(uint32_t));

    // Iterate through the outer vector
    for (const auto& outerElem : NonBCLists) {
        // Write the size of the inner vector
        uint32_t size = static_cast<uint32_t>(outerElem.size());
        ClusterFile.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

        // Iterate through the inner vector
        for (const auto& innerElem : outerElem) {
            ClusterFile.write(reinterpret_cast<const char*>(&innerElem), sizeof(uint32_t));
        }
    }

    ClusterFile.close();
}


void writeNeighborList(
    const std::vector<std::vector<uint32_t>>& queryClusterIDs,
    const std::vector<std::vector<uint32_t>>& queryClusterIDIdx,
    const std::vector<std::vector<uint32_t>>& allVectorIDs,
    const std::vector<std::vector<float>>& AlphaCentroidDists,
    const std::vector<std::vector<float>>& AlphaCentroid,
    const std::string& path = PathNeighborList) {

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << "\n";
        return;
    }

    uint32_t nc = queryClusterIDs.size();
    file.write(reinterpret_cast<const char*>(&nc), sizeof(uint32_t));

    for (uint32_t i = 0; i < nc; i++) {
        uint32_t size = queryClusterIDs[i].size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(queryClusterIDs[i].data()), size * sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(queryClusterIDIdx[i].data()), size * sizeof(uint32_t));

        uint32_t allVectorIDSize = allVectorIDs[i].size();
        if (!queryClusterIDIdx[i].empty()) {
            assert(queryClusterIDIdx[i].back() == allVectorIDSize);
        } else {
            assert(allVectorIDSize == 0);
        }
        file.write(reinterpret_cast<const char*>(&allVectorIDSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(allVectorIDs[i].data()), allVectorIDSize * sizeof(uint32_t));

        file.write(reinterpret_cast<const char*>(AlphaCentroidDists[i].data()), size * sizeof(float));
        file.write(reinterpret_cast<const char*>(AlphaCentroid[i].data()), size * sizeof(float));
    }
    file.close();
}


void loadNeighborList(
    std::vector<std::vector<uint32_t>>& queryClusterIDs,
    std::vector<std::vector<uint32_t>>& queryClusterIDIdx,
    std::vector<std::vector<uint32_t>>& allVectorIDs,
    std::vector<std::vector<float>>& AlphaCentroidDists,
    std::vector<std::vector<float>>& AlphaCentroid,
    const std::string& path = PathNeighborList) {

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << "\n";
        return;
    }

    uint32_t nc;
    file.read(reinterpret_cast<char*>(&nc), sizeof(uint32_t));

    queryClusterIDs.resize(nc);
    queryClusterIDIdx.resize(nc);
    allVectorIDs.resize(nc);
    AlphaCentroidDists.resize(nc);
    AlphaCentroid.resize(nc);

    for (uint32_t i = 0; i < nc; i++) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

        queryClusterIDs[i].resize(size);
        queryClusterIDIdx[i].resize(size);
        AlphaCentroidDists[i].resize(size);
        AlphaCentroid[i].resize(size);

        file.read(reinterpret_cast<char*>(queryClusterIDs[i].data()), size * sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(queryClusterIDIdx[i].data()), size * sizeof(uint32_t));

        uint32_t allVectorIDSize;
        file.read(reinterpret_cast<char*>(&allVectorIDSize), sizeof(uint32_t));
        allVectorIDs[i].resize(allVectorIDSize);
        file.read(reinterpret_cast<char*>(allVectorIDs[i].data()), allVectorIDSize * sizeof(uint32_t));

        if (!queryClusterIDIdx[i].empty()) {
            assert(queryClusterIDIdx[i].back() == allVectorIDSize);
        } else {
            assert(allVectorIDSize == 0);
        }
        file.read(reinterpret_cast<char*>(AlphaCentroidDists[i].data()), size * sizeof(float));
        file.read(reinterpret_cast<char*>(AlphaCentroid[i].data()), size * sizeof(float));
    }

    file.close();
}


void computeCentroidNorm() {

    performance_recorder recorder("computeCentroidsNorm");
    std::vector<float> CentroidsNorm(nc);
    std::vector<float> Centroids(nc * Dimension);
    std::cout << "Computing the distance between centroid vectors \n";
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {
        std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";
        exit(1);
    }
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    faiss::fvec_norms_L2sqr(CentroidsNorm.data(), Centroids.data(), Dimension, nc);
    std::ofstream CentroidDistOutputStream(PathCentroidNorm, std::ios::binary);
    if (!CentroidDistOutputStream) {
        std::cerr << "Error opening PathCentroidNorm file: " << PathCentroidNorm << "\n";
        exit(1);
    }
    CentroidDistOutputStream.write((char*) CentroidsNorm.data(), nc * sizeof(float));
    CentroidDistOutputStream.close();
    recorder.print_performance("computeCentroidsNorm");
}