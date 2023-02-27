#ifndef _BIndex_H
#define _BIndex_H

#include <string>
#include <unistd.h>
#include "../hnswlib/hnswalg.h"
#include "../../faiss/VectorTransform.h"
#include "../../faiss/index_io.h"
#include "../../faiss/IndexFlat.h"
#include "../complib/include/codecfactory.h"
#include "../NCSearch/NCSearch.h"


// Change this type for different datasets
//typedef uint8_t DataType;

struct BoundQueue
{
    float Dist;
    uint32_t NeighborID;
    uint32_t OtherID;
    BoundQueue(float Dist, uint32_t NeighborID, uint32_t OtherID): Dist(Dist), NeighborID(NeighborID), OtherID(OtherID){}
    bool operator< (const BoundQueue& Node) const{
        return Dist > Node.Dist;
    }
};

struct BIndex
{
    public:
    size_t Dimension;
    size_t nb;
    size_t nc;
    size_t nt;

    bool Saving;
    bool Recording;
    bool Retrain;
    bool UseOPQ;
    memory_recorder Mrecorder;
    time_recorder Trecorder;

    size_t M_PQ;
    size_t CodeBits;

    std::vector<float> CNorms;
    std::vector<float> PQTable;
    std::vector<float> BaseNorms;
    std::vector<std::vector<uint32_t>> BaseIds;
    std::vector<uint8_t> BaseCodes;
    hnswlib::HierarchicalNSW * CentroidHNSW;
    faiss::ProductQuantizer * PQ;
    faiss::LinearTransform * OPQ;

    // For Search
    std::vector<float> QCDist;
    std::vector<int64_t> QCID;

    size_t VisitedItem;
    uint32_t AccuItem;

    explicit BIndex(const size_t Dimension, const size_t nb, const size_t nc, const size_t nt, const bool saving, const bool recording, const bool Retrain,
    const bool UseOPQ, const size_t M_PQ, const size_t CodeBits);

    float TestDistBound(size_t K, size_t ngt, size_t nq, std::string PathQuery, std::string PathGt, std::string PathBase);
    float TrainCentroids(size_t CenTrainSize, std::string PathLearn, std::string PathCentroid, std::string PathCentroidNorm, bool Optimize = true, bool UseGraph = false, size_t Nlevel = 1, size_t OptSize = 10, float Lambda = 50);

    uint32_t LearnCentroidsINI(size_t CenTrainSize, size_t nq, bool Optimize, size_t MinNC, size_t MaxNC, float TargetRecall, float MaxCandidateSize, size_t RecallK, size_t MaxFailureTimes, float CheckProp, size_t LowerBound,
        size_t NCBatch, size_t PQ_M, size_t nbits, size_t PQ_TrainSize, size_t GraphM, size_t GraphEf, size_t NLevel, size_t OptSize, float Lambda, size_t ngt,
        std::string Path_folder, std::string Path_GT, std::string Path_base, std::string Path_query, std::string Path_learn, 
        std::ofstream &RecordFile
    );

    void BuildGraph(size_t M, size_t efConstruction, std::string PathGraphInfo, std::string PathGraphEdge, std::string PathCentroid);
    void AssignVector(size_t N, float * BaseData, uint32_t * DataID);

    void DoOPQ(size_t N, float * Dataset);
    float QuantizationError(faiss::ProductQuantizer * PQ, float * TestVector, size_t N);


    void TrainQuantizer(size_t PQTrainSize, std::string PathLearn, std::string PathPQ, std::string PathOPQ);
    

    void AssignBaseset(size_t NumBatch, size_t BacthSize, std::string PathBase, std::string PathBaseIDInv, std::string PathBaseIDSeq);
    std::string ClusterFeature();
    

    void QuantizeBaseset(size_t NumBatch, size_t BacthSize, std::string PathBase, std::string PathBaseIDSeq, std::string PathBaseCode, std::string PathBaseNorm, std::string PathOPQCentroids = "");
    void QuantizeVector(size_t N, float * BaseData, uint32_t * BaseID, uint8_t * BaseCode, float * BaseNorm, float * OPQCentroids);


    std::string QuantizationFeature(std::string PathBase, std::string PathBaseIDSeq);
    uint32_t Search(size_t K, float * Query, int64_t * QueryIds, float * Dists, size_t EfSearch, size_t MaxItem);
    uint32_t SearchMulti(size_t K, size_t nq, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem);
    std::string Eval(std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, size_t NumRecall, size_t NumPara, size_t * RecallK, size_t * MaxItem, size_t * EfSearch);
    std::string NeighborCost(size_t Scale, std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, size_t EfSearch, float TargetRecall);
};





#endif