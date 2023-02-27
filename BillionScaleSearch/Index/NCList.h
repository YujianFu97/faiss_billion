#include "BIndex.h"

struct NCListIndex: BIndex
{
    uint64_t NumQuantUnits;
    uint32_t NClusterNeighbors;
    size_t MaxNCList;
    std::vector<std::vector<uint32_t>> ClusterNeighborID;
    std::vector<std::vector<float>> ClusterNeighborDist;
    std::vector<std::vector<float>> SqrtClusterNeighborDist;

    std::vector<std::vector<uint32_t>> NeighborList;

    // Initialize this for NClistIndex search
    std::vector<float> QCDist;
    std::vector<uint32_t> QCID;
    std::vector<uint32_t> QueryNeighborList;

    explicit NCListIndex(const size_t Dimension, const size_t nb, const size_t nc, const size_t nt, const bool saving, const bool recording, const bool Retrain,
    const bool UseOPQ, const size_t M_PQ, const size_t CodeBits, size_t NClusterNeighbors, size_t NumQuantUnits, size_t MaxNCList);

    void NeighborInfo(std::string PathCentroidNeighborID, std::string PathCentroidNeighborDist);
    void BuildNList(std::string PathNeighborList, std::string PathBase, std::string PathBaseIDSeq, size_t NumBatch, size_t NumClusterBatch, float Beta);
    void Search(size_t K, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem, size_t AccuStopItem);
    std::string Eval(std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, size_t NumRecall, size_t NumPara, size_t * RecallK, size_t * MaxItem, size_t * EfSearch, size_t * AccuStopItem);
};