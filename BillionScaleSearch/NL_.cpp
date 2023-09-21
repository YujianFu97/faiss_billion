#include "NL.h"
// Define rerun flags
bool rerunCentroids = false;
bool rerunComputeNN = false;
bool rerunComputeNL = true;
bool prunevectors = false;

struct cmp{bool operator ()( std::pair<uint32_t, float> &a, std::pair<uint32_t, float> &b){return a.second < b.second;}};

void computeCentroids() {
    if (!rerunCentroids && exists(PathCentroid))
        return;

    performance_recorder recorder = performance_recorder("computeCentroids");
    std::vector<float> Centroids(nc * Dimension);
    std::vector<float> TrainSet(nt * Dimension);
    std::ifstream TrainInputStream(PathTrain, std::ios::binary);
    if (!TrainInputStream) {
        std::cerr << "Error opening PathBase file: " << PathTrain << "\n";
        exit(1);
    }
    readXvecFvec<DType>(TrainInputStream, TrainSet.data(), Dimension, nt, true, true);
    faisskmeansClustering(TrainSet.data(), Centroids.data());

    std::ofstream CentroidsOutputStream(PathCentroid, std::ios::binary);
    if (!CentroidsOutputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}

    uint32_t D = Dimension;
    for (size_t i = 0; i < nc; i++) {
        CentroidsOutputStream.write(reinterpret_cast<char*>(&D), sizeof(uint32_t));
        CentroidsOutputStream.write(reinterpret_cast<char*>(Centroids.data() + i * Dimension), Dimension * sizeof(float));
    }
}

void computeCenHNSWGraph() {
    if (!rerunCentroids && !rerunComputeNN && exists(PathCenGraphIndex))
        return;

    performance_recorder recorder("computeHNSWGraph");
    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * HNSWGraph;
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    HNSWGraph = new hnswlib::HierarchicalNSW<float>(&l2space, nc, M, EfConstruction);
    HNSWGraph->addPoint(Centroids.data(), 0);
    #pragma omp parallel for
    for (size_t i = 1; i < nc; i++){
        HNSWGraph->addPoint(Centroids.data() + i * Dimension, i);
    }
    HNSWGraph->saveIndex(PathCenGraphIndex);
    delete HNSWGraph; // Free the allocated memory
}

/*
Assign the vectors and save computed distance
*/
void computeDatasetNeighbor() {
    performance_recorder recorder("computeDatasetNeighbor");
    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    if (rerunCentroids || !exists(PathBaseNeighborID) || !exists(PathBaseNeighborDist)){
        assert(Assign_num_batch > 0);
        size_t Batch_size = nb / Assign_num_batch;
        std::ofstream BasesetNeighborIDOutputStream(PathBaseNeighborID, std::ios::binary);
        std::ofstream BasesetNeighborDistOutputStream(PathBaseNeighborDist, std::ios::binary);
        std::ifstream BasesetInputStream(PathBase, std::ios::binary);
        if (!BasesetNeighborIDOutputStream || !BasesetNeighborDistOutputStream || !BasesetInputStream) {
            std::cerr << "Error opening PathBaseNeighborID PathBaseNeighborDist PathBase file: " << "\n";exit(1);}

        std::vector<float> BaseBatch(Batch_size * Dimension);
        std::vector<uint32_t> BaseIDBatch(Batch_size * NeighborNum);
        std::vector<float> BaseDistBatch(Batch_size * NeighborNum);

        for (size_t i = 0; i < Assign_num_batch; i++){
            readXvecFvec<DType>(BasesetInputStream, BaseBatch.data(), Dimension, Batch_size, true, true);

            #pragma omp parallel for
            for (size_t j = 0; j < Batch_size; j++){
                auto result = CenGraph->searchKnn(BaseBatch.data() + j * Dimension, NeighborNum);
                for (size_t temp = 0; temp < NeighborNum; temp++){
                    BaseIDBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().second;
                    BaseDistBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().first;
                    result.pop();
                }
            }
            BasesetNeighborIDOutputStream.write((char *) BaseIDBatch.data(), Batch_size * NeighborNum * sizeof(uint32_t));
            BasesetNeighborDistOutputStream.write((char *) BaseDistBatch.data(), Batch_size * NeighborNum * sizeof(float));
        }
        BasesetNeighborIDOutputStream.close();
        BasesetNeighborDistOutputStream.close();
    }

    recorder.print_performance("computeBasesetNeighbor");

    // Compute the neighbor cluster ID for trainset
    if (rerunCentroids || !exists(PathTrainNeighborID) || !exists(PathTrainNeighborDist)){
        std::ofstream TrainsetNeighborIDOutputStream(PathTrainNeighborID, std::ios::binary);
        std::ofstream TrainsetNeighborDistOutputStream(PathTrainNeighborDist, std::ios::binary);
        std::ifstream TrainsetInputStream(PathTrain, std::ios::binary);
        if (!TrainsetNeighborIDOutputStream || !TrainsetNeighborDistOutputStream || !TrainsetInputStream) {
            std::cerr << "Error opening PathTrainNeighborID PathTrainNeighborDist PathTrain file: " << "\n";exit(1);}

        std::vector<float> Trainset(nt * Dimension);
        std::vector<uint32_t> TrainsetID(nt * NeighborNum);
        std::vector<float> TrainsetDist(nt * NeighborNum);
        readXvecFvec<DType>(TrainsetInputStream, Trainset.data(), Dimension, nt, true, true);

        #pragma omp parallel for
        for (size_t i = 0; i < nt; i++){
            auto result = CenGraph->searchKnn(Trainset.data() + i * Dimension, NeighborNum);
            for (size_t temp = 0; temp < NeighborNum; temp++){
                TrainsetID[i * NeighborNum + NeighborNum - temp - 1] = result.top().second;
                TrainsetDist[i * NeighborNum + NeighborNum - temp - 1] = result.top().first;
                result.pop();
            }
        }
        TrainsetNeighborIDOutputStream.write((char *) TrainsetID.data(), nt * NeighborNum * sizeof(uint32_t));
        TrainsetNeighborDistOutputStream.write((char *)TrainsetDist.data(), nt * NeighborNum * sizeof(float));
        TrainsetNeighborIDOutputStream.close();
        TrainsetNeighborDistOutputStream.close();
    }
    delete CenGraph;
    recorder.print_performance("computeTrainsetNeighbor");
}

void computeNN() {
    if (exists(PathTrainsetNN) && (!rerunComputeNN)) 
        return;
    performance_recorder recorder("computeNN");
    PrepareFolder(PathSubGraphFolder.c_str());

    #pragma omp parallel for
    for (size_t i = 0; i < Graph_num_batch; i++) {
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        if (exists(PathSubGraphInfo) && exists(PathSubGraphEdge)){
            continue;
        }
        std::ifstream BasesetInputStream(PathBase, std::ios::binary);
        if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}

        std::vector<float> BaseBatch(Graph_batch_size * Dimension);
        BasesetInputStream.seekg(i * Graph_batch_size * (Dimension * sizeof(DType) + sizeof(uint32_t)), std::ios::beg);
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);
        BasesetInputStream.close();
        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(Dimension, Graph_batch_size, M, 2 * M, EfConstruction);
        for (size_t j = 0; j < Graph_batch_size; j++){
            SubGraph->addPoint(BaseBatch.data() + j * Dimension);
        }
        SubGraph->SaveInfo(PathSubGraphInfo);
        SubGraph->SaveEdges(PathSubGraphEdge);
        delete SubGraph;
    }
    recorder.print_performance("buildSubGraphs");

    // Search NNs
    std::vector<float> TrainSet(nt * Dimension);
    std::ifstream TrainInputStream(PathTrain, std::ios::binary);
    if (!TrainInputStream) {std::cerr << "Error opening PathTrain file: " << PathTrain << "\n";exit(1);}
    readXvecFvec<DType>(TrainInputStream, TrainSet.data(), Dimension, nt, true, true);
    TrainInputStream.close();
    std::vector<std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, cmp>> NNQueueList(nt);

    std::ifstream BasesetInputStream(PathBase, std::ios::binary);
    if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::vector<float> BaseBatch(Graph_batch_size * Dimension);

    for (size_t i = 0; i < Graph_num_batch; i++){
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        assert(exists(PathSubGraphInfo) && exists(PathSubGraphEdge));
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);        

        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, BaseBatch);
        SubGraph->efSearch = EfNN;

        uint32_t StartIndex = i * Graph_batch_size;
        #pragma omp parallel for
        for (size_t j = 0; j < nt; j++){
            auto result = SubGraph->searchKnn(TrainSet.data() + j * Dimension, SearchK);
            for (size_t temp = 0; temp < SearchK; temp++){
                if (NNQueueList[j].size() < SearchK){
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                else if (result.top().first < NNQueueList[j].top().second){
                    NNQueueList[j].pop();
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                result.pop();
            }
        }
        delete(SubGraph);
    }
    // Write Results
    std::ofstream BasesetNNOutputStream(PathTrainsetNN, std::ios::binary);
    if (!BasesetNNOutputStream) {std::cerr << "Error opening PathTrainsetNN file: " << PathTrainsetNN << "\n";exit(1);}
    for (size_t i = 0; i < nt; i++){
        std::vector<uint32_t> EachVecNN(SearchK);
        for (size_t j = 0; j < SearchK; j++)
        {
            EachVecNN[SearchK - j - 1] = NNQueueList[i].top().first;
            NNQueueList[i].pop();
        }
        BasesetNNOutputStream.write((char *) EachVecNN.data(), SearchK * sizeof(uint32_t));
    }
    BasesetNNOutputStream.close();
}

void testNNGraphQuality(size_t recallK = 20){
    assert(recallK <= ngt);
    performance_recorder recorder("testNNGraphQuality");
    std::vector<float> QuerySet(nq * Dimension);
    std::vector<uint32_t> GTSet(nq * ngt);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvecFvec<DType>(QueryInput, QuerySet.data(), Dimension, nq, true, true);
    readXvec<uint32_t>(GTInput, GTSet.data(), ngt, nq, true, true);

    std::vector<std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, cmp>> NNQueueList(nq);
    std::ifstream BasesetInputStream(PathBase, std::ios::binary);
    if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::vector<float> BaseBatch(Graph_batch_size * Dimension);
    for (size_t i = 0; i < Graph_num_batch; i++){
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        assert(exists(PathSubGraphInfo) && exists(PathSubGraphEdge));
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);        

        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, BaseBatch);
        SubGraph->efSearch = EfNN;

        uint32_t StartIndex = i * Graph_batch_size;
        #pragma omp parallel for
        for (size_t j = 0; j < nq; j++){
            auto result = SubGraph->searchKnn(QuerySet.data() + j * Dimension, recallK);
            for (size_t temp = 0; temp < recallK; temp++){
                if (NNQueueList[j].size() < recallK){
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                else if (result.top().first < NNQueueList[j].top().second){
                    NNQueueList[j].pop();
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                result.pop();
            }
        }
        delete(SubGraph);
    }
    float SumCorrect = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<uint32_t> gt;
        for (size_t j = 0; j < recallK; j++){gt.insert(GTSet[i * ngt + j]);}

        for (size_t j  = 0; j < recallK; j++){
            if (gt.count(NNQueueList[i].top().first) != 0){
                SumCorrect ++;
            }
        }
    }
    std::cout << "*****************************************************************************************************************\n"; 
    std::cout << "The overall recall@" << recallK << " for the query using this parameter setting: " << SumCorrect / (nq * recallK) << "\n";
    std::cout << "*****************************************************************************************************************\n"; 
}


void cmputeNNConflictListAndSave_(){
    // Determine the number of list size based on the conflict gt
    // Clustering the conflict gt and pruned gt
    // Decide the pruning process

    if (!rerunComputeNL && exists(PathBoundaryConflictMapVec))
        return;
    
    std::vector<uint32_t> TrainsetNN(nt * SearchK);
    std::ifstream TrainsetNNInputStream(PathTrainsetNN, std::ios::binary);
    if (!TrainsetNNInputStream) {std::cerr << "Error opening PathBase file: " << PathTrainsetNN << "\n";exit(1);}
    TrainsetNNInputStream.read(reinterpret_cast<char*>(TrainsetNN.data()), nt * SearchK * sizeof(uint32_t));
    TrainsetNNInputStream.close();
    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);
    std::vector<uint32_t> TrainsetAssignment(nt);
    loadNeighborAssignments(PathTrainNeighborID, nt, NeighborNum, TrainsetAssignment);

    // If we need to prune the vector, we need to store the cluster ID pair
    // Other wise, we just need to keep the central cluster ID
    std::vector<std::vector<uint32_t>> BCCluster;
    std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> BCClusterPrune;

    if (!prunevectors){
        BCCluster.resize(nc);
    }
    else{
        BCClusterPrune.resize(nc);
    }
    for (size_t i = 0; i < nt; i++){
        uint32_t QueryClusterID = TrainsetAssignment[i];
        for (size_t j = 0; j < NLClusterTargetK; j++){
            uint32_t NNVecID = TrainsetNN[i * SearchK + j];
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            if (NNClusterID != QueryClusterID){
                if (prunevectors){
                    BCClusterPrune[NNClusterID][QueryClusterID].insert(NNVecID);
                }
                else{
                    BCCluster[NNClusterID].emplace_back(NNVecID);
                }
            }
        }
    }

    if ()
}

void computeNNConflictListAndSave() {
    if (!rerunComputeNL && exists(PathBoundaryConflictMapVec) && exists(PathBoundaryConflictMapCluster))
        return;

    assert(exists(PathBaseNeighborID) && exists(PathTrainsetNN) && exists(PathTrainNeighborID));
    performance_recorder recorder("computeNNConflictListAndSave");
    std::vector<uint32_t> TrainsetNN(nt * SearchK);
    std::ifstream TrainsetNNInputStream(PathTrainsetNN, std::ios::binary);
    if (!TrainsetNNInputStream) {std::cerr << "Error opening PathBase file: " << PathTrainsetNN << "\n";exit(1);}
    TrainsetNNInputStream.read(reinterpret_cast<char*>(TrainsetNN.data()), nt * SearchK * sizeof(uint32_t));
    TrainsetNNInputStream.close();

    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);
    std::vector<uint32_t> TrainsetAssignment(nt);
    loadNeighborAssignments(PathTrainNeighborID, nt, NeighborNum, TrainsetAssignment);

    std::vector<float> TrainsetNeighborDist(nt * SearchK);
    std::ifstream TrainsetNeighborDistInputStream(PathTrainNeighborDist, std::ios::binary);
    TrainsetNeighborDistInputStream.read(reinterpret_cast<char*>(TrainsetNeighborDist.data()), nt * SearchK * sizeof(float));
    std::vector<uint32_t> TrainsetNeighborID(nt * SearchK);
    std::ifstream TrainsetNeighborIDInputStream(PathTrainNeighborID, std::ios::binary);
    TrainsetNeighborIDInputStream.read(reinterpret_cast<char*>(TrainsetNeighborID.data()), nt * SearchK * sizeof(float));
    std::vector<float> TrainSet(nt * Dimension);
    std::ifstream TrainsetInputStream(PathTrain, std::ios::binary);
    readXvec<DType> (TrainsetInputStream, TrainSet.data(), Dimension, nt, true, true);
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    readXvec<float> (CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    // The struct is VectorID -> QueryClusterID <-> Number of conflicts
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>> BoundaryConflictMapVec;
    // The struct is NNClusterID -> QueryClusterID -> VectorID
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>> BoundaryConflictMapCluster;

    for (size_t i = 0; i < nt; i++){
        uint32_t QueryClusterID = TrainsetAssignment[i];
        for (size_t j = 0; j < NLClusterTargetK; j++){
            uint32_t NNVecID = TrainsetNN[i * SearchK + j];
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            if (NNClusterID != QueryClusterID){
                auto & QueryClusterMap = BoundaryConflictMapVec[NNVecID];
                if (QueryClusterMap.find(QueryClusterID) == QueryClusterMap.end()){
                    QueryClusterMap[QueryClusterID] = 1;
                }
                else{
                    QueryClusterMap[QueryClusterID] ++;
                }
            }
        }
    }
    for (const auto &  NNID_pair : BoundaryConflictMapVec){
        uint32_t NNVecID = NNID_pair.first;
        uint32_t NNClusterID = BaseAssignment[NNVecID];
        const std::unordered_map<uint32_t, size_t> & QueryClusterMap = NNID_pair.second; 

        for (const auto & QueryClusterIt : QueryClusterMap){
            uint32_t QueryClusterID = QueryClusterIt.first;
            BoundaryConflictMapCluster[NNClusterID][QueryClusterID].emplace_back(NNVecID);
        }
    }

    // Parallelizing the outer loop
    #pragma omp parallel for
    for (size_t j = NLClusterTargetK; j < NLTargetK; j++) {

        for (size_t i = 0; i < nt; i++) {
            uint32_t QueryClusterID = TrainsetAssignment[i];
            uint32_t NNVecID = TrainsetNN[i * SearchK + j];
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            if (NNClusterID != QueryClusterID) {
                // Locks to protect shared resources

                if (BoundaryConflictMapCluster.find(NNClusterID) != BoundaryConflictMapCluster.end() &&
                    BoundaryConflictMapCluster[NNClusterID].find(QueryClusterID) != BoundaryConflictMapCluster[NNClusterID].end()) {
                    #pragma omp critical
                    {
                        if (BoundaryConflictMapVec[NNVecID].find(QueryClusterID) == BoundaryConflictMapVec[NNVecID].end()) {
                            BoundaryConflictMapVec[NNVecID][QueryClusterID] = 1;
                        } else {
                            BoundaryConflictMapVec[NNVecID][QueryClusterID]++;
                        }

                        BoundaryConflictMapCluster[NNClusterID][QueryClusterID].emplace_back(NNVecID);
                    }
                } else {
                    float MinQueryClusterDist = std::numeric_limits<float>::max();
                    uint32_t ResultQueryClusterID = nc;

                    if (BoundaryConflictMapCluster.find(NNClusterID) == BoundaryConflictMapCluster.end() || BoundaryConflictMapCluster[NNClusterID].size() < MinNeSTSize){
                        ResultQueryClusterID = QueryClusterID;
                    }
                    else{
                        for (auto ClusterIt = BoundaryConflictMapCluster[NNClusterID].begin(); ClusterIt != BoundaryConflictMapCluster[NNClusterID].end(); ClusterIt++) {
                            QueryClusterID = ClusterIt->first;
                            auto QueryClusterDist = faiss::fvec_L2sqr(TrainSet.data() + i * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);

                            if (QueryClusterDist < MinQueryClusterDist) {
                                ResultQueryClusterID = QueryClusterID;
                                MinQueryClusterDist = QueryClusterDist;
                            }
                        }
                    }

                    assert(ResultQueryClusterID < nc);
                    #pragma omp critical
                    {
                        if (BoundaryConflictMapVec[NNVecID].find(ResultQueryClusterID) == BoundaryConflictMapVec[NNVecID].end()) {
                            BoundaryConflictMapVec[NNVecID][ResultQueryClusterID] = 1;
                        } else {
                            BoundaryConflictMapVec[NNVecID][ResultQueryClusterID]++;
                        }
                        BoundaryConflictMapCluster[NNClusterID][ResultQueryClusterID].emplace_back(NNVecID);
                    }
                }
            }
        }
    }

    writeBoundaryConflictMapVec(BoundaryConflictMapVec);
    writeBoundaryConflictMapCluster(BoundaryConflictMapCluster);

    size_t NumNestList = 0;
    for (auto NNit = BoundaryConflictMapCluster.begin(); NNit != BoundaryConflictMapCluster.end(); NNit ++){
        NumNestList += NNit->second.size();
        std::cout << NNit->second.size() << " " ;
    }
    std::cout << "The total number of NeST list is: " << NumNestList << "\n";

    /*****************************/
    std::vector<float> QuerySet(nq * Dimension);
    std::vector<uint32_t> GtSet(nq * ngt);    
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GtSet.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvec<float>(QueryInput, QuerySet.data(), Dimension, nq, true, true);
    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    // Find whether the conflict gt cluster pair exists in the found conflict pair
    auto ef = EfSearch[0];
    auto recallk = RecallK[0];
    std::vector<uint32_t> ResultID(nq * ef);
    std::vector<float> ResultDist(nq * ef);
    size_t NumConflictCluster = 0;
    size_t NumFoundCC = 0;
    size_t NumConflictGt = 0;
    size_t NumFoundGt = 0;
    for (size_t i = 0; i < nq; i++){
        auto Result = CenGraph->searchKnn(QuerySet.data() + i * Dimension, EfSearch[0]);
        for (size_t j = 0; j < ef; j++){
            ResultID[i * ef + ef - j - 1] = Result.top().second;
            ResultDist[i * ef + ef - j - 1] = Result.top().first;
            Result.pop();
        }
        auto CentralID = ResultID[i * ef];
        for (size_t j = 0; j < recallk; j++){
            auto GtClusterID = BaseAssignment[GtSet[i * ngt + j]];
            if (GtClusterID != CentralID){
                NumConflictCluster++;
                auto GtID = GtSet[i * ngt + j];
                if (BoundaryConflictMapVec.find(GtID) != BoundaryConflictMapVec.end()){
                    NumFoundGt ++;
                }
            }

            if (BoundaryConflictMapCluster.find(CentralID) != BoundaryConflictMapCluster.end()){
                if (BoundaryConflictMapCluster[CentralID].find(GtClusterID) != BoundaryConflictMapCluster[CentralID].end()){
                    NumFoundCC++;
                }
            }
        }
    }
    std::cout << "Total number of conflict clusters: " << NumConflictCluster << " Number of found conflict clusters: " << NumFoundCC << " Number of NumFoundGt: " << NumFoundGt << "\n";
     std::cout << "\n";

    /*****************************/
}

/*
Compute the distance threshold for the boundary between clusters
*/
void computeConflictDistListAndSave(){
    if (!rerunComputeNL && exists(PathBoundaryConflictMapDist))
        return;

    performance_recorder recorder("computeConflictDistListAndSave");

    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>> BoundaryConflictMapDist;
    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>> BoundaryConflictMapVec;
    loadBoundaryConflictMapVec(BoundaryConflictMapVec);

    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    std::ifstream BaseInputStream (PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::ifstream BaseNeighborIDInputStream(PathBaseNeighborID, std::ios::binary);
    if (!BaseNeighborIDInputStream) {std::cerr << "Error opening PathBaseNeighborID file: " << PathBaseNeighborID << "\n";exit(1);}
    std::ifstream BaseNeighborDistInputStream(PathBaseNeighborDist, std::ios::binary);
    if (!BaseNeighborDistInputStream) {std::cerr << "Error opening PathBaseNeighborDist file: " << PathBaseNeighborDist << "\n";exit(1);}

    std::vector<float> Centroids(nc * Dimension);
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);
    CentroidsInputStream.close();

    std::vector<float> BaseBatch(Assign_batch_size * Dimension);
    std::vector<uint32_t> BaseNeighborIDBatch(Assign_batch_size * NeighborNum);
    std::vector<float> BaseNeighborDistBatch(Assign_batch_size * NeighborNum);

    for (uint32_t i = 0; i < Assign_num_batch; i++) {
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, false);
        BaseNeighborIDInputStream.read(reinterpret_cast<char*>(BaseNeighborIDBatch.data()), Assign_batch_size * NeighborNum * sizeof(uint32_t));
        BaseNeighborDistInputStream.read(reinterpret_cast<char*>(BaseNeighborDistBatch.data()), Assign_batch_size * NeighborNum * sizeof(float));

        #pragma omp parallel for
        for (uint32_t j = 0; j < Assign_batch_size; j++) {
            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            auto VecQueryIDit = BoundaryConflictMapVec.find(NNVecID);

            if (VecQueryIDit != BoundaryConflictMapVec.end()) {
                // Key found, so iterate over the associated vector
                for (const auto& QueryClusterIt : VecQueryIDit->second) {
                    uint32_t QueryClusterID = QueryClusterIt.first;

                    auto distances = computeDistNLBoundary(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, QueryClusterID);
                    float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                    float QueryClusterDist = distances.first;
                    float DistNLBoundary = distances.second;
                    // Critical section to ensure safe update
                    #pragma omp critical
                    {
                        auto ClusterIDPair = std::make_pair(NNClusterID, QueryClusterID);
                        if (BoundaryConflictMapDist.find(ClusterIDPair) == BoundaryConflictMapDist.end()){
                            BoundaryConflictMapDist[ClusterIDPair] = std::make_pair(1, std::make_tuple(DistNLBoundary, QueryClusterDist, NNClusterDist, (NNClusterDist / QueryClusterDist)));
                        }
                        else{
                            updateBoundaryConflictMapDist(BoundaryConflictMapDist, ClusterIDPair, NNClusterDist, QueryClusterDist, DistNLBoundary);
                        }
                    }
                }
            }
        }
    }
    for (auto it = BoundaryConflictMapDist.begin(); it != BoundaryConflictMapDist.end(); it++){
        uint32_t ListSize = it->second.first;
        std::get<0>(it->second.second) /= ListSize;
        std::get<1>(it->second.second) /= ListSize;
        std::get<2>(it->second.second) /= ListSize;
        std::get<3>(it->second.second) /= ListSize;
    }
    BaseInputStream.close();
    BaseNeighborIDInputStream.close();
    BaseNeighborDistInputStream.close();
    writeBoundaryConflictMapDist(BoundaryConflictMapDist);

    std::cout << "The total number of NeST list size is: " << BoundaryConflictMapDist.size() << "\n";
}

void computeBoundaryPruneAndSave() {
    if (!rerunComputeNL && exists(PathBoundaryConflictMapVec) && exists(PathBoundaryConflictMapCluster))
        return;
    performance_recorder recorder("computeBoundaryPruneAndSave");

    size_t NumNLVec = 0;
    size_t NumNonNLVec = 0;
    // The struct is VectorID -> QueryClusterID <-> Number of conflicts
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, size_t>> BoundaryConflictMapVec;
    loadBoundaryConflictMapVec(BoundaryConflictMapVec);
    // The struct is NNClusterID -> QueryClusterID -> VectorID -> Number of conflicts
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>> BoundaryConflictMapCluster;
    loadBoundaryConflictMapCluster(BoundaryConflictMapCluster);

    std::vector<uint32_t> BoundaryConflictVecResult(nb, nc);
    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> BoundaryConflictClusterResult(nc);
    std::vector<std::vector<uint32_t>> NonBCLists(nc);

    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>> BoundaryConflictMapDist;
    if (prunevectors){
        loadBoundaryConflictMapDist(BoundaryConflictMapDist);
    }

    std::ifstream BaseNeighborIDInputStream(PathBaseNeighborID, std::ios::binary);
    if (!BaseNeighborIDInputStream) {std::cerr << "Error opening PathBaseNeighborID file: " << PathBaseNeighborID << "\n";exit(1);}
    std::ifstream BaseNeighborDistInputStream(PathBaseNeighborDist, std::ios::binary);
    if (!BaseNeighborDistInputStream) {std::cerr << "Error opening PathBaseNeighborDist file: " << PathBaseNeighborDist << "\n";exit(1);}
    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}

    std::vector<float> BaseBatch(Assign_batch_size * Dimension);
    std::vector<uint32_t> BaseNeighborIDBatch(Assign_batch_size * NeighborNum);
    std::vector<float> BaseNeighborDistBatch(Assign_batch_size * NeighborNum);

    // Ensure each conflict vector only exists in one neighbor list in both boundary conflict map
    for (size_t i = 0; i < Assign_num_batch; i++){
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, true);
        BaseNeighborIDInputStream.read((char *)BaseNeighborIDBatch.data(), Assign_batch_size * NeighborNum * sizeof(uint32_t));
        BaseNeighborDistInputStream.read((char *) BaseNeighborDistBatch.data(), Assign_batch_size * NeighborNum * sizeof(float));

        #pragma omp parallel for
        for (size_t j = 0; j < Assign_batch_size; j++) {
            bool BoundaryConflictFlag = false;

            uint32_t NNVecID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseNeighborIDBatch[j * NeighborNum];

            // This is a original conflict vector, it will be only assigned to the existing conflict pair
            if (BoundaryConflictMapVec.find(NNVecID) != BoundaryConflictMapVec.end()){
                BoundaryConflictFlag = true;
                size_t MaxBoundaryConflictNum = 0;
                std::vector<uint32_t> ResultQueryClusters;

                /*
                float MinQueryClusterDist = std::numeric_limits<float>::max();
                uint32_t ResultQueryClusterID = 0;
                ResultQueryClusters.resize(1);
                for (auto it = BoundaryConflictMapCluster[NNClusterID].begin(); it != BoundaryConflictClusterResult[NNClusterID].end(); it++){
                    //float QueryClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, Centroids.data() + it->first * Dimension, Dimension);
                    float QueryClusterDist = computeQueryClusterDist(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, it->first);
                    if (QueryClusterDist < MinQueryClusterDist){
                        MinQueryClusterDist = QueryClusterDist;
                        ResultQueryClusterID = it->first;
                        ResultQueryClusters[0] = it->first;
                    }
                }
                */

                for (auto it = BoundaryConflictMapVec[NNVecID].begin(); it != BoundaryConflictMapVec[NNVecID].end(); it ++){

                    if (it->second > MaxBoundaryConflictNum){
                        MaxBoundaryConflictNum = it->second;
                        ResultQueryClusters.clear();
                        ResultQueryClusters.push_back(it->first);
                    }
                    else if(it->second == MaxBoundaryConflictNum){
                        ResultQueryClusters.push_back(it->first);
                    }
                    ResultQueryClusters.push_back(it->first);
                }

                #pragma omp critical
                {
                    if (ResultQueryClusters.size() > 1) {
                        float MinQueryClusterDist = std::numeric_limits<float>::max();
                        uint32_t ResultQueryClusterID = 0;
                        for (uint32_t QueryClusterID : ResultQueryClusters){
                            float QueryClusterDist = computeQueryClusterDist(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, QueryClusterID);
                            if (QueryClusterDist < MinQueryClusterDist){
                                MinQueryClusterDist = QueryClusterDist;
                                ResultQueryClusterID = QueryClusterID;
                            }
                        }
                        BoundaryConflictVecResult[NNVecID] = ResultQueryClusterID;
                        BoundaryConflictClusterResult[NNClusterID][ResultQueryClusterID].emplace_back(NNVecID);
                    }
                    else {
                        BoundaryConflictVecResult[NNVecID] = ResultQueryClusters[0];
                        BoundaryConflictClusterResult[NNClusterID][ResultQueryClusters[0]].emplace_back(NNVecID);
                    }
                }
            }
            // This is not a original conflict vector, check all conflict boundaries in this NNCluster
            else if (prunevectors) {
                // There exists conflict boundary in this NN cluster, iterate over all query clusters
                // Find the query cluster that vector holds minimal distance to the query cluster centroid
                if (BoundaryConflictMapCluster.find(NNClusterID) != BoundaryConflictMapCluster.end()){
                    float MinQueryClusterDist = std::numeric_limits<float>::max();
                    uint32_t ResultQueryClusterID = nc;

                    for (auto it = BoundaryConflictMapCluster[NNClusterID].begin(); it != BoundaryConflictMapCluster[NNClusterID].end(); it++){
                        uint32_t QueryClusterID = it->first;
                        auto distances = computeDistNLBoundary(BaseNeighborDistBatch, BaseNeighborIDBatch, BaseBatch, Centroids, j, NNClusterID, QueryClusterID);

                        float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                        float QueryClusterDist = distances.first;
                        float DistNLBoundary = distances.second;

                        // Prune with threshold on distance to boundary and query cluster
                        // Can change this condition to different settings
                        auto condition = BoundaryConflictMapDist[std::make_pair(NNClusterID, QueryClusterID)].second;
                        if (DistNLBoundary < std::get<0>(condition) && QueryClusterDist < std::get<1>(condition)
                        && NNClusterDist > std::get<2>(condition)  && NNClusterDist / QueryClusterDist > std::get<3>(condition)){
                            if (QueryClusterDist < MinQueryClusterDist){
                                MinQueryClusterDist = QueryClusterDist;
                                ResultQueryClusterID = QueryClusterID;
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        if (ResultQueryClusterID < nc) {
                            BoundaryConflictFlag = true;
                            BoundaryConflictVecResult[NNVecID] = ResultQueryClusterID;
                            BoundaryConflictClusterResult[NNClusterID][ResultQueryClusterID].emplace_back(NNVecID);
                        }
                    }
                }
            }
            #pragma omp critical
            {
                if (!BoundaryConflictFlag){
                    NonBCLists[NNClusterID].emplace_back(NNVecID);
                    NumNonNLVec ++;
                }
                else{
                    NumNLVec ++;
                }
            }
        }
    }
    writeBoundaryConflictVecResult(BoundaryConflictVecResult);
    writeBoundaryConflictClusterResult(BoundaryConflictClusterResult);
    writeNonBCLists(NonBCLists);

    BaseNeighborIDInputStream.close();
    BaseNeighborDistInputStream.close();
    BaseInputStream.close();

    std::cout << "*****************************************************************************************************************\n";
    std::cout << "Neighbor list size conclusion: Number of vectors in NL: " << NumNLVec << " Number of vectors not in NL: " << NumNonNLVec << "\n";
    std::cout << "*****************************************************************************************************************\n";

    size_t NumNestList = 0;
    for (auto NNit = BoundaryConflictClusterResult.begin(); NNit != BoundaryConflictClusterResult.end(); NNit ++){
        NumNestList += (*NNit).size();
    }
    std::cout << "The total number of NeST list is: " << NumNestList << "\n";
}


void computeConflictListCentroids(){
    if (!rerunComputeNL && exists(PathBoundaryConflictMapCentroid))
       return;
    performance_recorder recorder("computeConflictListCentroids");
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    std::vector<uint32_t> BoundaryConflictVecResult;
    loadBoundaryConflictVecResult(BoundaryConflictVecResult);

    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> BoundaryConflictClusterResult;
    loadBoundaryConflictClusterResult(BoundaryConflictClusterResult);

    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);

    // Format: neighbor list centroid: [i, j], central list centroid: [i, i], [i, j + nc], central list with no neighbor list: [i, i]
    std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>> BoundaryConflictCentroidAlpha(nc);

    // The neighbor list size can be found in clustermp
    std::vector<uint32_t> CentralListSize(nc, 0);
    std::ifstream BaseInputStream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::vector<float> BaseBatch(Assign_batch_size * Dimension);
    for (size_t i = 0; i < Assign_num_batch; i++){
        readXvecFvec<DType>(BaseInputStream, BaseBatch.data(), Dimension, Assign_batch_size, true, true);
        for (size_t j = 0; j < Assign_batch_size; j++){
            uint32_t VectorID = i * Assign_batch_size + j;
            uint32_t NNClusterID = BaseAssignment[VectorID];
            uint32_t QueryClusterID;

            if (BoundaryConflictVecResult[VectorID] < nc){
                QueryClusterID = BoundaryConflictVecResult[VectorID];
            }
            else{
                QueryClusterID = NNClusterID;
                CentralListSize[NNClusterID] ++;
            }

            auto & CentroidItem = BoundaryConflictCentroidAlpha[NNClusterID][QueryClusterID];
            if (CentroidItem.second.size() < Dimension){
                CentroidItem = std::make_pair(std::make_pair(0.0f, 0.0f), std::vector<float> (Dimension, 0));
            }

            faiss::fvec_add(Dimension, CentroidItem.second.data(), BaseBatch.data() + j * Dimension, CentroidItem.second.data());
        }
    }

    for (uint32_t i = 0; i < nc; i++){
        if (BoundaryConflictClusterResult[i].size() > 0){
        // There exists neighbor list in i-th cluster
            // We are handling all neighbor conflict lists
            for (auto QueryIt = BoundaryConflictClusterResult[i].begin(); QueryIt != BoundaryConflictClusterResult[i].end(); QueryIt ++){
                uint32_t QueryClusterID = QueryIt->first;
                assert(QueryClusterID != i);

                size_t ListSize = QueryIt->second.size();
                auto &BoundaryConflictCentroid = BoundaryConflictCentroidAlpha[i][QueryClusterID];
                assert(BoundaryConflictCentroid.second.size() == Dimension);
                for (size_t d = 0; d < Dimension; d++){BoundaryConflictCentroid.second[d] /= ListSize;}

                float DistCenNNCen = faiss::fvec_L2sqr(BoundaryConflictCentroid.second.data(), Centroids.data() + i * Dimension, Dimension);
                float DistCenTarCen = faiss::fvec_L2sqr(BoundaryConflictCentroid.second.data(), Centroids.data() + QueryClusterID * Dimension, Dimension);
                float DistTarCenNNCen = faiss::fvec_L2sqr(Centroids.data() + i * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);
                float CosNLNNTarget = (DistCenNNCen + DistTarCenNNCen - DistCenTarCen) / (2 * sqrt(DistCenNNCen * DistTarCenNNCen));
                float Alpha = sqrt(DistCenNNCen) * CosNLNNTarget / sqrt(DistTarCenNNCen);
                BoundaryConflictCentroid.first.first = Alpha;
                BoundaryConflictCentroid.first.second = Alpha * (Alpha - 1) * DistTarCenNNCen;
            }

            // Initialize the central list
            if (CentralListSize[i] == 0){ // There is neighbor list in cluster, but no vector in the central list, just skip with Alpha = 0
                BoundaryConflictCentroidAlpha[i][i].first = std::make_pair(0, 0);
            }
            else{ // There is neighbor list in the cluster, iterate over all neighbor lists for closest
                float MinSqrDist2Line = std::numeric_limits<float>::max();
                float ResultQueryID = nc;
                float ResultAlpha = 0;
                float ResultAlphaNorm = 0;
                auto CentralCentroid = BoundaryConflictCentroidAlpha[i].find(i);
                assert(CentralCentroid != BoundaryConflictCentroidAlpha[i].end() && CentralCentroid->second.second.size() == Dimension);

                for (size_t d = 0; d < Dimension; d++){CentralCentroid->second.second[d] /= CentralListSize[i];}
                for (auto QueryIt = BoundaryConflictClusterResult[i].begin(); QueryIt != BoundaryConflictClusterResult[i].end(); QueryIt ++){
                    uint32_t QueryClusterID = QueryIt->first;

                    float DistCenNNCen = faiss::fvec_L2sqr(CentralCentroid->second.second.data(), Centroids.data() + i * Dimension, Dimension);
                    if (DistCenNNCen == 0){ResultQueryID = QueryClusterID; ResultAlpha = 0; ResultAlphaNorm = 0; break;}
                    float DistCenTarCen = faiss::fvec_L2sqr(CentralCentroid->second.second.data(), Centroids.data() + QueryClusterID * Dimension, Dimension);
                    float DistTarCenNNCen = faiss::fvec_L2sqr(Centroids.data() + i * Dimension, Centroids.data() + QueryClusterID * Dimension, Dimension);
                    float CosNLNNTarget = (DistCenNNCen + DistTarCenNNCen - DistCenTarCen) / (2 * sqrt(DistCenNNCen * DistTarCenNNCen));
                    float SqrDist2Line = DistCenNNCen * (1 - CosNLNNTarget * CosNLNNTarget);
                    if (SqrDist2Line < MinSqrDist2Line){ // Found the closest line for project
                        ResultQueryID = QueryClusterID;
                        ResultAlpha = sqrt(DistCenNNCen) * CosNLNNTarget / sqrt(DistTarCenNNCen);
                        ResultAlphaNorm = ResultAlpha * (ResultAlpha - 1) * DistTarCenNNCen;   
                        MinSqrDist2Line = SqrDist2Line;                     
                    }
                }
                if (ResultQueryID < nc){ // Find the line to project central list centroid
                    BoundaryConflictCentroidAlpha[i][nc + ResultQueryID].first = std::make_pair(ResultAlpha, ResultAlphaNorm);
                    BoundaryConflictCentroidAlpha[i].erase(i);
                }
                else{
                    std::cout << "Error in projecting cental list to neighbor cluster lines, " << ResultQueryID << " exit \n";
                    exit(1);
                }
            }
        }
        else{
        // There is no neighbor list in i-th cluster
            BoundaryConflictCentroidAlpha[i][i].first = std::make_pair(0, 0);
        }
    }
    writeBoundaryConflictCentroidsAlpha(BoundaryConflictCentroidAlpha);
    recorder.print_performance("computeConflictListCentroids");
}

void computeSaveNeighborList(){
    //if (!rerunComputeNL && exists(PathNeighborList))
    //    return;
    performance_recorder recorder("computeSaveNeighborList");
    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> BoundaryConflictClusterResult;
    loadBoundaryConflictClusterResult(BoundaryConflictClusterResult);

    std::vector<std::unordered_map<uint32_t, std::pair<std::pair<float, float>, std::vector<float>>>> BoundaryConflictCentroidsAlpha;
    loadBoundaryConflictCentroidsAlpha(BoundaryConflictCentroidsAlpha);

    std::vector<std::vector<uint32_t>> NonBCLists;
    loadNonBCLists(NonBCLists);

    std::vector<std::vector<uint32_t>> queryClusterIDs(nc); // Separate storage for queryClusterIDs
    std::vector<std::vector<uint32_t>> queryClusterIDIdx(nc); // Separate storage for merged vector index and ID
    std::vector<std::vector<uint32_t>> allVectorIDs(nc);

    std::vector<std::vector<float>> AlphaCentroidDists(nc);
    std::vector<std::vector<float>> AlphaCentroid(nc);

    for (uint32_t i = 0; i < nc; i++){
        auto & CentroidIter = BoundaryConflictCentroidsAlpha[i];
        uint32_t CurrentIDIdx = 0;
        for (auto QueryClusterIt = CentroidIter.begin(); QueryClusterIt != CentroidIter.end(); QueryClusterIt++){
            uint32_t QueryClusterID = QueryClusterIt->first;
            if (QueryClusterID >= nc){ // This is the central list projected to line between central and neighbor cluster
                uint32_t ActualQueryClusterID = QueryClusterID - nc;
                queryClusterIDs[i].emplace_back(QueryClusterID);
                auto & NonBCVec = NonBCLists[i];
                CurrentIDIdx += NonBCVec.size();
                queryClusterIDIdx[i].emplace_back(CurrentIDIdx);
                for (uint32_t j = 0; j < NonBCVec.size(); j++){
                    allVectorIDs[i].emplace_back(NonBCVec[j]);
                }
                AlphaCentroid[i].emplace_back(QueryClusterIt->second.first.first);
                AlphaCentroidDists[i].emplace_back(QueryClusterIt->second.first.second);
            }
            else if(QueryClusterID == i){ // This is the central list but no elements, or there is no neighbor list
                if (NonBCLists[i].size() > 0){ // There is no neighbor list
                    queryClusterIDs[i].emplace_back(i);
                    auto & NonBCVec = NonBCLists[i];
                    CurrentIDIdx += NonBCVec.size();
                    queryClusterIDIdx[i].emplace_back(CurrentIDIdx);
                    for (uint32_t j = 0; j < NonBCVec.size(); j++){
                        allVectorIDs[i].emplace_back(NonBCVec[j]);
                    }
                    AlphaCentroid[i].emplace_back(QueryClusterIt->second.first.first);
                    AlphaCentroidDists[i].emplace_back(QueryClusterIt->second.first.second);
                }
            }
            else{ // This is the neighbor list
                queryClusterIDs[i].emplace_back(QueryClusterID);
                auto & NeighborListVec = BoundaryConflictClusterResult[i][QueryClusterID];
                CurrentIDIdx += NeighborListVec.size();
                queryClusterIDIdx[i].emplace_back(CurrentIDIdx);
                for (uint32_t j = 0; j < NeighborListVec.size(); j++){
                    allVectorIDs[i].emplace_back(NeighborListVec[j]);
                }
                AlphaCentroid[i].emplace_back(QueryClusterIt->second.first.first);
                AlphaCentroidDists[i].emplace_back(QueryClusterIt->second.first.second);
            }
        }
    }

    std::vector<uint32_t> AssignID(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, AssignID);
    for (uint32_t i = 0; i < nb; i++){
        uint32_t Assignment = AssignID[i];
        assert(std::find(allVectorIDs[Assignment].begin(), allVectorIDs[Assignment].end(), i) != allVectorIDs[Assignment].end());
    }

    writeNeighborList(queryClusterIDs, queryClusterIDIdx, allVectorIDs, AlphaCentroidDists, AlphaCentroid, PathNeighborList);

    size_t NumNestList = 0;
    for (auto NNit = queryClusterIDs.begin(); NNit != queryClusterIDs.end(); NNit ++){
        NumNestList += (*NNit).size();
    }
    std::cout << "The total number of NeST list is: " << NumNestList << "\n";
}

void computeNeighborList() {
    // Compute the boundary conflict map
    computeNNConflictListAndSave();
    computeConflictDistListAndSave();
    computeBoundaryPruneAndSave();
    computeConflictListCentroids();
    computeSaveNeighborList();
}

// Check the proportion of conflict vectors
// Check how many gt vectors are in neighbor list
// We want to know: 
// 1. Total proportion of boundary conflict in gt
// 2. Proportion of boundary conflict gt in neighbor list
// 3. Proportion of boundary conflict gt in hnsw clusters
// 4. Index of the cluster for each gt 
// 5. Distance of the cluster for each gt
// 6. Distance of the neighbor list for each gt (we expect it should be much smaller than the distance to cluster)
void analyzeGT(){
    performance_recorder recorder("analyzeGT");
    std::vector<std::vector<uint32_t>> queryClusterIDs;
    std::vector<std::vector<uint32_t>> queryClusterIDIx;
    std::vector<std::vector<uint32_t>> allVectorIds;
    std::vector<std::vector<float>> AlphaCentroidDists;
    std::vector<std::vector<float>> AlphaCentroid;

    loadNeighborList(queryClusterIDs, queryClusterIDIx, allVectorIds, AlphaCentroidDists, AlphaCentroid);

    std::vector<uint32_t> AssignID(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, AssignID);

    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidInputStream = std::ifstream(PathCentroid, std::ios::binary);
    readXvec<float>(CentroidInputStream, Centroids.data(), Dimension, nc, true, true);

    for (uint32_t i = 0; i < nb; i++){
        uint32_t ClusterID = AssignID[i];
        assert(std::find(allVectorIds[ClusterID].begin(), allVectorIds[ClusterID].end(), i) != allVectorIds[ClusterID].end());
    }

    std::cout << "Loaded neighbor list without compression \n";
    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInputStream = std::ifstream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    readXvecFvec<DType>(BaseInputStream, BaseVectors.data(), Dimension, nb, true, true);
    BaseInputStream.close();

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex);
    CenGraph->ef_ = EfQueryEvalaution;

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
    if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInputStream(PathQuery, std::ios::binary);
    if (!QueryInputStream) {std::cerr << "Error opening PathQuery file: " << PathQuery << "\n";exit(1);}
    readXvecFvec<DType>(QueryInputStream, Query.data(), Dimension, nq, true, true);
    GTInputStream.close(); QueryInputStream.close();

    // Check the GT distribution in base index and in NL
    size_t recallk = RecallK[0];
    size_t ef = EfSearch[0];
    std::vector<uint32_t> HNSWID(ef);
    std::vector<float> HNSWDist(ef);

    std::vector<uint32_t> AvgGTClusterth(recallk, 0);
    std::vector<uint32_t> AvgGTClusterNum(recallk, 0);
    std::vector<float> AvgGTClusterDist(recallk, 0);

    std::vector<float> AvgGTNLNum(recallk, 0);
    std::vector<float> AvgGTNLDist(recallk, 0);
    size_t NumConflictgt = 0;     // Total number of conflict NN
    size_t NumNeSTList = 0;
    size_t NumClusterVisited = 0;
    size_t NumConflictinNL = 0;   // Number of conflict NN in NL
    size_t NumConflictinHNSW = 0; // Number of conflict NN in efList

    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto searchresult = CenGraph->searchKnn(Query.data() + QueryIdx * Dimension, ef);
        for (size_t i = 0; i < ef; i++){
            HNSWID[ef - i - 1] = searchresult.top().second;
            HNSWDist[ef - i - 1] = searchresult.top().first;
            searchresult.pop();
        }

        std::vector<uint32_t> GTClusterth(recallk, 0);
        std::vector<float> GTClusterDist(recallk, 0);
        std::vector<float> GTNLDist(recallk, 0);

        for (size_t i = 0; i < recallk; i++){
            auto gt = GT[QueryIdx * ngt + i];
            auto Assignment = AssignID[gt];
            if (AssignID[gt] == HNSWID[0]){
                continue; // Gt is in the same cluster with query, but we only consider the conflict GT
            }
            NumConflictgt ++;

            bool exitFlag = false;
            for (size_t j = 1 ; j < ef && !exitFlag; j++){
                uint32_t ClusterID = HNSWID[j];
                NumNeSTList += queryClusterIDs[ClusterID].size();
                NumClusterVisited ++;

                for (size_t k = 0; k < queryClusterIDs[ClusterID].size(); k++){
                    auto QueryClusterID = queryClusterIDs[ClusterID][k];
                    float Alpha = AlphaCentroid[ClusterID][k];
                    float AlphaNorm = AlphaCentroidDists[ClusterID][k];

                    auto StartIndex = k == 0 ? 0 : queryClusterIDIx[ClusterID][k - 1];
                    auto EndIndex = queryClusterIDIx[ClusterID][k];
                    assert(allVectorIds[ClusterID].size() >= EndIndex);
                    for (auto Index = StartIndex; Index < EndIndex; Index++){
                        if (allVectorIds[ClusterID][Index] == gt){ // Found gt in the sublist
                            GTClusterth[i] = j; GTClusterDist[i] = HNSWDist[j];
                            NumConflictinHNSW ++;
                            if (ClusterID != QueryClusterID && QueryClusterID < nc){

                                NumConflictinNL ++;
                                float QueryClusterDist = CenGraph->fstdistfunc_(Query.data() + QueryIdx * Dimension, Centroids.data() + QueryClusterID * Dimension, CenGraph->dist_func_param_);
                                float QueryNLDist = (1 - Alpha) * (HNSWDist[j]) + Alpha * QueryClusterDist + AlphaNorm;
                                GTNLDist[i] = QueryNLDist;
                                exitFlag = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < recallk; i++){
            if (GTClusterth[i] > 0){
                AvgGTClusterNum[i] += 1; AvgGTClusterth[i] += GTClusterth[i]; AvgGTClusterDist[i] += GTClusterDist[i];
            }
            if (GTNLDist[i] > 0){
                AvgGTNLNum[i] += 1; AvgGTNLDist[i] += GTNLDist[i];
            }
        }
    }

    std::cout << "Total conflict gt: " << float (NumConflictgt) / nq << " Num of Conflict gt in HNSW " << float(NumConflictinHNSW) / nq  << " Num of Conflict gt in NL: " << float (NumConflictinNL) / nq << " Proportion of conflict gt in NL: " << float(NumConflictinNL) / NumConflictgt << 
                 " Total number of NeST list: " << float(NumNeSTList) / (NumClusterVisited) << "\n";
    std::cout << std::left << std::setw(20) << "GT-th:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << i + 1; }
    std::cout << " |\n" << std::setw(20) << "Cluster-th:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterth[i]) / (AvgGTClusterNum[i]); }
    std::cout << " |\n" << std::setw(20) << "Num in Cluster:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterNum[i]) / nq; }
    std::cout << " |\n" << std::setw(20) << "Dist to Cluster:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterDist[i]) / (AvgGTClusterNum[i]); }
    std::cout << " |\n" << std::setw(20) << "Num in NList:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTNLNum[i] / nq); }
    std::cout << " |\n" << std::setw(20) << "GT-dist:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTNLDist[i] / (AvgGTNLNum[i])); }
    std::cout << " |\n";
    delete CenGraph;

    size_t CentralListSize = 0;
    for (uint32_t i = 0; i < nc; i++){
        for (size_t j = 0; j < queryClusterIDs[i].size(); j++){
            size_t QueryClusterID = queryClusterIDs[i][j];
            if (QueryClusterID == i || QueryClusterID > nc){
                auto StartIndex = j == 0? 0 : queryClusterIDIx[i][j - 1];
                auto EndIndex = queryClusterIDIx[i][j];
                CentralListSize += (EndIndex - StartIndex);
            }
        }
    }
    std::cout << "The vectors in NeST list: " << nb - CentralListSize << " The vectors in central list: " << CentralListSize << "\n"; 
}


void evaluateNeighborList(){
    performance_recorder recorder("evaluateNeighborList");
    std::vector<std::vector<uint32_t>> queryClusterIDs;
    std::vector<std::vector<uint32_t>> queryClusterIDIx;
    std::vector<std::vector<uint32_t>> allVectorIds;
    std::vector<std::vector<float>> AlphaCentroidDists;
    std::vector<std::vector<float>> AlphaCentroid;

    loadNeighborList(queryClusterIDs, queryClusterIDIx, allVectorIds, AlphaCentroidDists, AlphaCentroid);
    std::cout << "Loaded neighbor list without compression \n";
    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInputStream = std::ifstream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    readXvecFvec<DType>(BaseInputStream, BaseVectors.data(), Dimension, nb, false, false);
    BaseInputStream.close();

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex);
    CenGraph->ef_ = EfQueryEvalaution;

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
    if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, false, false);
    std::ifstream QueryInputStream(PathQuery, std::ios::binary);
    if (!QueryInputStream) {std::cerr << "Error opening PathQuery file: " << PathQuery << "\n";exit(1);}
    readXvecFvec<DType>(QueryInputStream, Query.data(), Dimension, nq, false, false);
    GTInputStream.close(); QueryInputStream.close();

    std::cout << "Setting of NLTargetK:  " << NLTargetK << " Use prune or not: " << prunevectors << "\n";
    for (size_t recallidx = 0; recallidx < NumRecall; recallidx++) {
        size_t recallk = RecallK[recallidx];
        std::cout << "Search results for recallk = " << recallk << "\n";

        // Print the table header
        std::cout << "-------------------------------------------------------------------------------------------------------\n";
        std::cout << "| EfSearch | MaxItem | Recall Rate | Recall@1 Rate | Time per Query(us) |    Sum    |  All Potential  |\n";
        std::cout << "-------------------------------------------------------------------------------------------------------\n";
        std::vector<float> QueryDists(nq * recallk);
        std::vector<int64_t>QueryIds(nq * recallk);

    for (size_t searchnumidx = 0; searchnumidx < NumPara; searchnumidx++) {
    size_t ef = EfSearch[searchnumidx];
    size_t maxitem = MaxItem[searchnumidx];

    std::vector<bool> VisitedClusterFlag(nc, false);
    std::vector<float> HNSWClusterDist(nc, 0);
    std::vector<uint32_t>HNSWID(ef);
    size_t SumVisitedItem = 0;
    size_t SumPotentialItem = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        std::vector<uint32_t> NLNNClusterID;
        std::vector<uint32_t> NLQueryClusterIndex;
        std::vector<float> NLQueryNLDist;
        std::vector<uint32_t> FoundCluster;

        faiss::maxheap_heapify(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
        auto SearchResult = CenGraph->searchKnn(Query.data() + QueryIdx * Dimension, ef);

        for (uint32_t i = 0; i < ef; i++){
            uint32_t NNClusterID =SearchResult.top().second;
            HNSWID[ef - i - 1] = NNClusterID;
            VisitedClusterFlag[NNClusterID] = true;
            HNSWClusterDist[NNClusterID] = SearchResult.top().first;
            FoundCluster.emplace_back(NNClusterID);
            SumPotentialItem += allVectorIds[NNClusterID].size();
            SearchResult.pop();

        }

        // Search the base cluster
        uint32_t CentralClusterID = HNSWID[0];
        SumVisitedItem += allVectorIds[CentralClusterID].size();
        for (size_t j = 0; j < allVectorIds[CentralClusterID].size(); ++j){

            float Dist = CenGraph->fstdistfunc_(BaseVectors.data() + allVectorIds[CentralClusterID][j] * Dimension, Query.data() + QueryIdx * Dimension, CenGraph->dist_func_param_);
            if (Dist < QueryDists[QueryIdx * recallk]){
                faiss::maxheap_pop(recallk,  QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
                faiss::maxheap_push(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk, Dist, allVectorIds[CentralClusterID][j]);
            }
        }

        // Compute the dist to list centroids
        size_t SumNLNum = 0;
        float SumQueryNLDist = 0;
        for (size_t i = 1; i < ef; i++){
            uint32_t NNClusterID = HNSWID[i];
            for (size_t j = 0; j < queryClusterIDs[NNClusterID].size(); j++){
                uint32_t QueryClusterID = queryClusterIDs[NNClusterID][j];
                if (QueryClusterID >= nc){QueryClusterID -= nc;}

                if (!VisitedClusterFlag[QueryClusterID]){
                    HNSWClusterDist[QueryClusterID] = CenGraph->fstdistfunc_(Query.data() + QueryIdx * Dimension, CenGraph->getDataByInternalId(QueryClusterID), CenGraph->dist_func_param_);
                    VisitedClusterFlag[QueryClusterID] = true;
                }
                float Alpha = AlphaCentroid[NNClusterID][j];
                float AlphaNorm = AlphaCentroidDists[NNClusterID][j];
                float QueryNLDist = (1-Alpha) * (HNSWClusterDist[NNClusterID]) + Alpha * (HNSWClusterDist[QueryClusterID]) + AlphaNorm;
                NLNNClusterID.emplace_back(NNClusterID);
                NLQueryClusterIndex.emplace_back(j);
                NLQueryNLDist.emplace_back(QueryNLDist);
                SumQueryNLDist += QueryNLDist;
            }
        }
        float AvgQueryNLDist = SumQueryNLDist / NLNNClusterID.size();


        // Compute the dist to vector in close lists
        for (size_t i = 0; i < NLNNClusterID.size(); i++){
            if (NLQueryNLDist[i] < AvgQueryNLDist){
                uint32_t QueryClusterIndex = NLQueryClusterIndex[i];

                if (queryClusterIDs[NLNNClusterID[i]][QueryClusterIndex] >= nc){
                    continue;
                }
                uint32_t NNClusterID = NLNNClusterID[i];
                uint32_t StartIndex = QueryClusterIndex == 0? 0 : queryClusterIDIx[NNClusterID][QueryClusterIndex - 1];
                uint32_t EndIndex = queryClusterIDIx[NNClusterID][QueryClusterIndex];
                SumVisitedItem += (EndIndex - StartIndex);

                for (uint32_t Index = StartIndex; Index < EndIndex; Index++){
                    float Dist = CenGraph->fstdistfunc_(BaseVectors.data() + allVectorIds[NNClusterID][Index] * Dimension, Query.data() + QueryIdx * Dimension, CenGraph->dist_func_param_);
                    if(Dist < QueryDists[QueryIdx * recallk]){
                        faiss::maxheap_pop(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
                        faiss::maxheap_push(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk, Dist, allVectorIds[NNClusterID][Index]);
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    size_t correctCount = 0;
    size_t firstElementCorrectCount = 0;

    for (size_t queryIdx = 0; queryIdx < nq; ++queryIdx){
        std::unordered_set<int64_t> targetSet(GT.begin() + ngt * queryIdx, GT.begin() + ngt * queryIdx + recallk);
        std::unordered_set<int64_t> firstElementSet = {GT[ngt * queryIdx]};

        for (size_t recallIdx = 0; recallIdx < recallk; recallIdx++) {
            int64_t queryId = QueryIds[queryIdx * recallk + recallIdx];
            correctCount += targetSet.count(queryId);
            firstElementCorrectCount += firstElementSet.count(queryId);
        }

        // Ensure the correct count of matching elements is within expected bounds
        if (!(correctCount <= recallk * (queryIdx + 1) && firstElementCorrectCount <= queryIdx + 1)) {
            // Print an error message indicating the mismatched values
            std::cout << "Error: Correct Count = " << correctCount 
                    << ", First Element Correct Count = " << firstElementCorrectCount 
                    << ", Query Index = " << queryIdx << "\n";
                    
            // Exit the program as the inconsistency indicates a problem
            exit(1);
        }
    }

    float recallRateForAllElements = float(correctCount) / (nq * recallk);
    float recallRateForFirstElement = float(firstElementCorrectCount) / nq;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Print the results in rows
    std::cout << "| " << std::setw(8) << ef << " | " << std::setw(7) << maxitem   << " | " << std::setw(11) << recallRateForAllElements
                << " | " << std::setw(13) << recallRateForFirstElement << " | " << std::setw(18) << float(duration.count()) / nq << " | " << std::setw(9) << SumVisitedItem / nq << " | " << std::setw(15) << SumPotentialItem / nq << " |\n";
    }
    }
    delete CenGraph;
}

int main(int argc, char* argv[]) {
    int option;
    while ((option = getopt(argc, argv, "c:n:l:")) != -1) {
        switch (option) {
            case 'c':
                rerunCentroids = std::atoi(optarg) != 0;
                break;
            case 'n':
                rerunComputeNN = std::atoi(optarg) != 0;
                break;
            case 'l':
                rerunComputeNL = std::atoi(optarg) != 0;
                break;
            default:
                std::cerr << "Invalid option\n";
                return EXIT_FAILURE;
        }
    }
    // Output the parsed values (for testing purposes)
    std::cout << "rerunCentroids: " << rerunCentroids << "\n"  << "rerunComputeNN: " << rerunComputeNN << "\n" << "rerunComputeNL: " << rerunComputeNL << "\n";

    PrepareFolder(PathNLFolder.c_str());

    computeCentroids();
    computeCenHNSWGraph();
    computeDatasetNeighbor();
    //testNNGraphQuality();

    computeNN();
    computeNeighborList();
    analyzeGT();
    evaluateNeighborList();
    return 0;
}





#include "NL.h"
// Define rerun flags
bool rerunCentroids = false;
bool rerunComputeNN = false;
bool rerunComputeNL = false;

struct cmp{bool operator ()( std::pair<uint32_t, float> &a, std::pair<uint32_t, float> &b){return a.second < b.second;}};

void computeCentroids() {
    if (!rerunCentroids && exists(PathCentroid))
        return;

    performance_recorder recorder = performance_recorder("computeCentroids");
    std::vector<float> Centroids(nc * Dimension);
    std::vector<float> TrainSet(nt * Dimension);
    std::ifstream TrainInputStream(PathTrain, std::ios::binary);
    if (!TrainInputStream) {
        std::cerr << "Error opening PathBase file: " << PathTrain << "\n";
        exit(1);
    }
    readXvecFvec<DType>(TrainInputStream, TrainSet.data(), Dimension, nt, true, true);
    faisskmeansClustering(TrainSet.data(), Centroids.data());

    std::ofstream CentroidsOutputStream(PathCentroid, std::ios::binary);
    if (!CentroidsOutputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}

    uint32_t D = Dimension;
    for (size_t i = 0; i < nc; i++) {
        CentroidsOutputStream.write(reinterpret_cast<char*>(&D), sizeof(uint32_t));
        CentroidsOutputStream.write(reinterpret_cast<char*>(Centroids.data() + i * Dimension), Dimension * sizeof(float));
    }
}

void computeCenHNSWGraph() {
    if (!rerunCentroids && !rerunComputeNN && exists(PathCenGraphIndex))
        return;

    performance_recorder recorder("computeHNSWGraph");
    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * HNSWGraph;
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);

    HNSWGraph = new hnswlib::HierarchicalNSW<float>(&l2space, nc, M, EfConstruction);
    HNSWGraph->addPoint(Centroids.data(), 0);
    #pragma omp parallel for
    for (size_t i = 1; i < nc; i++){
        HNSWGraph->addPoint(Centroids.data() + i * Dimension, i);
    }
    HNSWGraph->saveIndex(PathCenGraphIndex);
    delete HNSWGraph; // Free the allocated memory
}

/*
Assign the vectors and save computed distance
*/
void computeDatasetNeighbor() {
    performance_recorder recorder("computeDatasetNeighbor");
    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex, false, nc, false);
    CenGraph->ef_ = EfAssignment;

    if (rerunCentroids || !exists(PathBaseNeighborID) || !exists(PathBaseNeighborDist)){
        assert(Assign_num_batch > 0);
        size_t Batch_size = nb / Assign_num_batch;
        std::ofstream BasesetNeighborIDOutputStream(PathBaseNeighborID, std::ios::binary);
        std::ofstream BasesetNeighborDistOutputStream(PathBaseNeighborDist, std::ios::binary);
        std::ifstream BasesetInputStream(PathBase, std::ios::binary);
        if (!BasesetNeighborIDOutputStream || !BasesetNeighborDistOutputStream || !BasesetInputStream) {
            std::cerr << "Error opening PathBaseNeighborID PathBaseNeighborDist PathBase file: " << "\n";exit(1);}

        std::vector<float> BaseBatch(Batch_size * Dimension);
        std::vector<uint32_t> BaseIDBatch(Batch_size * NeighborNum);
        std::vector<float> BaseDistBatch(Batch_size * NeighborNum);

        for (size_t i = 0; i < Assign_num_batch; i++){
            readXvecFvec<DType>(BasesetInputStream, BaseBatch.data(), Dimension, Batch_size, true, true);

            #pragma omp parallel for
            for (size_t j = 0; j < Batch_size; j++){
                auto result = CenGraph->searchKnn(BaseBatch.data() + j * Dimension, NeighborNum, NumComp);
                for (size_t temp = 0; temp < NeighborNum; temp++){
                    BaseIDBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().second;
                    BaseDistBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().first;
                    result.pop();
                }
            }
            BasesetNeighborIDOutputStream.write((char *) BaseIDBatch.data(), Batch_size * NeighborNum * sizeof(uint32_t));
            BasesetNeighborDistOutputStream.write((char *) BaseDistBatch.data(), Batch_size * NeighborNum * sizeof(float));
        }
        BasesetNeighborIDOutputStream.close();
        BasesetNeighborDistOutputStream.close();
    }

    recorder.print_performance("computeBasesetNeighbor");

    // Compute the neighbor cluster ID for trainset
    if (rerunCentroids || !exists(PathTrainNeighborID) || !exists(PathTrainNeighborDist)){
        std::ofstream TrainsetNeighborIDOutputStream(PathTrainNeighborID, std::ios::binary);
        std::ofstream TrainsetNeighborDistOutputStream(PathTrainNeighborDist, std::ios::binary);
        std::ifstream TrainsetInputStream(PathTrain, std::ios::binary);
        if (!TrainsetNeighborIDOutputStream || !TrainsetNeighborDistOutputStream || !TrainsetInputStream) {
            std::cerr << "Error opening PathTrainNeighborID PathTrainNeighborDist PathTrain file: " << "\n";exit(1);}

        std::vector<float> Trainset(nt * Dimension);
        std::vector<uint32_t> TrainsetID(nt * NeighborNum);
        std::vector<float> TrainsetDist(nt * NeighborNum);
        readXvecFvec<DType>(TrainsetInputStream, Trainset.data(), Dimension, nt, true, true);
        #pragma omp parallel for
        for (size_t i = 0; i < nt; i++){
            auto result = CenGraph->searchKnn(Trainset.data() + i * Dimension, NeighborNum, NumComp);
            for (size_t temp = 0; temp < NeighborNum; temp++){
                TrainsetID[i * NeighborNum + NeighborNum - temp - 1] = result.top().second;
                TrainsetDist[i * NeighborNum + NeighborNum - temp - 1] = result.top().first;
                result.pop();
            }
        }
        TrainsetNeighborIDOutputStream.write((char *) TrainsetID.data(), nt * NeighborNum * sizeof(uint32_t));
        TrainsetNeighborDistOutputStream.write((char *)TrainsetDist.data(), nt * NeighborNum * sizeof(float));
        TrainsetNeighborIDOutputStream.close();
        TrainsetNeighborDistOutputStream.close();
    }
    delete CenGraph;
    recorder.print_performance("computeTrainsetNeighbor");
}

void computeNN() {
    if (exists(PathTrainsetNN) && (!rerunComputeNN)) 
        return;
    performance_recorder recorder("computeNN");
    PrepareFolder(PathSubGraphFolder.c_str());

    #pragma omp parallel for
    for (size_t i = 0; i < Graph_num_batch; i++) {
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        if (exists(PathSubGraphInfo) && exists(PathSubGraphEdge)){
            continue;
        }
        std::ifstream BasesetInputStream(PathBase, std::ios::binary);
        if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}

        std::vector<float> BaseBatch(Graph_batch_size * Dimension);
        BasesetInputStream.seekg(i * Graph_batch_size * (Dimension * sizeof(DType) + sizeof(uint32_t)), std::ios::beg);
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);
        BasesetInputStream.close();
        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(Dimension, Graph_batch_size, M, 2 * M, EfConstruction);
        for (size_t j = 0; j < Graph_batch_size; j++){
            SubGraph->addPoint(BaseBatch.data() + j * Dimension);
        }
        SubGraph->SaveInfo(PathSubGraphInfo);
        SubGraph->SaveEdges(PathSubGraphEdge);
        delete SubGraph;
    }
    recorder.print_performance("buildSubGraphs");

    // Search NNs
    std::vector<float> TrainSet(nt * Dimension);
    std::ifstream TrainInputStream(PathTrain, std::ios::binary);
    if (!TrainInputStream) {std::cerr << "Error opening PathTrain file: " << PathTrain << "\n";exit(1);}
    readXvecFvec<DType>(TrainInputStream, TrainSet.data(), Dimension, nt, true, true);
    TrainInputStream.close();
    std::vector<std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, cmp>> NNQueueList(nt);

    std::ifstream BasesetInputStream(PathBase, std::ios::binary);
    if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::vector<float> BaseBatch(Graph_batch_size * Dimension);

    for (size_t i = 0; i < Graph_num_batch; i++){
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        assert(exists(PathSubGraphInfo) && exists(PathSubGraphEdge));
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);        

        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, BaseBatch);
        SubGraph->efSearch = EfNN;

        uint32_t StartIndex = i * Graph_batch_size;
        #pragma omp parallel for
        for (size_t j = 0; j < nt; j++){
            auto result = SubGraph->searchKnn(TrainSet.data() + j * Dimension, SearchK);
            for (size_t temp = 0; temp < SearchK; temp++){
                if (NNQueueList[j].size() < SearchK){
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                else if (result.top().first < NNQueueList[j].top().second){
                    NNQueueList[j].pop();
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                result.pop();
            }
        }
        delete(SubGraph);
    }
    // Write Results
    std::ofstream BasesetNNOutputStream(PathTrainsetNN, std::ios::binary);
    if (!BasesetNNOutputStream) {std::cerr << "Error opening PathTrainsetNN file: " << PathTrainsetNN << "\n";exit(1);}
    for (size_t i = 0; i < nt; i++){
        std::vector<uint32_t> EachVecNN(SearchK);
        for (size_t j = 0; j < SearchK; j++)
        {
            EachVecNN[SearchK - j - 1] = NNQueueList[i].top().first;
            NNQueueList[i].pop();
        }
        BasesetNNOutputStream.write((char *) EachVecNN.data(), SearchK * sizeof(uint32_t));
    }
    BasesetNNOutputStream.close();
}

void testNNGraphQuality(size_t recallK = 20){
    assert(recallK <= ngt);
    performance_recorder recorder("testNNGraphQuality");
    std::vector<float> QuerySet(nq * Dimension);
    std::vector<uint32_t> GTSet(nq * ngt);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvecFvec<DType>(QueryInput, QuerySet.data(), Dimension, nq, true, true);
    readXvec<uint32_t>(GTInput, GTSet.data(), ngt, nq, true, true);

    std::vector<std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, cmp>> NNQueueList(nq);
    std::ifstream BasesetInputStream(PathBase, std::ios::binary);
    if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    std::vector<float> BaseBatch(Graph_batch_size * Dimension);
    for (size_t i = 0; i < Graph_num_batch; i++){
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        assert(exists(PathSubGraphInfo) && exists(PathSubGraphEdge));
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);        

        hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, BaseBatch);
        SubGraph->efSearch = EfNN;

        uint32_t StartIndex = i * Graph_batch_size;
        #pragma omp parallel for
        for (size_t j = 0; j < nq; j++){
            auto result = SubGraph->searchKnn(QuerySet.data() + j * Dimension, recallK);
            for (size_t temp = 0; temp < recallK; temp++){
                if (NNQueueList[j].size() < recallK){
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                else if (result.top().first < NNQueueList[j].top().second){
                    NNQueueList[j].pop();
                    NNQueueList[j].emplace(std::make_pair(StartIndex + result.top().second, result.top().first));
                }
                result.pop();
            }
        }
        delete(SubGraph);
    }
    float SumCorrect = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<uint32_t> gt;
        for (size_t j = 0; j < recallK; j++){gt.insert(GTSet[i * ngt + j]);}

        for (size_t j  = 0; j < recallK; j++){
            if (gt.count(NNQueueList[i].top().first) != 0){
                SumCorrect ++;
            }
        }
    }
    std::cout << "*****************************************************************************************************************\n"; 
    std::cout << "The overall recall@" << recallK << " for the query using this parameter setting: " << SumCorrect / (nq * recallK) << "\n";
    std::cout << "*****************************************************************************************************************\n"; 
}

void testMinNumTrain(){
    performance_recorder recorder("testMinNumTrain");
    std::vector<uint32_t> TrainsetNN(nt * SearchK);
    std::ifstream TrainsetNNInputStream(PathTrainsetNN, std::ios::binary);
    TrainsetNNInputStream.read(reinterpret_cast<char*>(TrainsetNN.data()), nt * SearchK * sizeof(uint32_t));
    TrainsetNNInputStream.close();

    std::vector<uint32_t> TrainsetAssignment(nt);
    loadNeighborAssignments(PathTrainNeighborID, nt, NeighborNum, TrainsetAssignment);
    std::vector<uint32_t> BasesetAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BasesetAssignment);

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
    if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, false, false);
    std::ifstream QueryInputStream(PathQuery, std::ios::binary);
    if (!QueryInputStream) {std::cerr << "Error opening PathQuery file: " << PathQuery << "\n";exit(1);}
    readXvecFvec<DType>(QueryInputStream, Query.data(), Dimension, nq, false, false);
    GTInputStream.close(); QueryInputStream.close();

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex);
    CenGraph->ef_ = EfQueryEvalaution;

    std::unordered_set<uint32_t> AllQueryBCNN;

    // Table Header
    std::cout << std::setw(10) << "nt_NEST" << " | "
            << std::setw(15) << "tolerateprop" << " | "
            << std::setw(10) << "NLTargetK" << " | "
            << std::setw(10) << "Recall[0]" << " | "
            << std::setw(15) << "MinTrainNum" << " | "
            << std::setw(20) << "RemainingUnfoundGT" << " | "
            << std::setw(15) << "NeST List Size" << " | "
            << std::setw(20) << "Min Trainset Size" << " | "
            << std::setw(25) << "Min Trainset NeST Size" << std::endl;

    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    std::vector<size_t> nt_NEST_values = {1000000};
    std::vector<float> tolerateprop_values = {0.01};
    std::vector<size_t> NLTargetK_values = {10};
    std::vector<size_t> Recall = {10};
    // Loop through values for nt_NEST, tolerateprop, NLTargetK, and Recall[0]
    for(auto nt_NEST: nt_NEST_values){
        for(auto tolerateprop: tolerateprop_values){
            for(auto NLTargetK: NLTargetK_values){
                for(auto R0: Recall){

                    for (size_t i = 0; i < nq; i++){
                        auto Result = CenGraph->searchKnn(Query.data() + i * Dimension, 1, NumComp);
                        uint32_t CentralID = Result.top().second;
                        for (size_t j = 0; j < R0; j++){
                            uint32_t NNAssignment = BasesetAssignment[GT[i * ngt + j]];
                            if (NNAssignment != CentralID){ // This is the conflict NN
                                AllQueryBCNN.insert(GT[i * ngt + j]);
                            }
                        }
                    }
                    size_t SumBCGTSize = AllQueryBCNN.size();

                    size_t MaxLoss = tolerateprop * AllQueryBCNN.size();
                    std::unordered_set<uint32_t> AllTrainBCNN;
                    std::unordered_set<uint32_t> AllQueryBCTrainID;
                    std::unordered_set<uint32_t> MinTrainBCNN;
                    size_t MinTrainNum = 0;
                    for (; MinTrainNum < nt_NEST; MinTrainNum++){
                        uint32_t CentralID = TrainsetAssignment[MinTrainNum];
                        for (size_t j = 0; j < NLTargetK; j++){
                            uint32_t NNAssignment = BasesetAssignment[TrainsetNN[MinTrainNum * SearchK + j]];
                            if (CentralID != NNAssignment){
                                AllTrainBCNN.insert(TrainsetNN[MinTrainNum * SearchK + j]);
                                if (AllQueryBCNN.find(TrainsetNN[MinTrainNum * SearchK + j]) != AllQueryBCNN.end()){
                                    AllQueryBCNN.erase(TrainsetNN[MinTrainNum * SearchK + j]);
                                    AllQueryBCTrainID.insert(MinTrainNum);
                                    for (size_t temp = 0; temp < NLTargetK; temp++){
                                        if (BasesetAssignment[TrainsetNN[MinTrainNum * SearchK + temp]] != CentralID){
                                            MinTrainBCNN.insert(TrainsetNN[MinTrainNum * SearchK + temp]);
                                        }
                                    }
                                    if (AllQueryBCNN.size() < MaxLoss){
                                        break;
                                    }
                                }
                            }
                        }
                        if (AllQueryBCNN.size() < MaxLoss){
                            break;
                        }
                    }

                // Print row for the current iteration
                std::cout << std::setw(10) << nt_NEST << " | "
                          << std::setw(15) << tolerateprop << " | "
                          << std::setw(10) << NLTargetK << " | "
                          << std::setw(10) << R0 << " | "  // Assuming Recall[0] is represented by R0 here
                          << std::setw(15) << MinTrainNum << " | "
                          << std::setw(10) << AllQueryBCNN.size() << " " << float(AllQueryBCNN.size()) / SumBCGTSize << " | "
                          << std::setw(15) << AllTrainBCNN.size() << " | "
                          << std::setw(20) << AllQueryBCTrainID.size() << " | "
                          << std::setw(25) << MinTrainBCNN.size() << std::endl;
                size_t MinTrainBCNNNum = MinTrainBCNN.size();
                std::string PathMinTrainIDs = PathNLFolder  + Dataset + "-" + std::to_string(MinTrainBCNNNum) + ".id";
                std::cout << "Saving trainset ID to: " << PathMinTrainIDs <<"\n";
                std::ofstream OutFileMinTrainIDs(PathMinTrainIDs, std::ios::binary);
                for (auto ID : AllQueryBCTrainID){
                    OutFileMinTrainIDs.write(reinterpret_cast<char*>(&ID), sizeof(uint32_t));
                }
                OutFileMinTrainIDs.close();
                }
            }
        }
    }
}

void computeNeSTList(){
    // Determine the number of list size based on the conflict gt
    // Clustering the conflict gt and pruned gt
    // Prune the vectors based on the distance information
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidsInputStream(PathCentroid, std::ios::binary);
    if (!CentroidsInputStream) {std::cerr << "Error opening PathCentroid file: " << PathCentroid << "\n";exit(1);}
    readXvec<float>(CentroidsInputStream, Centroids.data(), Dimension, nc, true, true);
    CentroidsInputStream.close();
    std::vector<uint32_t> TrainsetNN(nt * SearchK);
    std::ifstream TrainsetNNInputStream(PathTrainsetNN, std::ios::binary);
    if (!TrainsetNNInputStream) {std::cerr << "Error opening PathBase file: " << PathTrainsetNN << "\n";exit(1);}
    TrainsetNNInputStream.read(reinterpret_cast<char*>(TrainsetNN.data()), nt * SearchK * sizeof(uint32_t));
    TrainsetNNInputStream.close();

    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);
    std::vector<uint32_t> TrainsetAssignment(nt);
    loadNeighborAssignments(PathTrainNeighborID, nt, NeighborNum, TrainsetAssignment);
    std::vector<float> BaseBatch(Assign_batch_size * Dimension);

    size_t nt_NEST = 1000000; 
    float tolerateprop = 0.01;
    size_t R0 = 10;
    size_t NLTargetKEval = 10;
    size_t MinTrainNum = 6950;
    size_t MinTrainBCNNNum = 42577;
    std::string PathMinTrainIDs = PathNLFolder + Dataset +  "-" + std::to_string(MinTrainBCNNNum) + ".id";

    std::cout <<"Loading trainset ID from: " << PathMinTrainIDs << "\n";
    std::unordered_set<uint32_t> MinTrainIDSet;
    std::ifstream InFile(PathMinTrainIDs, std::ios::binary);
    if (!InFile){
        std::cout << "Cannot find this file\n";
        exit(0);
    }
    uint32_t ID = 0;
    for (size_t i = 0; i < MinTrainBCNNNum; i++){
        InFile.read(reinterpret_cast<char*>(&ID), sizeof(uint32_t));
        MinTrainIDSet.insert(ID);
    }

    /*******************************************************************************************/
    // If we need to prune the vector, we need to store the cluster ID pair
    // Other wise, we just need to keep the central cluster ID
    std::vector<std::unordered_set<uint32_t>> BCCluster(nc);
    std::vector<std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> BCClusterPrune;
    std::vector<std::unordered_set<uint32_t>> BCClusterConflict;

    BCClusterConflict.resize(nc);
    if (prunevectors){
        BCClusterPrune.resize(nc);
    }
    size_t SumConflict = 0;
    for (uint32_t i = 0; i < nt; i++){
        if (MinTrainIDSet.find(i) == MinTrainIDSet.end()){
            continue;
        }
        uint32_t QueryClusterID = TrainsetAssignment[i];
        for (size_t j = 0; j < NLTargetK; j++){
            uint32_t NNVecID = TrainsetNN[i * SearchK + j];
            uint32_t NNClusterID = BaseAssignment[NNVecID];

            if (NNClusterID != QueryClusterID){
                SumConflict ++;
                BCCluster[NNClusterID].insert(NNVecID);
                BCClusterConflict[NNClusterID].insert(QueryClusterID);
                if (prunevectors){
                    BCClusterPrune[NNClusterID][QueryClusterID].insert(NNVecID);
                }
            }
        }
    }
    std::cout << "The total number of conflict for target NN = " << NLTargetK << " is: " << SumConflict << "\n";
    size_t SumSize = 0;
    for (size_t i = 0; i < nc; i++){SumSize += BCCluster[i].size();}
    std::cout << "The total number of conflict vectors: " << SumSize << "\n";

    /*******************************************************************************************/
    if (prunevectors){pruneBaseSet(BaseBatch, BaseAssignment, Centroids, BCCluster, BCClusterPrune);}
    /*******************************************************************************************/
    std::vector<std::vector<uint32_t>> AssignmentNum(nc);
    std::vector<std::vector<uint32_t>> AssignmentID(nc);
    std::vector<std::vector<uint32_t>> AlphaLineIDs(nc);
    std::vector<std::vector<float>> AlphaLineValues(nc);
    std::vector<std::vector<float>> AlphaLineNorms(nc);

    if (exists(PathNeSTClustering) && !rerunComputeNL){
    }
    else{
        trainAndClusterVectors(BaseBatch, BaseAssignment, BCCluster, BCClusterConflict, AssignmentNum, AssignmentID, AlphaLineIDs, AlphaLineValues, AlphaLineNorms);
        std::ofstream outFile(PathNeSTClustering, std::ios::binary);
        writeVector(outFile, AssignmentNum);
        writeVector(outFile, AssignmentID);
        writeVector(outFile, AlphaLineIDs);
        writeVector(outFile, AlphaLineValues);
        writeVector(outFile, AlphaLineNorms);
        outFile.close();
    }

    /*****************************************************************************************************/ 
    //Compute the vectors not in NeST list
    std::vector<std::vector<uint32_t>> NeSTNonList(nc);
    if (exists(PathNeSTNonList) && !rerunComputeNL){
        ;
    }
    else{
        for (uint32_t i = 0; i < nb; i++){
            if (BCCluster[BaseAssignment[i]].find(i) == BCCluster[BaseAssignment[i]].end()){
                NeSTNonList[BaseAssignment[i]].emplace_back(i);
            }
        }
        std::ofstream outFile(PathNeSTNonList, std::ios::binary);
        writeVector(outFile, NeSTNonList);
    }
}

// Check the proportion of conflict vectors
// Check how many gt vectors are in neighbor list
// We want to know: 
// 1. Total proportion of boundary conflict in gt
// 2. Proportion of boundary conflict gt in neighbor list
// 3. Proportion of boundary conflict gt in hnsw clusters
// 4. Index of the cluster for each gt 
// 5. Distance of the cluster for each gt
// 6. Distance of the neighbor list for each gt (we expect it should be much smaller than the distance to cluster)
void analyzeGT(){
    performance_recorder recorder("analyzeGT");
    std::vector<std::vector<uint32_t>> AssignmentNum(nc);
    std::vector<std::vector<uint32_t>> AssignmentID(nc);
    std::vector<std::vector<uint32_t>> AlphaLineIDs(nc);
    std::vector<std::vector<float>> AlphaLineValues(nc);
    std::vector<std::vector<float>> AlphaLineNorms(nc);

    std::ifstream inFile = std::ifstream(PathNeSTClustering, std::ios::binary);
    readVector(inFile, AssignmentNum);
    readVector(inFile, AssignmentID);
    readVector(inFile, AlphaLineIDs);
    readVector(inFile, AlphaLineValues);
    readVector(inFile, AlphaLineNorms);
    inFile.close();

    std::vector<float> Centroids(nc * Dimension);
    inFile = std::ifstream(PathCentroid, std::ios::binary);
    readXvec<float>(inFile, Centroids.data(), Dimension, nc, false, false);

    std::vector<std::vector<uint32_t>> NeSTNonList(nc);
    inFile = std::ifstream (PathNeSTNonList, std::ios::binary);
    readVector(inFile, NeSTNonList);
    inFile.close();

    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInputStream = std::ifstream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    readXvecFvec<DType>(BaseInputStream, BaseVectors.data(), Dimension, nb, true, true);
    BaseInputStream.close();

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex);
    CenGraph->ef_ = EfQueryEvalaution;

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
   if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInputStream(PathQuery, std::ios::binary);
    if (!QueryInputStream) {std::cerr << "Error opening PathQuery file: " << PathQuery << "\n";exit(1);}
    readXvecFvec<DType>(QueryInputStream, Query.data(), Dimension, nq, true, true);
    GTInputStream.close(); QueryInputStream.close();

    std::vector<uint32_t> BaseAssignment(nb);
    loadNeighborAssignments(PathBaseNeighborID, nb, NeighborNum, BaseAssignment);

    // Check the GT distribution in base index and in NL
    size_t recallk = RecallK[0];
    size_t ef = EfSearch[0];
    std::vector<uint32_t> HNSWID(ef);
    std::vector<float> HNSWDist(ef);

    std::vector<uint32_t> AvgGTClusterth(recallk, 0);
    std::vector<uint32_t> AvgGTClusterNum(recallk, 0);
    std::vector<float> AvgGTClusterDist(recallk, 0);

    std::vector<float> AvgGTNLNum(recallk, 0);
    std::vector<float> AvgGTNLDist(recallk, 0);
    size_t NumConflictgt = 0;     // Total number of conflict NN
    size_t NumNeSTList = 0;
    size_t NumClusterVisited = 0;
    size_t NumConflictinNL = 0;   // Number of conflict NN in NL
    size_t NumConflictinHNSW = 0; // Number of conflict NN in efList
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto searchresult = CenGraph->searchKnn(Query.data() + QueryIdx * Dimension, ef, NumComp);
        for (size_t i = 0; i < ef; i++){
            HNSWID[ef - i - 1] = searchresult.top().second;
            HNSWDist[ef - i - 1] = searchresult.top().first;
            searchresult.pop();
        }

        std::vector<uint32_t> GTClusterth(recallk, 0);
        std::vector<float> GTClusterDist(recallk, 0);
        std::vector<float> GTNLDist(recallk, 0);

        for (size_t i = 0; i < recallk; i++){
            auto gt = GT[QueryIdx * ngt + i];
            auto Assignment = BaseAssignment[gt];
            if (BaseAssignment[gt] == HNSWID[0]){
                continue; // Gt is in the same cluster with query, but we only consider the conflict GT
            }
            NumConflictgt ++;
\
            bool exitFlag = false;
            for (size_t j = 1 ; j < ef && !exitFlag; j++){
                uint32_t ClusterID = HNSWID[j];
                NumNeSTList += AlphaLineIDs[ClusterID].size();
                NumClusterVisited ++;

                for (size_t k = 0; k < AlphaLineIDs[ClusterID].size(); k++){
                    auto QueryClusterID = AlphaLineIDs[ClusterID][k];
                    float Alpha = AlphaLineValues[ClusterID][k];
                    float AlphaNorm = AlphaLineNorms[ClusterID][k];

                    auto StartIndex = k == 0 ? 0 : AssignmentNum[ClusterID][k - 1];
                    auto EndIndex = AssignmentNum[ClusterID][k];

                    assert(AssignmentID[ClusterID].size() >= EndIndex);
                    for (auto Index = StartIndex; Index < EndIndex; Index++){
                        if (AssignmentID[ClusterID][Index] == gt){ // Found gt in the sublist
                            GTClusterth[i] = j; GTClusterDist[i] = HNSWDist[j];
                            NumConflictinHNSW ++;
                            NumConflictinNL ++;
                            float QueryClusterDist = CenGraph->fstdistfunc_(Query.data() + QueryIdx * Dimension, Centroids.data() + QueryClusterID * Dimension, CenGraph->dist_func_param_);
                            float QueryNLDist = (1 - Alpha) * (HNSWDist[j]) + Alpha * QueryClusterDist + AlphaNorm;

                            float CenNNDist = CenGraph->fstdistfunc_(CenGraph->getDataByLabel<float>(HNSWID[i]).data(), CenGraph->getDataByLabel<float>(QueryClusterID).data(), CenGraph->dist_func_param_);
                            float CorrectAlphaNorm = (Alpha - 1) * Alpha * CenNNDist;

                            if (QueryNLDist <=0){
                                std::cout << QueryNLDist << "\n";
                                exit(0);
                            }
                            GTNLDist[i] = QueryNLDist;
                            exitFlag = true;
                            break;
                        }
                    }
                }
                for (size_t k = 0; k < NeSTNonList[ClusterID].size() && !exitFlag; k++){
                    if (NeSTNonList[ClusterID][k] == gt){
                        GTClusterth[i] = j; GTClusterDist[i] = HNSWDist[j];
                        NumConflictinHNSW ++;
                        break;
                    }
                }
            }
        }

        for (size_t i = 0; i < recallk; i++){
            if (GTClusterth[i] > 0){
                AvgGTClusterNum[i] += 1; AvgGTClusterth[i] += GTClusterth[i]; AvgGTClusterDist[i] += GTClusterDist[i];
            }
            if (GTNLDist[i] != 0){
                AvgGTNLNum[i] += 1; AvgGTNLDist[i] += GTNLDist[i];
            }
        }
    }

    std::cout << "Total conflict gt: " << float (NumConflictgt) / nq << " HNSW EfSearch: " << ef << " Num of Conflict gt in HNSW " << float(NumConflictinHNSW) / nq  << " Num of Conflict gt in NL: " << float (NumConflictinNL) / nq << " Proportion of conflict gt in NL: " << float(NumConflictinNL) / NumConflictinHNSW << 
                 " Total number of NeST list: " << float(NumNeSTList) / (NumClusterVisited) << "\n";
    std::cout << std::left << std::setw(20) << "GT-th:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << i + 1; }
    std::cout << " |\n" << std::setw(20) << "Cluster-th:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterth[i]) / (AvgGTClusterNum[i]); }
    std::cout << " |\n" << std::setw(20) << "Num in Cluster:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterNum[i]) / nq; }
    std::cout << " |\n" << std::setw(20) << "Dist to Cluster:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTClusterDist[i]) / (AvgGTClusterNum[i]); }
    std::cout << " |\n" << std::setw(20) << "Num in NList:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTNLNum[i] / nq); }
    std::cout << " |\n" << std::setw(20) << "Dist to NL Cluster:";
    for (size_t i = 0; i < recallk; i++) { std::cout << " | " << std::setw(10) << float(AvgGTNLDist[i] / (AvgGTNLNum[i])); }
    std::cout << " |\n";
    delete CenGraph;

    size_t CentralListSize = 0;
    for (size_t i = 0; i < nc; i++){
        CentralListSize += NeSTNonList[i].size();
    }

    std::cout << "The vectors in NeST list: " << nb - CentralListSize << " The vectors in central list: " << CentralListSize << "\n"; 
}


void evaluateNeighborList(){
    performance_recorder recorder("evaluateNeighborList");
    std::vector<std::vector<uint32_t>> AssignmentNum(nc);
    std::vector<std::vector<uint32_t>> AssignmentID(nc);
    std::vector<std::vector<uint32_t>> AlphaLineIDs(nc);
    std::vector<std::vector<float>> AlphaLineValues(nc);
    std::vector<std::vector<float>> AlphaLineNorms(nc);

    std::ifstream inFile = std::ifstream(PathNeSTClustering, std::ios::binary);
    readVector(inFile, AssignmentNum);
    readVector(inFile, AssignmentID);
    readVector(inFile, AlphaLineIDs);
    readVector(inFile, AlphaLineValues);
    readVector(inFile, AlphaLineNorms);
    inFile.close();

    std::vector<std::vector<uint32_t>> NeSTNonList(nc);
    inFile = std::ifstream (PathNeSTNonList, std::ios::binary);
    readVector(inFile, NeSTNonList);
    inFile.close();

    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInputStream = std::ifstream(PathBase, std::ios::binary);
    if (!BaseInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}
    readXvecFvec<DType>(BaseInputStream, BaseVectors.data(), Dimension, nb, false, false);
    BaseInputStream.close();

    hnswlib::L2Space l2space(Dimension);
    hnswlib::HierarchicalNSW<float> * CenGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathCenGraphIndex);
    CenGraph->ef_ = EfQueryEvalaution;

    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
    if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, false, false);
    std::ifstream QueryInputStream(PathQuery, std::ios::binary);
    if (!QueryInputStream) {std::cerr << "Error opening PathQuery file: " << PathQuery << "\n";exit(1);}
    readXvecFvec<DType>(QueryInputStream, Query.data(), Dimension, nq, false, false);
    GTInputStream.close(); QueryInputStream.close();

    std::cout << "Setting of NLTargetK:  " << NLTargetK << " Use prune or not: " << prunevectors << "\n";
    for (size_t recallidx = 0; recallidx < NumRecall; recallidx++) {
        size_t recallk = RecallK[recallidx];
        std::cout << "Search results for recallk = " << recallk << "\n";

        // Print the table header
        std::cout << "-------------------------------------------------------------------------------------------------------\n";
        std::cout << "| EfSearch | MaxItem | Recall Rate | Recall@1 Rate | Time per Query(us) |    Sum    |  All Potential  |\n";
        std::cout << "-------------------------------------------------------------------------------------------------------\n";
        std::vector<float> QueryDists(nq * recallk);
        std::vector<int64_t>QueryIds(nq * recallk);

    for (size_t searchnumidx = 0; searchnumidx < NumPara; searchnumidx++) {
    size_t ef = EfSearch[searchnumidx];
    size_t maxitem = MaxItem[searchnumidx];

    std::vector<bool> VisitedClusterFlag(nc, false);
    std::vector<float> HNSWClusterDist(nc, 0);
    std::vector<uint32_t>HNSWID(ef);
    size_t SumVisitedItem = 0;
    size_t SumPotentialItem = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        std::vector<uint32_t> NLNNClusterID;
        std::vector<uint32_t> NLQueryClusterIndex;
        std::vector<float> NLQueryNLDist;
        std::vector<uint32_t> FoundCluster;

        faiss::maxheap_heapify(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
        auto SearchResult = CenGraph->searchKnn(Query.data() + QueryIdx * Dimension, ef, NumComp);

        for (uint32_t i = 0; i < ef; i++){
            uint32_t NNClusterID =SearchResult.top().second;
            HNSWID[ef - i - 1] = NNClusterID;
            VisitedClusterFlag[NNClusterID] = true;
            HNSWClusterDist[NNClusterID] = SearchResult.top().first;
            FoundCluster.emplace_back(NNClusterID);
            SumPotentialItem += AssignmentID[NNClusterID].size();
            SearchResult.pop();
        }
        // Search the base cluster
        uint32_t CentralClusterID = HNSWID[0];
        SumVisitedItem += AssignmentID[CentralClusterID].size();
        for (size_t j = 0; j < AssignmentID[CentralClusterID].size(); ++j){
            float Dist = CenGraph->fstdistfunc_(BaseVectors.data() + AssignmentID[CentralClusterID][j] * Dimension, Query.data() + QueryIdx * Dimension, CenGraph->dist_func_param_);
            if (Dist < QueryDists[QueryIdx * recallk]){
                faiss::maxheap_pop(recallk,  QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
                faiss::maxheap_push(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk, Dist, AssignmentID[CentralClusterID][j]);
            }
        }
        SumVisitedItem += NeSTNonList[CentralClusterID].size();
        for (size_t j = 0; j < NeSTNonList[CentralClusterID].size(); ++j){
            float Dist = CenGraph->fstdistfunc_(BaseVectors.data() + NeSTNonList[CentralClusterID][j] * Dimension, Query.data() + QueryIdx * Dimension, CenGraph->dist_func_param_);
            if (Dist < QueryDists[QueryIdx * recallk]){
                faiss::maxheap_pop(recallk,  QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
                faiss::maxheap_push(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk, Dist, NeSTNonList[CentralClusterID][j]);
            }
        }

        // Compute the dist to list centroids
        size_t SumNLNum = 0;
        float SumQueryNLDist = 0;
        for (size_t i = 1; i < ef; i++){
            uint32_t NNClusterID = HNSWID[i];
            for (size_t j = 0; j < AlphaLineIDs[NNClusterID].size(); j++){
                uint32_t QueryClusterID = AlphaLineIDs[NNClusterID][j];
                assert(QueryClusterID < nc);

                if (!VisitedClusterFlag[QueryClusterID]){
                    HNSWClusterDist[QueryClusterID] = CenGraph->fstdistfunc_(Query.data() + QueryIdx * Dimension, CenGraph->getDataByLabel<float>(QueryClusterID).data(), CenGraph->dist_func_param_);
                    VisitedClusterFlag[QueryClusterID] = true;
                }
                float Alpha = AlphaLineValues[NNClusterID][j];
                float AlphaNorm = AlphaLineNorms[NNClusterID][j];
                float QueryNLDist = (1-Alpha) * (HNSWClusterDist[NNClusterID]) + Alpha * (HNSWClusterDist[QueryClusterID]) + AlphaNorm;
                NLNNClusterID.emplace_back(NNClusterID);
                NLQueryClusterIndex.emplace_back(j);
                NLQueryNLDist.emplace_back(QueryNLDist);
                SumQueryNLDist += QueryNLDist;
            }
        }
        float AvgQueryNLDist = SumQueryNLDist / NLNNClusterID.size();


        // Compute the dist to vector in close lists
        for (size_t i = 0; i < NLNNClusterID.size(); i++){
            if (NLQueryNLDist[i] < AvgQueryNLDist){
                uint32_t QueryClusterIndex = NLQueryClusterIndex[i];
                assert (AlphaLineIDs[NLNNClusterID[i]][QueryClusterIndex] <= nc);
                uint32_t NNClusterID = NLNNClusterID[i];
                uint32_t StartIndex = QueryClusterIndex == 0? 0 : AssignmentNum[NNClusterID][QueryClusterIndex - 1];
                uint32_t EndIndex = AssignmentNum[NNClusterID][QueryClusterIndex];
                SumVisitedItem += (EndIndex - StartIndex);

                for (uint32_t Index = StartIndex; Index < EndIndex; Index++){
                    float Dist = CenGraph->fstdistfunc_(BaseVectors.data() + AssignmentID[NNClusterID][Index] * Dimension, Query.data() + QueryIdx * Dimension, CenGraph->dist_func_param_);
                    if(Dist < QueryDists[QueryIdx * recallk]){
                        faiss::maxheap_pop(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk);
                        faiss::maxheap_push(recallk, QueryDists.data() + QueryIdx * recallk, QueryIds.data() + QueryIdx * recallk, Dist, AssignmentID[NNClusterID][Index]);
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    size_t correctCount = 0;
    size_t firstElementCorrectCount = 0;

    for (size_t queryIdx = 0; queryIdx < nq; ++queryIdx){
        std::unordered_set<int64_t> targetSet(GT.begin() + ngt * queryIdx, GT.begin() + ngt * queryIdx + recallk);
        std::unordered_set<int64_t> firstElementSet = {GT[ngt * queryIdx]};

        for (size_t recallIdx = 0; recallIdx < recallk; recallIdx++) {
            int64_t queryId = QueryIds[queryIdx * recallk + recallIdx];
            correctCount += targetSet.count(queryId);
            firstElementCorrectCount += firstElementSet.count(queryId);
        }

        // Ensure the correct count of matching elements is within expected bounds
        if (!(correctCount <= recallk * (queryIdx + 1) && firstElementCorrectCount <= queryIdx + 1)) {
            // Print an error message indicating the mismatched values
            std::cout << "Error: Correct Count = " << correctCount 
                    << ", First Element Correct Count = " << firstElementCorrectCount 
                    << ", Query Index = " << queryIdx << "\n";
                    
            // Exit the program as the inconsistency indicates a problem
            exit(1);
        }
    }

    float recallRateForAllElements = float(correctCount) / (nq * recallk);
    float recallRateForFirstElement = float(firstElementCorrectCount) / nq;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Print the results in rows
    std::cout << "| " << std::setw(8) << ef << " | " << std::setw(7) << maxitem   << " | " << std::setw(11) << recallRateForAllElements
                << " | " << std::setw(13) << recallRateForFirstElement << " | " << std::setw(18) << float(duration.count()) / nq << " | " << std::setw(9) << SumVisitedItem / nq << " | " << std::setw(15) << SumPotentialItem / nq << " |\n";
    }
    }
    delete CenGraph;
}


int main(int argc, char* argv[]) {
    int option;
    while ((option = getopt(argc, argv, "c:n:l:")) != -1) {
        switch (option) {
            case 'c':
                rerunCentroids = std::atoi(optarg) != 0;
                break;
            case 'n':
                rerunComputeNN = std::atoi(optarg) != 0;
                break;
            case 'l':
                rerunComputeNL = std::atoi(optarg) != 0;
                break;
            default:
                std::cerr << "Invalid option\n";
                return EXIT_FAILURE;
        }
    }
    // Output the parsed values (for testing purposes)
    std::cout << "rerunCentroids: " << rerunCentroids << "\n"  << "rerunComputeNN: " << rerunComputeNN << "\n" << "rerunComputeNL: " << rerunComputeNL << "\n";

    PrepareFolder(PathNLFolder.c_str());

    computeCentroids();
    computeCenHNSWGraph();
    computeDatasetNeighbor();
    //testNNGraphQuality();
    computeNN();

    //testMinNumTrain();
    computeNeSTList();

    analyzeGT();
    evaluateNeighborList();
    return 0;
}

