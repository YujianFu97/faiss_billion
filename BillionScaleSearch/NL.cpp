#include "NL.h"
// Define rerun flags
bool rerunCentroids = false;
bool rerunneighbors = true;
bool rerunComputeNN = false;
bool rerunComputeNL = true;

struct cmp{bool operator ()( std::pair<uint32_t, float> &a, std::pair<uint32_t, float> &b){return a.second < b.second;}};

void computeCentroids() {
    if (!rerunCentroids && exists(PathCentroid))
        return;
    performance_recorder recorder = performance_recorder("computeCentroids");
    std::vector<float> Centroids(nc * Dimension);
    std::vector<float> TrainSet(nt_c * Dimension);
    std::ifstream TrainInputStream(PathTrain, std::ios::binary);
    if (!TrainInputStream) {
        std::cerr << "Error opening PathBase file: " << PathTrain << "\n";
        exit(1);
    }
    readXvecFvec<DType>(TrainInputStream, TrainSet.data(), Dimension, nt_c, true, true);
    //faisskmeansClustering(TrainSet.data(), Centroids.data());

    simplekmeans(TrainSet, Dimension, nt_c, nc, Centroids);

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

    std::cout << "The graph M, efcons, efsearch is: " << CenGraph->M_ << " " << CenGraph->ef_construction_ << " " << CenGraph->ef_ << "\n";    

    if (rerunCentroids || rerunneighbors || !exists(PathBaseNeighborID) || !exists(PathBaseNeighborDist)){
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
            readXvecFvec<DType>(BasesetInputStream, BaseBatch.data(), Dimension, Batch_size, false, false);

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
            recorder.print_performance("Completed the " + std::to_string(i) + " th iteration");
        }
        BasesetNeighborIDOutputStream.close();
        BasesetNeighborDistOutputStream.close();
    }

    recorder.print_performance("computeBasesetNeighbor");

    // Compute the neighbor cluster ID for trainset
    if (rerunCentroids || rerunneighbors || !exists(PathTrainNeighborID) || !exists(PathTrainNeighborDist)){
        std::ofstream TrainsetNeighborIDOutputStream(PathTrainNeighborID, std::ios::binary);
        std::ofstream TrainsetNeighborDistOutputStream(PathTrainNeighborDist, std::ios::binary);
        std::ifstream TrainsetInputStream(PathTrain, std::ios::binary);
        if (!TrainsetNeighborIDOutputStream || !TrainsetNeighborDistOutputStream || !TrainsetInputStream) {
            std::cerr << "Error opening PathTrainNeighborID PathTrainNeighborDist PathTrain file: " << "\n";exit(1);}

        std::vector<float> Trainset(nt * Dimension);
        std::vector<uint32_t> TrainsetID(nt * NeighborNum);
        std::vector<float> TrainsetDist(nt * NeighborNum);
        readXvecFvec<DType>(TrainsetInputStream, Trainset.data(), Dimension, nt, false, false);
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
    hnswlib::L2Space l2space(Dimension);

    std::vector<float> BaseBatch(Graph_batch_size * Dimension);
    for (size_t i = 0; i < Graph_num_batch; i++) {
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        //std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        std::ifstream BasesetInputStream(PathBase, std::ios::binary);
        if (!BasesetInputStream) {std::cerr << "Error opening PathBase file: " << PathBase << "\n";exit(1);}

        
        BasesetInputStream.seekg(i * Graph_batch_size * (Dimension * sizeof(DType) + sizeof(uint32_t)), std::ios::beg);
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);
        BasesetInputStream.close();

        hnswlib::HierarchicalNSW<float> * SubGraph = new hnswlib::HierarchicalNSW<float>(&l2space, Graph_batch_size, M, EfConstruction);
        //hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(Dimension, Graph_batch_size, M, 2 * M, EfConstruction);

        #pragma omp parallel for
        for (size_t j = 0; j < Graph_batch_size; j++){
            SubGraph->addPoint(BaseBatch.data() + j * Dimension, j);
        }
        SubGraph->saveIndexNoData(PathSubGraphInfo);
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

    for (size_t i = 0; i < Graph_num_batch; i++){
        std::string PathSubGraphInfo = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".info";
        //std::string PathSubGraphEdge = PathSubGraphFolder + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + "_" + std::to_string(M) + "_" + std::to_string(EfConstruction) + ".edge";

        assert(exists(PathSubGraphInfo));
        readXvecFvec<DType> (BasesetInputStream, BaseBatch.data(), Dimension, Graph_batch_size, false, false);        

        //hnswlib_old::HierarchicalNSW * SubGraph = new hnswlib_old::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, BaseBatch);
        hnswlib::HierarchicalNSW<float> * SubGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathSubGraphInfo, BaseBatch, false, Graph_batch_size, false);
        SubGraph->ef_ = EfNN;

        uint32_t StartIndex = i * Graph_batch_size;
        #pragma omp parallel for
        for (size_t j = 0; j < nt; j++){
            size_t NumComp_ = 0;
            auto result = SubGraph->searchKnn(TrainSet.data() + j * Dimension, SearchK, NumComp_);
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
        recorder.print_performance("Completed " + std::to_string(i + 1) + " th batch");
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
    std::vector<float> tolerateprop_values = {0.02};
    std::vector<size_t> NLTargetK_values = {10};
    std::vector<size_t> Recall = {1};
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
                std::string PathMinTrainIDs = PathNLFolder  + Dataset + "-" + std::to_string(R0) + "-" + std::to_string(nt_NEST) + "-" + std::to_string(NLTargetK) + "-" + std::to_string(MinTrainBCNNNum) + ".id";
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
    size_t MinTrainBCNNNum = 6421;
    std::string PathMinTrainIDs = PathNLFolder  + Dataset + "-" + std::to_string(R0) + "-" + std::to_string(nt_NEST) + "-" + std::to_string(NLTargetK) + "-" + std::to_string(MinTrainBCNNNum) + ".id";

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
        std::cout << "-----------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "| EfSearch | MaxItem | Recall Rate | Recall@1 Rate | Time per Query(us) |    Sum NC    |     Sum    |  All Potential  |\n";
        std::cout << "-----------------------------------------------------------------------------------------------------------------------\n";
        std::vector<float> QueryDists(nq * recallk);
        std::vector<int64_t>QueryIds(nq * recallk);

    for (size_t searchnumidx = 0; searchnumidx < NumPara; searchnumidx++) {
    size_t ef = EfSearch[searchnumidx];
    size_t maxitem = MaxItem[searchnumidx];

    std::vector<bool> VisitedClusterFlag(nc, false);
    std::vector<float> HNSWClusterDist(nc, 0);
    std::vector<uint32_t>HNSWID(ef);
    size_t SumVistedNC = 0;
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
        SumVistedNC += NumComp;

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
        SumPotentialItem += NeSTNonList[CentralClusterID].size();
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
        size_t SumQueryNLSize = 0;
        for (size_t i = 1; i < ef; i++){
            uint32_t NNClusterID = HNSWID[i];
            for (size_t j = 0; j < AlphaLineIDs[NNClusterID].size(); j++){
                uint32_t QueryClusterID = AlphaLineIDs[NNClusterID][j];
                assert(QueryClusterID < nc);

                if (!VisitedClusterFlag[QueryClusterID]){
                    HNSWClusterDist[QueryClusterID] = CenGraph->fstdistfunc_(Query.data() + QueryIdx * Dimension, CenGraph->getDataByLabel<float>(QueryClusterID).data(), CenGraph->dist_func_param_);
                    VisitedClusterFlag[QueryClusterID] = true;
                    SumVistedNC ++;
                }
                float Alpha = AlphaLineValues[NNClusterID][j];
                float AlphaNorm = AlphaLineNorms[NNClusterID][j];
                float QueryNLDist = (1-Alpha) * (HNSWClusterDist[NNClusterID]) + Alpha * (HNSWClusterDist[QueryClusterID]) + AlphaNorm;
                uint32_t StartIndex = j == 0 ? 0 : AssignmentNum[NNClusterID][j - 1];
                uint32_t EndIndex = AssignmentNum[NNClusterID][j];
                NLNNClusterID.emplace_back(NNClusterID);
                NLQueryClusterIndex.emplace_back(j);
                NLQueryNLDist.emplace_back(QueryNLDist);
                SumQueryNLDist += QueryNLDist * (EndIndex - StartIndex);
                SumQueryNLSize += EndIndex - StartIndex;
            }
        }
        float AvgQueryNLDist = SumQueryNLDist / SumQueryNLSize;


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
                << " | " << std::setw(13) << recallRateForFirstElement << " | " << std::setw(18) << float(duration.count()) / nq <<  
                " | " << std::setw(13) << SumVistedNC / nq <<
                " | " << std::setw(9) << SumVisitedItem / nq << " | " << std::setw(15) << SumPotentialItem / nq << " |\n";
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
    exit(0);
    computeNN();
    testNNGraphQuality();
    exit(0);

    testMinNumTrain();
    exit(0);
    computeNeSTList();
    analyzeGT();
    evaluateNeighborList();
    return 0;
}


