#include "NCList.h"

NCListIndex::NCListIndex(const size_t Dimension, const size_t nb, const size_t nc, const size_t nt, const bool Saving, const bool Recording,
    const bool Retrain, const bool UseOPQ, const size_t M_PQ, const size_t CodeBits, size_t NumQuantUnits, size_t NClusterNeighbors, size_t MaxNCList): 
    BIndex(Dimension, nb, nc, nt, Saving, Recording, Retrain, UseOPQ, M_PQ, CodeBits), NumQuantUnits(NumQuantUnits), NClusterNeighbors(NClusterNeighbors), 
    MaxNCList(MaxNCList){}

// The vector location in QuantUnit j in neighbor i: 
// from: NumQuantUnits * NClusterNeighbors + NeighborList[i * NumQuantUnits + j], to: NumQuantUnits * NClusterNeighbors + NeighborList[i * NumQuantUnits + j + 1]
// If j == NumQuantUnits and i = NClusterNeighbors, the end is the end of NeighborCluster

std::string NCListIndex::Eval(std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, size_t NumRecall, size_t NumPara, size_t * RecallK, size_t * MaxItem, size_t * EfSearch, size_t * AccuStopItem){
    // Do the query and evaluate
    std::cout << "Start Evaluate the query" << std::endl;
    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<float>(QueryInput, Query.data(), Dimension, nq, true, true);
    GTInput.close(); QueryInput.close();
    std::string SearchMode = "NonParallel";

    PQTable.resize(PQ->ksub * PQ->M);
    recall_recorder Rrecorder = recall_recorder();
    std::string Result = "";
    QueryNeighborList.resize(1 + NumQuantUnits * NClusterNeighbors + MaxNCList);

    // Cannot change the query in retrieval

    for (size_t i = 0; i < NumRecall; i++){
        for (size_t j = 0; j < NumPara; j++){

            QCDist.resize(EfSearch[j]);
            QCID.resize(EfSearch[j]);

            std::vector<float> QueryDists(nq * RecallK[i]);
            std::vector<int64_t>QueryIds(nq * RecallK[i]);
            std::cout << "Do the retrieval for Recall@" << RecallK[i] << " with MaxItem " << MaxItem[j] << " and EfSearch " << EfSearch[j] << std::endl;
            Trecorder.reset();
            for (size_t k = 0; k < nq; k++){
                Search(RecallK[i], Query.data() + k * Dimension, QueryIds.data() + k * RecallK[i], QueryDists.data() + k * RecallK[i], EfSearch[j], MaxItem[j], AccuStopItem[j]);
            }
            Trecorder.print_time_usage("Search Completed");
            Result +=  "Search Time (ms) :" + std::to_string(Trecorder.getTimeConsumption() / (1000 * nq)) + " EfSearch: " + std::to_string(EfSearch[j]) + " MaxItem: " + std::to_string(MaxItem[j]) + " ";

            size_t Correct = 0;
            size_t Correct1 = 0;
            for (size_t k = 0; k < nq; k++){
                std::unordered_set<int64_t> GtSet;
                std::unordered_set<int64_t> GtSet1;

                for (size_t m = 0; m < RecallK[i]; m++){
                    GtSet.insert(GT[ngt * k + m]);
                    GtSet1.insert(GT[ngt * k]);
                    //std::cout << GT[ngt * k + m] << " ";
                }
                //std::cout << std::endl;

                for (size_t m = 0; m < RecallK[i]; m++){
                    //std::cout << QueryIds[k * RecallK[i] + m] <<" ";
                    if (GtSet.count(QueryIds[k * RecallK[i] + m]) != 0){
                        Correct ++;
                    }
                    if (GtSet1.count(QueryIds[k * RecallK[i] + m]) != 0){
                        Correct1 ++;
                    }
                }
                //std::cout << std::endl;
            }
            float Recall = float(Correct) / (nq * RecallK[i]);
            float Recall1 = float(Correct1) / RecallK[i];
            Result += "Recall" + std::to_string(RecallK[i]) + "@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall) + " Recall@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall1) + "\n";
            Rrecorder.print_recall_performance(nq, Recall, RecallK[i], SearchMode, EfSearch[j], MaxItem[j]);
        }
    }
    return Result;
}


void NCListIndex::NeighborInfo(std::string PathCentroidNeighborID, std::string PathCentroidNeighborDist){
    /*Input Parameter*/
    /*                     */

    // Ensure the number of neighbors considered is larger than the nc setting
    assert(NClusterNeighbors < nc);
    ClusterNeighborID.resize(nc);
    ClusterNeighborDist.resize(nc);
    SqrtClusterNeighborDist.resize(nc);

    for (size_t i = 0; i < nc; i++){
        ClusterNeighborID[i].resize(NClusterNeighbors);
        ClusterNeighborDist[i].resize(NClusterNeighbors);
        SqrtClusterNeighborDist[i].resize(NClusterNeighbors);
    }

    if (!Retrain && exists(PathCentroidNeighborID) && exists(PathCentroidNeighborDist)){
        std::ifstream ClusterNeighborIDInput (PathCentroidNeighborID, std::ios::binary);
        std::ifstream ClusterNeighborDistInput (PathCentroidNeighborDist, std::ios::binary);
        uint32_t TempNCN = 0;
        for (size_t i = 0; i < nc; i++){
            ClusterNeighborIDInput.read((char *) & TempNCN, sizeof(uint32_t)); assert(TempNCN == NClusterNeighbors);
            ClusterNeighborIDInput.read((char *) ClusterNeighborID[i].data(), NClusterNeighbors * sizeof(uint32_t));
            ClusterNeighborDistInput.read((char *) & TempNCN, sizeof(uint32_t)); assert(TempNCN == NClusterNeighbors);
            ClusterNeighborDistInput.read((char *) ClusterNeighborDist[i].data(), NClusterNeighbors * sizeof(float));
        }
    }
    else{

        for (size_t i = 0; i < nc; i++){
            auto result = CentroidHNSW->searchKnn(CentroidHNSW->getDataByInternalId(i), NClusterNeighbors + 1);
            for (size_t j = 0; j < NClusterNeighbors; j++){
                ClusterNeighborID[i][NClusterNeighbors - j - 1] = result.top().second;
                ClusterNeighborDist[i][NClusterNeighbors - j - 1] = result.top().first;
                result.pop();
            }
            assert(result.top().second == i);
            assert(result.top().first == 0);
        }

        for (size_t i = 0; i < nc; i++){
            std::cout << "Cluster " << i<< std::endl;
            for (size_t j = 0; j < NClusterNeighbors; j++){
                std::cout << ClusterNeighborID[i][j] << " " << ClusterNeighborDist[i][j] << " ";
            }
        }

        std::ofstream ClusterNeighborIDOutput(PathCentroidNeighborID, std::ios::binary);
        std::ofstream ClusterNeighborDistOutput (PathCentroidNeighborDist, std::ios::binary);

        /*Save the cluster neighbor info*/
        for (size_t i = 0; i < nc; i++){
            ClusterNeighborIDOutput.write((char *) & NClusterNeighbors, sizeof(uint32_t));
            ClusterNeighborIDOutput.write((char *) ClusterNeighborID[i].data(), NClusterNeighbors * sizeof(uint32_t));
            ClusterNeighborDistOutput.write((char *) & NClusterNeighbors, sizeof(uint32_t));
            ClusterNeighborDistOutput.write((char *) ClusterNeighborDist[i].data(), NClusterNeighbors * sizeof(float));
        }
        ClusterNeighborDistOutput.close(); ClusterNeighborIDOutput.close();
    }

    for (size_t i = 0; i < nc; i++){
        for (size_t j = 0; j < NClusterNeighbors; j++){
            SqrtClusterNeighborDist[i][j] = std::sqrt(ClusterNeighborDist[i][j]);
        }
    }
    
    return;
}


void NCListIndex::Search(size_t K, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem, size_t AccuStopItem){

    Trecorder.reset();
    
    //Trecorder.print_time_usage("");

    if (UseOPQ){
        OPQ->apply(1, Query);
    }

    // Ensure the EfSearch is larger than the NClusterNeighbors
    auto Result =  CentroidHNSW->searchBaseLayer(Query, EfSearch);

    // Todo: Should use int_fast32_t?
    // Todo: Move all initialization into the initialized struct
    for (size_t i = 0; i < EfSearch; i++){
        QCID[EfSearch - i - 1] = Result.top().second;
        QCDist[QCID[EfSearch - i - 1]] = Result.top().first;
        Result.pop();
    }
    uint32_t CentralID = QCID[0];

    // We only check the closest centroid and its neighbors
    // We firstly evaluate the vectors in central clusters
    faiss::maxheap_heapify(K, QueryDists, QueryIds);
    PQ->compute_inner_prod_table(Query, PQTable.data());
    VisitedItem = 0;
    for (size_t i = 0; i < BaseIds[CentralID].size(); i++){
        uint32_t BaseID = BaseIds[CentralID][i];
        float VNorm = BaseNorms[BaseID];
        float ProdDist = 0;
        for (size_t j = 0; j < PQ->code_size; j++){
            ProdDist += PQTable[PQ->ksub + BaseCodes[BaseID * PQ->code_size + j]];
        }
        float Dist = QCDist[CentralID] - CNorms[CentralID] + VNorm  - 2 * ProdDist;
        if (Dist < QueryDists[0]){
            faiss::maxheap_pop(K, QueryDists, QueryIds);
            faiss::maxheap_push(K, QueryDists, QueryIds, Dist, BaseID);
        }
    }
    VisitedItem += BaseIds[CentralID].size(); if(VisitedItem > MaxItem) return;

    SIMDCompressionLib::IntegerCODEC & codec = *SIMDCompressionLib::CODECFactory::getFromName("fastpfor");
    size_t QueryNeighborListSize = QueryNeighborList.size();
    codec.decodeArray(NeighborList[CentralID].data(), NeighborList[CentralID].size(), QueryNeighborList.data(), QueryNeighborListSize);

    float CentralDist = QCDist[0];
    std::priority_queue<BoundQueue> CandidateUnits;
    for (size_t i = 0; i < NClusterNeighbors; i++){
        if (QueryNeighborList[QueryNeighborList[i * NumQuantUnits + 1]] - QueryNeighborList[QueryNeighborList[i * NumQuantUnits]] > 0){
            float QPDist = QCDist[ClusterNeighborID[CentralID][i]] != 0 ? QCDist[ClusterNeighborID[CentralID][i]]: faiss::fvec_L2sqr(Query, CentroidHNSW->getDataByInternalId(ClusterNeighborID[CentralID][i]), Dimension);
            float BoundDist = SqrtClusterNeighborDist[CentralID][i] / 2 - (CentralDist + ClusterNeighborDist[CentralID][i] - QPDist) / (2 * SqrtClusterNeighborDist[CentralID][i]);
            CandidateUnits.emplace(BoundDist, i, 0);
        }
    }

    bool NextUnit = true;
    AccuItem = 0;

    // Todo: Should set a new pointer for the BaseId[Neighbor] before the for loop?
    while(NextUnit){
        auto CurrentUnit = CandidateUnits.top();
        CandidateUnits.pop();
        uint32_t NeighborID = ClusterNeighborID[CentralID][CurrentUnit.NeighborID];
        uint32_t UnitSize = QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID + 1] - QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID];
        AccuItem += UnitSize;
        VisitedItem += UnitSize;

        for (size_t i = QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID]; i < QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID + 1]; i++){
            uint32_t BaseID = BaseIds[NeighborID][QueryNeighborList[i]];
            float VNorm = BaseNorms[BaseID];
            float ProdDist = 0;
            for (size_t j = 0; j < PQ->code_size; j++){
                ProdDist += PQTable[PQ->ksub + BaseCodes[BaseID * PQ->code_size + j]];
            }
            float Dist = QCDist[NeighborID] - CNorms[NeighborID] + VNorm - 2 * ProdDist;
            if (Dist < QueryDists[0]){
                faiss::maxheap_pop(K, QueryDists, QueryIds);
                faiss::maxheap_push(K, QueryDists, QueryIds, Dist, BaseID);
                AccuItem = 0;
            }
        }
        if (AccuItem > AccuStopItem || VisitedItem > MaxItem){
            return;
        }

        // Insert a new quant unit
        if (CurrentUnit.OtherID < NumQuantUnits - 1 && QueryNeighborList[QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID + 2]] - QueryNeighborList[QueryNeighborList[CurrentUnit.NeighborID * NumQuantUnits + CurrentUnit.OtherID + 1]] > 0){
            CandidateUnits.emplace(CurrentUnit.Dist + SqrtClusterNeighborDist[CentralID][CurrentUnit.NeighborID] / (2 * NumQuantUnits), CurrentUnit.NeighborID, CurrentUnit.OtherID);
        }
    }
    return;
}


/**
 * The Math Equations of the Neighbor Cluster Computation, v is in the side of p
 * D(v, p): a; D(v, c): b; D(c, p): c; The mediod of c, p: m
 * The distance between v and the mid line:
 * cos(A) = (a^2 + c^2 - b^2) / (2 * a * c)
 * D(v', m) = c / 2 - a * cos(A) = c / 2 - a * (a^2 + c^2 - b^2) / (2 * a * c) = c / 2 - (a^2 + c^2 - b^2) / (2 * c)
 *  
 **/

void NCListIndex::BuildNList(std::string PathNeighborList, std::string PathBase, std::string PathBaseIDSeq, size_t NumBatch, size_t NumClusterBatch, float Beta){

    NeighborList.resize(nc);

    if (!Retrain && exists(PathNeighborList)){
        std::ifstream NeighborListInput(PathNeighborList, std::ios::binary);
        uint32_t ClusterListLength = 0;
        for (size_t i = 0; i < nc; i++){
            NeighborListInput.read((char *) & ClusterListLength, sizeof(uint32_t));
            NeighborList[i].resize(ClusterListLength);
            NeighborListInput.read((char *) NeighborList[i].data() , ClusterListLength * sizeof(uint32_t));
        }
        return;
    }

    size_t BatchSize = std::ceil(nb / NumBatch);

    std::vector<uint32_t> MaxNClistSize(nc, 0);
    // Two limitations: Beta: the proportion of the vectors in neighbor cluster. MaxNCList: A Solid limitation
    for (size_t i = 0; i < nc; i++){
        for (size_t j = 0; j < NClusterNeighbors; j++){
            MaxNClistSize[i] += BaseIds[ClusterNeighborID[i][j]].size();
        }
        MaxNClistSize[i] = MaxNCList < MaxNClistSize[i] * Beta ? MaxNCList : MaxNClistSize[i] * Beta;
    }
    

    size_t ClusterBatchSize = std::ceil(nc / NumClusterBatch);

    for (size_t ClusterBatchIndice = 0; ClusterBatchIndice < NumClusterBatch; ClusterBatchIndice++){

    std::vector<std::priority_queue<BoundQueue>> NeighborVectorList(ClusterBatchSize);
    std::cout << ClusterBatchIndice << " in " << NumClusterBatch << " Cluster Batches\n";

    for (size_t BaseBatch = 0; BaseBatch < NumBatch; BaseBatch++){
        std::cout << BaseBatch << " in " << NumBatch << " Base Batches\n";
        std::ifstream BaseInput(PathBase, std::ios::binary);
        std::ifstream BaseIDInput(PathBaseIDSeq, std::ios::binary);
        std::vector<float> BaseSetBatch(Dimension * BatchSize);
        std::vector<uint32_t> BaseIDBatch(BatchSize);
        readXvecFvec<DataType>(BaseInput, BaseSetBatch.data(), Dimension, BatchSize, true, true);
        BaseIDInput.read((char *) BaseIDBatch.data(), BatchSize * sizeof(uint32_t));

        for (size_t VectorIndice = 0; VectorIndice < BatchSize; VectorIndice ++){
            // Scan the centrodis in the chosen range and check whether the vector locate in one of its neighbor cluster

            for (size_t CentralIndice = 0; CentralIndice < ClusterBatchSize; CentralIndice ++ ){
                uint32_t CentralID = CentralIndice + ClusterBatchIndice * ClusterBatchSize;
                //std::cout << "Check Cluster\n";
                if (CentralID >= nc){break;}

                // Note the vector locate in the neighbor of the central ID
                for (uint32_t NeighborIndice = 0; NeighborIndice < NClusterNeighbors; NeighborIndice++){
                    //std::cout << "Check neighbor cluster\n";
                    // Find a vector in the neighbor cluster of central cluster

                    if (ClusterNeighborID[CentralID][NeighborIndice] == BaseIDBatch[VectorIndice]){
                        float VCDist = faiss::fvec_L2sqr(BaseSetBatch.data() + VectorIndice * Dimension, CentroidHNSW->getDataByInternalId(CentralID), Dimension);
                        float VPDist = faiss::fvec_L2sqr(BaseSetBatch.data() + VectorIndice * Dimension, CentroidHNSW->getDataByInternalId(BaseIDBatch[VectorIndice]), Dimension);
                        float CPDist = ClusterNeighborDist[CentralID][NeighborIndice];

                        // The dist between vector and the mid line, Note this is the non-square distance
                        // Note this is the sqrt distance
                        float SqrtCPDist = SqrtClusterNeighborDist[CentralID][NeighborIndice];
                        float NSDist = SqrtCPDist / 2 - (VPDist + CPDist - VCDist) / (2 * SqrtCPDist);

                        // Prune the vectors in neighbor clusters for Central Cluster
                        if(NeighborVectorList[CentralIndice].size() < MaxNClistSize[CentralID] || NSDist < NeighborVectorList[CentralIndice].top().Dist){

                            if (NeighborVectorList[CentralIndice].size() >= MaxNClistSize[CentralID]) {
                                NeighborVectorList[CentralIndice].pop();
                            }
                            uint32_t VectorID = VectorIndice + BaseBatch * BatchSize;
                            auto ClusterInnerLoc = std::find(BaseIds[ClusterNeighborID[CentralID][NeighborIndice]].begin(), BaseIds[ClusterNeighborID[CentralID][NeighborIndice]].end(), VectorID);
                            assert(ClusterInnerLoc != BaseIds[ClusterNeighborID[CentralID][NeighborIndice]].end());
                            uint32_t ClusterInnerID = ClusterInnerLoc - BaseIds[ClusterNeighborID[CentralID][NeighborIndice]].begin();
                            // The distance between v' and c is recorded
                            NeighborVectorList[CentralIndice].emplace(BoundQueue(NSDist, NeighborIndice, ClusterInnerID));
                        }
                    }
                    else{
                        continue;
                    }
                }
            }
        }
        BaseIDInput.close(); BaseInput.close(); 
    }

    for (size_t ClusterIndice = 0; ClusterIndice < ClusterBatchSize; ClusterIndice ++){
        uint32_t ClusterID = ClusterIndice + ClusterBatchIndice * ClusterBatchSize; if (ClusterID >= nc){break;}

        assert(NeighborVectorList[ClusterIndice].size() == MaxNClistSize[ClusterID]);
        // Build the ivf-based storage for the selected neighbor vectors
        std::vector<std::vector<std::vector<uint32_t>>> QuantizeListID(NClusterNeighbors);
        std::vector<float> UnitDist(NClusterNeighbors);
        for (size_t NeighborIndice = 0; NeighborIndice < NClusterNeighbors; NeighborIndice++){
            float HalfSqrtCPDist = SqrtClusterNeighborDist[ClusterID][NeighborIndice] / 2;
            UnitDist[NeighborIndice] = HalfSqrtCPDist / NumQuantUnits;

            QuantizeListID[NeighborIndice].resize(NumQuantUnits);
        }

        for (size_t VectorIndice = 0; VectorIndice < MaxNClistSize[ClusterID]; VectorIndice++){
            uint32_t NeighborID = NeighborVectorList[ClusterIndice].top().NeighborID;

            float HalfSqrtCPDist = SqrtClusterNeighborDist[ClusterID][NeighborID] / 2;

            uint32_t QuantizeID = std::floor((NeighborVectorList[ClusterIndice].top().Dist - HalfSqrtCPDist) / UnitDist[NeighborID]);

            // Ensure the ID ranges in 0 ~ NumQuantUnits-1
            QuantizeID = QuantizeID < NumQuantUnits ? QuantizeID : NumQuantUnits-1;
            QuantizeListID[NeighborID][QuantizeID].emplace_back(NeighborVectorList[ClusterIndice].top().OtherID);
        }
        // We use the first part in the NList for indexing and the second part for storage
        // The start and end indice of unit j in neighbor i is: NeighborList[ID][i * NCLusterNeighbor + j] ~ NeighborList[ID][i * NCLusterNeighbor + j + 1]
        size_t QuantSize = 0;
        std::vector<uint32_t> NCListIndex(1 + NumQuantUnits * NClusterNeighbors + MaxNClistSize[ClusterID]);
        for (size_t NeighborIndice = 0; NeighborIndice < NClusterNeighbors; NeighborIndice++){
            for (size_t QuantIndice = 0; QuantIndice < NumQuantUnits; QuantIndice++){
                NCListIndex[NeighborIndice * NumQuantUnits + QuantIndice] = QuantSize + 1 + NeighborIndice * NumQuantUnits;
                memcpy(NCListIndex.data() + QuantSize + 1 + NClusterNeighbors * NumQuantUnits, QuantizeListID[NeighborIndice][QuantIndice].data(), QuantizeListID[NeighborIndice][QuantIndice].size() * sizeof(uint32_t));
                QuantSize += QuantizeListID[NeighborIndice][QuantIndice].size();
            }
        }
        NCListIndex[NClusterNeighbors * NumQuantUnits] = QuantSize;
        assert(QuantSize == MaxNClistSize[ClusterID]);
        
        // Do list compression for billion scale storage

        std::vector<uint32_t> CompressResult(1 + NClusterNeighbors * NumQuantUnits + MaxNClistSize[ClusterID] + 1024);
        size_t CompressedSize = CompressResult.size();
        SIMDCompressionLib::IntegerCODEC & codec = *SIMDCompressionLib::CODECFactory::getFromName("fastpfor");
        codec.encodeArray(NCListIndex.data(), NCListIndex.size(), CompressResult.data(), CompressedSize);
        CompressResult.resize(CompressedSize); CompressResult.shrink_to_fit();

        NeighborList[ClusterID].resize(CompressResult.size());
        memcpy(NeighborList[ClusterID].data(), CompressResult.data(), CompressResult.size() * sizeof(uint32_t));
    }
    }

    if(Saving){
        std::ofstream NeighborListOutput(PathNeighborList, std::ios::binary);
        for (size_t i = 0; i < nc; i++){
            uint32_t ClusterListLength = NeighborList[i].size();
            NeighborListOutput.write((char *) & ClusterListLength, sizeof(uint32_t));
            NeighborListOutput.write((char *) NeighborList[i].data(), ClusterListLength * sizeof(uint32_t));
        }
        NeighborListOutput.close();
    }

    return;
}
