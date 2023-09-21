

/*
        uint32_t DeCompVecID[MaxVecListSize];
        uint32_t DeCompNumTar[MaxVecListSize];
        uint32_t DeCompTarID[MaxTarListSize];
        uint32_t DeCompAlphaTarID[MaxAlphaListSize];

        size_t VecStartIndice = NeighborListVecIndice[nc - 2], VecIndice = NeighborListVecIndice[nc - 1] - NeighborListVecIndice[nc - 2], 
               TarStartIndice = NeighborListNumTarIndice[nc - 2], TarIndice = NeighborListNumTarIndice[nc - 1] - NeighborListNumTarIndice[nc - 2],
               TarIDStartIndice = NeighborListTarIDIndice[nc - 2], TarIDIndice = NeighborListTarIDIndice[nc - 1] - NeighborListTarIDIndice[nc - 2], 
               AlphaStartIndice = NeighborListAlphaIndice[nc - 2], AlphaIndice = NeighborListAlphaIndice[nc - 1] - NeighborListAlphaIndice[nc - 2],
               AlphaCompStartIndice = NeighborListAlphaCompIndice[nc - 2], AlphaCompIndice = NeighborListAlphaCompIndice[nc - 1] - NeighborListAlphaCompIndice[nc - 2];
        // Decompress the data 

        std::cout << "CompStartIndice: " << AlphaCompStartIndice << " CompIndice: " << AlphaCompIndice << "\n";
        exit(0);
        time_recorder TempRecorder = time_recorder();
        // Print info in one cluster for checking

        size_t VecSize = MaxVecListSize;
        codec.decodeArray(NeighborListVec + VecStartIndice, VecIndice, DeCompVecID, VecSize);
        size_t VecIDSize = MaxVecListSize;
        codec.decodeArray(NeighborListNumTar + TarStartIndice, TarIndice, DeCompNumTar, VecIDSize);
        size_t TarListSize = MaxTarListSize;
        codec.decodeArray(NeighborListTarID + TarIDStartIndice, TarIDIndice, DeCompTarID, TarListSize);
        size_t AlphaSize = MaxAlphaListSize;
        codec.decodeArray(NeighborListAlphaTar + AlphaStartIndice, AlphaIndice, DeCompAlphaTarID, AlphaSize);

        TempRecorder.print_time_usage("Decompress the info of a cluster");

        assert(VecSize == VecIDSize);
        uint32_t Indice = 0;
        for (size_t i = 0; i < VecSize; i++){
            std::cout << DeCompVecID[i] << " ";
            std::cout << DeCompNumTar[i] << " ";
            for (size_t j = Indice; j < Indice + DeCompNumTar[i]; j++){
                std::cout << DeCompTarID[j] << " ";
            }
            Indice += DeCompNumTar[i];
            std::cout << " | ";
        }
        std::cout << "\n";

        exit(0);
*/

    bool Examine = false;

    std::ifstream GtInput(PathGt, std::ios::binary);
    std::vector<uint32_t> GTSet(ngt * nq);
    std::vector<float> QuerySet(nq * Dimension);
    QueryInput.seekg(0, std::ios::beg); (PathQuery, std::ios::binary);
    readXvecFvec<DataType> (QueryInput, QuerySet.data(), Dimension, nq, true, true);
    readXvec<uint32_t> (GtInput, GTSet.data(), ngt, nq, true, true);
/*-------------------------------------------------------------------------*//*
Test the angle of between the vector and the centroids:
Query: q; Centroid1: c1; Centroid2: c2

cos(c1qc2): (||c1, q|| + ||c2, q|| - ||c1, c2||) / 2 * (|c1, q| * |c2, q|
cos(qc1c2): (||c1, q|| + ||c1, c2|| - ||c2, q||) / 2 * |c1, q| * |c1, c2|

// Distance between the query and sub-centroid: ||q - (CenN + alpha * (CenT - CenN))|| = ||q - CenN - alpha * (CenT - CenN)|| 
// = ||q - CenN - alpha * CenT + alpha * CenN|| = ||q - alpha * CenT - (1 - alpha) * CenN|| 
// = ||alpha * q - alpha * CenT + (1 - alpha) * q - (1 - alpha) * CenN|| = ||alpha * (q - CenT) + (1 - alpha) * (q - CenN)|| 
// = ||alpha * (q - CenT)|| + ||(1 - alpha) * (q - CenN)|| + 2 * alpha * (1 - alpha) * (q - CenT) * (q - CenN)
// = alpha * alpha * ||q - CenT|| + (1 - alpha) * (1 - alpha) * ||q - CenN|| + 2 * alpha * (1 - alpha) * |q - CenT| * |q - CenN| * (||q - CenT|| + ||q - CenN|| - ||CenT - CenN||) / (|q - CenT| * |q - CenN|)
// = alpha * alpha * ||q - CenT|| + (1 - alpha) * (1 - alpha) * ||q - CenN|| + 2 * alpha * (1 - alpha) * (||q - CenT|| + ||q - CenN|| - ||CenT - CenN||)
// = alpha * alpha * ||q - CenT|| + (1 - alpha) * (1 - alpha) * ||q - CenN|| + 2 * (alpha - alpha * alpha) * (||q - CenT|| + ||q - CenN|| - ||CenT - CenN||)
// = (2 * alpha - alpha * alpha) * ||q - CenT|| + (1 - 2 * alpha + alpha * alpha + 2 * alpha - 2 * alpha * alpha) * ||q - CenN|| - 2 * alpha * (1 - alpha) * ||CenT - CenN||
// = (2 * alpha - alpha * alpha) * ||q - CenT|| + (1 - alpha * alpha) * ||q - CenN|| + 2 * alpha * (alpha - 1) * ||CenT - CenN||


*//*-------------------------------------------------------------------------*/

/*
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::pair<float, std::vector<uint32_t>>>> NeighborList;

    uint32_t PostiveTrue = 0;
    uint32_t NegativeTrue = 0;
    uint32_t SumPositive = 0;
    uint32_t SumNegative = 0;


            // Check the vectors in multiple neighbor lists, search the neighborlist from a close cluster to another far cluster

            uint32_t NLIndice = 0;

            while (!NLQueueList.empty())
            {
                if (VisitedVec > MaxElements){break;}
                VisitedNLNum ++;
                uint32_t TargetClusterIndice = NLQueueList.top().first.first;
                uint32_t NNClusterIndice = NLQueueList.top().first.second;
                float Dist = NLQueueList.top().second;
                uint32_t TargetClusterID = ClusterID[TargetClusterIndice];
                uint32_t NNClusterID = ClusterID[NNClusterIndice];
                if (Examine){
                    std::cout << "\nTargetClusterID: " << TargetClusterID << " NNClusterID: " << ClusterID[NNClusterIndice] << " List Size: " << NLQueueList.size() << "\n";
                    for (size_t temp = 0; temp < NeighborList[TargetClusterID][NNClusterID].second.size(); temp++){
                        std::cout << NeighborList[TargetClusterID][NNClusterID].second[temp] << " ";
                    }
                    std::cout << "\n";
                }
                size_t NeighborListSize = NeighborList[TargetClusterID][NNClusterID].second.size();
                NLIndice += 1;

                float Anglec1qc2 = (ClusterDist[TargetClusterIndice] + ClusterDist[NNClusterIndice] - CentroidsDist[TargetClusterID * nc + NNClusterID]) / (2 * (sqrt(ClusterDist[TargetClusterIndice] * ClusterDist[NNClusterIndice])));
                float Angleqc1c2 = (ClusterDist[TargetClusterIndice] + CentroidsDist[TargetClusterID * nc + NNClusterID] - ClusterDist[NNClusterIndice]) / (2 * (sqrt(ClusterDist[TargetClusterIndice] * CentroidsDist[TargetClusterID * nc + NNClusterID])));
                float Angleqc2c1 = (ClusterDist[NNClusterIndice] + CentroidsDist[TargetClusterID * nc + NNClusterID] - ClusterDist[TargetClusterIndice]) / (2 * (sqrt(ClusterDist[NNClusterIndice] * CentroidsDist[TargetClusterID * nc + NNClusterID])));

                if (Examine){
                    std::cout << "Gt Detected in neighbor list from: " << TargetClusterID << " to: " << NNClusterID <<"\n";
                    std::cout << "The distance between q and c1: " << ClusterDist[TargetClusterIndice] << " " << ", q and c2: " << ClusterDist[NNClusterIndice] << " the distance between c1 and c2: " << CentroidsDist[TargetClusterID * nc + NNClusterID] << "\n"; 
                    std::cout << "The cosine of angle c1qc2: " << Anglec1qc2 << " The cosine of qc1c2: " << Angleqc1c2
                                << " The cosine of angle qc2c1: " << Angleqc2c1
                                << "\n\n";
                }

                if (Angleqc2c1 < 0){SumNegative ++;} else{ SumPositive ++;};
                //if (TargetClusterIndice!= 0  &&   (Anglec1qc2 < 0 || Angleqc2c1 < 0 || Angleqc1c2 < 0)){continue;}

                for(size_t temp = 0; temp < NeighborListSize; temp++){
                    if (VisitedVecSet.count(NeighborList[TargetClusterID][NNClusterID].second[temp]) == 0){

                        VisitedVecSet.insert(NeighborList[TargetClusterID][NNClusterID].second[temp]);
                        if (GT.count(NeighborList[TargetClusterID][NNClusterID].second[temp]) != 0 ){
                            //std::cout << "Gt in NL: " << NeighborList[TargetClusterID][NNClusterID][temp] << " " << j << " " << k << " | ";
                            VisitedGt ++;
                            if (Angleqc2c1 < 0){NegativeTrue ++;} else{ PostiveTrue ++;}
                            //std::cout << " Gt " << NeighborList[TargetClusterID][NNClusterID].second[temp] << " in " << NLIndice << " Dist: " <<  NLQueueList.top().second << " Visited Vecs: " << VisitedVec << " " << " TarBase: " << ClusterID[0] << " Tar: " << TargetClusterID << " NN: " << NNClusterID << " | ";
                        }
                        VisitedNLSize += 1;
                        VisitedVec += 1;
                    }
                }

                std::cout << TargetClusterID << " " << NNClusterID << " " << VisitedGt << " " << NeighborListSize << " " << Dist << " | ";
                //std::cout << "Distance to NL centroid: " << NLQueueList.top().second << " Num of Gt in this NL: " << VisitedGt << " Num of Vec in this NL: " << NeighborListSize << "\n";
                NLQueueList.pop();
            }
            std::cout << VisitedGt << "\n";
*/
    /*---------------------------------------------------------------------------------------------------------------------*/


    std::ifstream BaseNeighborDistInput(PathBaseNeighborDist, std::ios::binary);
    std::ifstream BaseNeighborIDInput(PathBaseNeighborID, std::ios::binary);
    std::vector<uint32_t> BaseNeiID(10 * NeighborNum);
    std::vector<float> BaseNeiDist(10 * NeighborNum);
    BaseNeighborIDInput.read((char *) BaseNeiID.data(), 10 * NeighborNum * sizeof(uint32_t));
    BaseNeighborDistInput.read((char *) BaseNeiDist.data(), 10 * NeighborNum * sizeof(float));

    std::cout << "The NeiID of Base vector: \n";
    for (size_t i = 0; i < 10; i++){
        for (size_t j = 0; j < NeighborNum; j++){
            std::cout << BaseNeiID[i * NeighborNum + j] << " ";
        }
        std::cout << "\n";
        for (size_t j =0; j < NeighborNum; j++){
            std::cout << BaseNeiDist[i * NeighborNum + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    /*---------------------------------------------------------------------------------------------------------------------*/

    std::vector<std::vector<uint32_t>> BaseIds;
    std::vector<uint32_t> BaseAssignment;
    std::cout << "The cluster size: ";
    for (size_t i = 0; i < 10; i++){
        std::cout << BaseIds[i].size() << " ";
    }
    std::cout << "\n";

    std::vector<uint32_t> OriginAssignmentID(nb);
    memcpy(OriginAssignmentID.data(), BaseAssignment.data(), nb * sizeof(uint32_t));  

    // Idea: Assign the base vector with KNN classification: NN and Reverse NN: the maximum cluster number of NN / BeNN

    /*-----------------------------------------------------------------------------------------------------------------------*/

    // Check the boundary conflict and genrate the neighbor list for search
    // When to search the points not in the neighbor list? (After searching all vectors in the neighbor list (except for those with lager dist than threshold))

    // Index with distance to the boundary

    if (!exists(PathNeighborList)){

        // We use the neighborhood structure for neighbor list, try other methods: i.e. distance boundary from small trainset / distance ratio between two clusters
        // Struct: NNClusterID, TargetClusterID, VectorID
        std::map<uint32_t, std::unordered_map<uint32_t, std::set<uint32_t>>> BoundaryConflictMap;

        // Struct: VectorID, TargetClusterID, 
        std::map<uint32_t, std::set<std::pair<uint32_t, uint32_t>>> VectorConflictMap;

        std::ifstream BaseInput(PathBase, std::ios::binary);
        std::ifstream BaseNeighborIDInput(PathBaseNeighborID, std::ios::binary);
        std::ifstream BaseNeighborDistInput(PathBaseNeighborDist, std::ios::binary);
        std::ifstream BaseNNInput(PathDatasetNN, std::ios::binary);

        std::vector<float> BaseBatch(Assign_batch_size * Dimension);
        std::vector<uint32_t> BaseNeighborIDBatch(Assign_batch_size * NeighborNum);
        std::vector<float> BaseNeighborDistBatch(Assign_batch_size * NeighborNum);
        std::vector<uint32_t> BaseNNIDBatch(Assign_batch_size * SearchK);

        for (uint32_t i = 0; i < Assign_num_batch; i++){
            BaseNNInput.read((char *) BaseNNIDBatch.data(), Assign_batch_size * SearchK * sizeof(uint32_t));

            for (uint32_t j = 0; j < Assign_batch_size; j++){
                uint32_t TargetVectorID = i * Assign_batch_size + j;
                uint32_t TargetClusterID = OriginAssignmentID[TargetVectorID];

                for (size_t k = 0; k < NLTargetK; k++){
                    uint32_t NNID = BaseNNIDBatch[j * SearchK + k];
                    uint32_t NNClusterID = BaseAssignment[NNID];

                    if (TargetClusterID != NNClusterID){
                        // Boundary conflict, include this pair to neighbor conflict queue

                        if (BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end()){
                            // There is already boundary conflict recorded on this boundary

                            if (BoundaryConflictMap[NNClusterID].find(TargetClusterID) != BoundaryConflictMap[NNClusterID].end()){
                                BoundaryConflictMap[NNClusterID][TargetClusterID].insert(NNID);
                            }
                            else{
                                BoundaryConflictMap[NNClusterID][TargetClusterID] = std::set<uint32_t>({NNID});
                            }
                        }
                        else{
                            BoundaryConflictMap[NNClusterID] = std::unordered_map<uint32_t, std::set<uint32_t>> ({{TargetClusterID, std::set<uint32_t>({NNID})}});
                        }

                        if (VectorConflictMap.find(NNID) != VectorConflictMap.end()){
                            VectorConflictMap[NNID].insert(std::pair<uint32_t, uint32_t>(OriginAssignmentID[TargetVectorID], BaseAssignment[NNID]));
                        }
                        else{
                            VectorConflictMap[NNID] = {std::pair<uint32_t, uint32_t>(OriginAssignmentID[TargetVectorID], BaseAssignment[NNID])};
                        }

                        // Todo: should we use the reverse indice, i.e. the NN structure in the neighbor list?

/*
                        if (BoundaryConflictMap.find(TargetClusterID) != BoundaryConflictMap.end()){
                            if (BoundaryConflictMap[TargetClusterID].find(NNClusterID) != BoundaryConflictMap[TargetClusterID].end()){
                                BoundaryConflictMap[TargetClusterID][NNClusterID].insert(TargetVectorID);
                            }
                            else{
                                BoundaryConflictMap[TargetClusterID][NNClusterID] = std::set<uint32_t>({TargetVectorID});
                            }
                        }
                        else{
                            BoundaryConflictMap[TargetClusterID] = std::unordered_map<uint32_t, std::set<uint32_t>> ({{NNClusterID, std::set<uint32_t>({TargetVectorID})}});
                        }

                        if (VectorConflictMap.find(TargetVectorID) != VectorConflictMap.end()){
                            VectorConflictMap[TargetVectorID].insert(std::pair<uint32_t, uint32_t>(BaseAssignment[NNID], OriginAssignmentID[TargetVectorID]));
                        }
                        else{
                            VectorConflictMap[TargetVectorID] = {std::pair<uint32_t, uint32_t>(BaseAssignment[NNID], OriginAssignmentID[TargetVectorID])};
                        }
*/

/*
                        if (TargetVectorID == 705448){
                            std::cout << TargetVectorID <<" NNCluster: " <<  NNClusterID << " TargetCluster: " << TargetClusterID << " " << VectorConflictMap[TargetVectorID].size() << "\n";

                            assert(VectorConflictMap[TargetVectorID].count(std::make_pair(NNClusterID, TargetClusterID)) != 0);
                        }
                        if (NNID == 705448){
                            std::cout << NNID << " TargetCluster: " << TargetClusterID <<" NNCluster: " << NNClusterID << " " << VectorConflictMap[NNID].size() << "\n";
                            assert(VectorConflictMap[NNID].count(std::make_pair(TargetClusterID, NNClusterID)) != 0);
                        }
*/
                    }
                }
            }
        }
        TRecorder.print_time_usage("Identify all boundary conflicts");


        // Check the consistence of conflict in conflictmap and vectormap
        /*
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> TestMap;
        for (auto it = VectorConflictMap.begin(); it != VectorConflictMap.end(); it++){

            for (auto IndiceIt = it->second.begin(); IndiceIt != it->second.end(); IndiceIt++){
                uint32_t TargetClusterID = IndiceIt->first;
                uint32_t NNClusterID = IndiceIt->second;
                if (TestMap.find(std::make_pair(TargetClusterID, NNClusterID)) != TestMap.end()){
                    TestMap[std::make_pair(TargetClusterID, NNClusterID)].emplace_back(it->first);
                }
                else{
                    TestMap[std::make_pair(TargetClusterID, NNClusterID)] = {it->first};
                }
            }
        }
        for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end(); NNIt++){
            uint32_t NNClusterID = NNIt->first;
            for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end(); TargetIt++){
                uint32_t TargetClusterID = TargetIt->first;
                if (BoundaryConflictMap[NNClusterID][TargetClusterID].size() != TestMap[std::make_pair(TargetClusterID, NNClusterID)].size()){
                    std::cout << TargetClusterID << " " << NNClusterID << " \n";
                    std::cout << "Vector ID: ";
                    for (size_t i = 0; i < TestMap[std::make_pair(TargetClusterID, NNClusterID)].size(); i++){
                        std::cout << TestMap[std::make_pair(TargetClusterID, NNClusterID)][i] << " ";
                    }
                    std::cout << "\n";
                    
                    std::cout << "The vectors in the boundaryconflict list: " << BoundaryConflictMap[NNClusterID][TargetClusterID].size() << " " << TargetIt->second.size() << "\n";
                    for (auto SetIt = BoundaryConflictMap[NNClusterID][TargetClusterID].begin(); SetIt != BoundaryConflictMap[NNClusterID][TargetClusterID].end(); SetIt++){
                        std::cout << *SetIt << "\n";
                    }
                }
                assert(BoundaryConflictMap[NNClusterID][TargetClusterID].size() == TestMap[std::make_pair(TargetClusterID, NNClusterID)].size());
            }
        }
        exit(0);
        */

/*
        auto NNIt = BoundaryConflictMap.begin(); uint32_t NNClusterID = NNIt->first;
        auto TargetIt = NNIt->second.begin();
        while (TargetIt->second.second.size() < 140)
        {
            if (TargetIt == NNIt->second.end()){
                NNIt ++;
                TargetIt = NNIt->second.begin();
            }
            else{
                TargetIt ++;
            }
        }

        uint32_t TargetClusterID = TargetIt->first;
        std::cout << "Target Cluster: " << TargetClusterID << " NN Cluster: " << NNClusterID << "\n";
        std::cout << "Vectors in the neighbor list:\n";
        float AvgBoundist = 0;
        float AvgTarCenDist = 0;
        uint32_t NLSize = TargetIt->second.second.size();

        for (auto VectorID = TargetIt->second.second.begin(); VectorID != TargetIt->second.second.end(); VectorID++){
            uint32_t BaseVectorID = *VectorID;
            float NNClusterDist = faiss::fvec_L2sqr(BaseSet.data() + BaseVectorID * Dimension, Centroids.data() + NNClusterID * Dimension, Dimension);
            float TargetClusterDist = faiss::fvec_L2sqr(BaseSet.data() + BaseVectorID * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
            float CosNLNNTarget = (NNClusterDist + CentroidsDist[TargetClusterID * nc + NNClusterID] - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidsDist[TargetClusterID * nc + NNClusterID]));
            float DistNLBoundary = sqrt(CentroidsDist[TargetClusterID * nc + NNClusterID]) / 2 - sqrt(NNClusterDist) * CosNLNNTarget;

            if (CosNLNNTarget < 0){
                NLSize --;
                continue;
            }
            AvgBoundist += DistNLBoundary;
            AvgTarCenDist += TargetClusterDist;

            std::cout << BaseVectorID << " NN Dist: " <<  NNClusterDist << " Tar Dist: " << TargetClusterDist << " Cos: " << CosNLNNTarget << " Boundist: " << DistNLBoundary << "\n";
        }
        std::cout << "Avg bound dist: " << AvgBoundist / NLSize << " Avg TargetCluster dist: " << AvgTarCenDist / NLSize << "\n";

        std::cout << "\n\nVectors NOT in the neighbor list:\n";
        uint32_t NonNLSize = BaseIds[NNClusterID].size() - TargetIt->second.second.size();
        float NonAvgBoundist = 0;
        float NonAvgTarCenDist = 0;
        size_t BoundaryLimitSize = 0;
        size_t CenLimitSize = 0;
        size_t BothLimitSize = 0;

        for (size_t i = 0; i < BaseIds[NNClusterID].size(); i++){
            if (TargetIt->second.second.count(BaseIds[NNClusterID][i]) == 0){
                uint32_t BaseVectorID = BaseIds[NNClusterID][i];
                float NNClusterDist = faiss::fvec_L2sqr(BaseSet.data() + BaseVectorID * Dimension, Centroids.data() + NNClusterID * Dimension, Dimension);
                float TargetClusterDist = faiss::fvec_L2sqr(BaseSet.data() + BaseVectorID * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
                float CosNLNNTarget = (NNClusterDist + CentroidsDist[TargetClusterID * nc + NNClusterID] - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidsDist[TargetClusterID * nc + NNClusterID]));
                float DistNLBoundary = sqrt(CentroidsDist[TargetClusterID * nc + NNClusterID]) / 2 - sqrt(NNClusterDist) * CosNLNNTarget;
                std::cout << BaseVectorID << " NN Dist: " <<  NNClusterDist << " Tar Dist: " << TargetClusterDist << " Cos: " << CosNLNNTarget << " Boundist: " << DistNLBoundary << "\n";

                NonAvgBoundist += DistNLBoundary;
                NonAvgTarCenDist += TargetClusterDist;

                if (DistNLBoundary < AvgBoundist / NLSize){
                    BoundaryLimitSize ++;
                }
                if (TargetClusterDist < AvgTarCenDist / NLSize){
                    CenLimitSize ++;
                }
                if (DistNLBoundary < AvgBoundist / NLSize && TargetClusterDist < AvgTarCenDist / NLSize){
                    BothLimitSize ++;
                }
            }
        }
        std::cout << "Avg bound dist: " << NonAvgBoundist / NonNLSize << " Avg TargetCluster dist: " << NonAvgTarCenDist / NonNLSize << "\n";
        std::cout << "Total non NL size: " << NonNLSize << " NlSize: " << NLSize << " BoundaryLimitSize: " << BoundaryLimitSize << " CenLimitSize: " << CenLimitSize << " BothLimitSize: " << BothLimitSize << "\n";
        exit(0);
*/

        // Scan the base set and get the distance boundary (NN vector to the boundary) of the neighbor list
        // Two choices: Use the average boundary distance / Use the average boundary distance and the average centroid distance

        uint32_t NLVectorSize = 0;
        std::vector<float> CentroidsDists;
        // Calcultae the distance that is related to the cluster and boundary
        // Struct: targetclusterID, NNClusterID, SumBoundDist, SumTarCenDist, SumNNCenDist. SumDistRatio
        std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>> BoundaryConflictDistMap;

        for (uint32_t i = 0; i < Assign_num_batch; i++){
            readXvecFvec<float>(BaseInput, BaseBatch.data(), Dimension, Assign_batch_size, true,true);
            BaseNeighborDistInput.read((char *) BaseNeighborDistBatch.data(), Assign_batch_size * NeighborNum * sizeof(float));
            BaseNeighborIDInput.read((char *)BaseNeighborIDBatch.data(), Assign_batch_size * NeighborNum * sizeof(uint32_t));

            for (uint32_t j = 0; j < Assign_batch_size; j++){
                uint32_t VectorID = i * Assign_batch_size + j;
                uint32_t ClusterID = BaseAssignment[VectorID];

                if (VectorConflictMap.find(VectorID) != VectorConflictMap.end()){

                    for (auto IndiceIt = VectorConflictMap[VectorID].begin(); IndiceIt != VectorConflictMap[VectorID].end(); IndiceIt++){
                        uint32_t TargetClusterID = IndiceIt->first;
                        uint32_t NNClusterID = IndiceIt->second;

                        assert(NNClusterID == ClusterID && NNClusterID == BaseNeighborIDBatch[j * NeighborNum]);

                        float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                        float TargetClusterDist = -1;

                        for (size_t temp0 = 1; temp0 < NeighborNum; temp0++){
                            if (BaseNeighborIDBatch[j * NeighborNum + temp0] == TargetClusterID){
                                TargetClusterDist = BaseNeighborDistBatch[j * NeighborNum + temp0];
                                break;
                            }
                        }

                        if (TargetClusterDist < 0){
                            TargetClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
                        }

                        assert(NNClusterDist <= TargetClusterDist);

                        // Compute the distance between the vector and the boundary
                        float CosNLNNTarget = (NNClusterDist + CentroidsDists[TargetClusterID * nc + NNClusterID] - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidsDists[TargetClusterID * nc + NNClusterID]));
                        // Note this is sqrt distance, Todo: we use the boundary or the centroid distance?
                        float DistNLBoundary = sqrt(CentroidsDists[TargetClusterID * nc + NNClusterID]) / 2 - sqrt(NNClusterDist) * CosNLNNTarget;

                        assert(BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end() && BoundaryConflictMap[NNClusterID].find(TargetClusterID) != BoundaryConflictMap[NNClusterID].end());

                        auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);
                        if (BoundaryConflictDistMap.find(ClusterIndice) == BoundaryConflictDistMap.end()){
                            BoundaryConflictDistMap[ClusterIndice] = std::make_pair(1, std::make_tuple(DistNLBoundary, TargetClusterDist, NNClusterDist, (NNClusterDist / TargetClusterDist)));
                        }
                        else{
                            BoundaryConflictDistMap[ClusterIndice].first ++;
                            std::get<0>(BoundaryConflictDistMap[ClusterIndice].second) += DistNLBoundary;
                            std::get<1>(BoundaryConflictDistMap[ClusterIndice].second) += TargetClusterDist;
                            std::get<2>(BoundaryConflictDistMap[ClusterIndice].second) += NNClusterDist;
                            std::get<3>(BoundaryConflictDistMap[ClusterIndice].second) += (NNClusterDist / TargetClusterDist);
                        }

                        NLVectorSize ++;
                    }
                }
            }
        }

        for (auto it = BoundaryConflictDistMap.begin(); it != BoundaryConflictDistMap.end(); it++){
            uint32_t TargetClusterID = it->first.first; uint32_t NNClusterID = it->first.second;
            uint32_t ListSize = BoundaryConflictMap[NNClusterID][TargetClusterID].size();
            std::get<0>(it->second.second) /= ListSize;
            std::get<1>(it->second.second) /= ListSize;
            std::get<2>(it->second.second) /= ListSize;
            std::get<3>(it->second.second) /= ListSize;
        }

        TRecorder.print_time_usage("Compute the boundary threshold distance");
        std::cout << "The total number of vectors in NL: " << NLVectorSize << "\n";

        /*
        uint32_t InsertedSize = 0;
        BaseInput.seekg(0, std::ios::beg); BaseNeighborDistInput.seekg(0, std::ios::beg); BaseNeighborIDInput.seekg(0, std::ios::beg);
        for (uint32_t i = 0; i < Assign_num_batch; i++){
            readXvecFvec<float>(BaseInput, BaseBatch.data(), Dimension, Assign_batch_size, true,true);
            BaseNeighborDistInput.read((char *) BaseNeighborDistBatch.data(), Assign_batch_size * NeighborNum * sizeof(float));
            BaseNeighborIDInput.read((char *)BaseNeighborIDBatch.data(), Assign_batch_size * NeighborNum * sizeof(uint32_t));

            for (uint32_t j = 0; j < Assign_batch_size; j++){
                uint32_t VectorID = i * Assign_batch_size + j;
                uint32_t NNClusterID = BaseAssignment[VectorID];

                if (BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end()){

//#pragma omp parallel for
                    for (auto it = BoundaryConflictMap[NNClusterID].begin(); it != BoundaryConflictMap[NNClusterID].end(); it++){
                        uint32_t TargetClusterID = it->first;
                        if (it->second.count(VectorID) != 0){
                            // Already in the neighbor list
                            continue;
                        }
                        else{
                            float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                            float TargetClusterDist = -1;

                            for (size_t temp = 1; temp < NeighborNum; temp++){
                                if (BaseNeighborIDBatch[j * NeighborNum + temp] == TargetClusterID){
                                    TargetClusterDist = BaseNeighborDistBatch[j * NeighborNum + temp];
                                    break;
                                }
                            }
                            if (TargetClusterDist < 0){
                                TargetClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
                            }

                            if (NNClusterDist > TargetClusterDist){
                                std::cout << i << " " << j << " " << TargetClusterID << " " << NNClusterID << " " << TargetClusterDist << " " << NNClusterDist << "\n";
                                exit(0);
                            }

                            float CosVecNNTarget = (NNClusterDist + CentroidsDist[TargetClusterID * nc + NNClusterID] - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidsDist[TargetClusterID * nc + NNClusterID]));
                            float DistNLBoundary = sqrt(CentroidsDist[TargetClusterID * nc + NNClusterID]) / 2 - sqrt(NNClusterDist) * CosVecNNTarget;

                            auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);

                            if ((NNClusterDist / TargetClusterDist) < std::get<3>(BoundaryConflictDistMap[ClusterIndice].second)){
                                BoundaryConflictMap[NNClusterID][TargetClusterID].insert(VectorID);
                                InsertedSize ++;
                            }
        */

/*
                            if (TargetClusterDist < std::get<1>(BoundaryConflictDistMap[ClusterIndice].second)){
                                BoundaryConflictMap[NNClusterID][TargetClusterID].insert(VectorID);
                                InsertedSize ++;
                            }
*/

                            // The distance between the query and the boundary is smaller than 
                            // the threshold between the distance between the vector and the boundary 
/*
                            if (DistNLBoundary < std::get<0>(BoundaryConflictDistMap[ClusterIndice].second)){
                                BoundaryConflictMap[NNClusterID][TargetClusterID].insert(VectorID);
                                InsertedSize ++;
                            }
*/
/*
                        }
                    }
                }
            }
        }

        TRecorder.print_time_usage("Scan all vector candidates for neighbor list");
        std::cout << "Inserted number of vectors: " << InsertedSize << "\n";
*/
    /*----------------------------------------------------------------------------------------------------------------------*/

    std::vector<float> BaseSet(nb * Dimension);
    BaseInput = std::ifstream(PathBase, std::ios::binary);
    readXvecFvec<DataType>(BaseInput, BaseSet.data(), Dimension, nb, false, false);    

        std::vector<float> NeighborListAlpha;
        uint32_t NLIndice = 0;
        uint32_t NumNeighborList = 0;

        // Project the center of the neighborlist to the line between the target centroid and the NN centroid
        for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end(); NNIt ++){

            uint32_t NNClusterID = NNIt->first;
            NumNeighborList += NNIt->second.size();

            for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end(); TargetIt++){

                uint32_t TargetClusterID = TargetIt->first;

                size_t NeighborListSize = TargetIt->second.size();

                std::vector<float> NeighborListCentroid(Dimension, 0);
                for (auto VectorID = TargetIt->second.begin(); VectorID != TargetIt->second.end(); VectorID++){

                    assert(BaseAssignment[*VectorID] == NNClusterID);
                    for (size_t i = 0; i < Dimension; i++){
                        NeighborListCentroid[i] += BaseSet[(*VectorID) * Dimension + i];
                    }
                }

                for (size_t i = 0; i < Dimension; i++){
                    NeighborListCentroid[i] /= NeighborListSize;
                }

                // Project the neighborlist centroid to the line between two centroids of two clusters: TargetCluster and NNCLuster
                // Let the project point is: p0, the target cluster centroid: p1, the NN cluster centroid: p2, the neighborlist centroid: p3
                // As the neighborlist centroid are in the NN cluster centroid, we compute the alpha: alpha = ||p3, p2|| / ||p1, p2|| and p0 = p2 + alpha * (p1 - p2)
                // cos(A) = (a^2 + c^2 - b^2) / (2 * a * c)
                float DistCenNNCen = faiss::fvec_L2sqr(NeighborListCentroid.data(), Centroids.data() + NNClusterID * Dimension, Dimension);
                float DistCenTarCen = faiss::fvec_L2sqr(NeighborListCentroid.data(), Centroids.data() + TargetClusterID * Dimension, Dimension);
                float DistTarCenNNCen = faiss::fvec_L2sqr(Centroids.data() + TargetClusterID * Dimension, Centroids.data() + NNClusterID * Dimension, Dimension);
                float CosNLNNTarget = (DistCenNNCen + DistTarCenNNCen - DistCenTarCen) / (2 * sqrt(DistCenNNCen * DistTarCenNNCen));

                NeighborListAlpha.emplace_back(sqrt(DistCenNNCen * CosNLNNTarget * CosNLNNTarget / (DistTarCenNNCen)));

                // Record the proportion of distance between NN centroid and the projected point

                if (NeighborListAlpha[NLIndice] > 0.5){
                    std::cout << TargetClusterID << " " << NNClusterID << " " << NLIndice << " " << NeighborListSize << " " << DistCenNNCen << " " << DistCenTarCen << " " << DistTarCenNNCen << " " << CosNLNNTarget << " " << NeighborListAlpha[NLIndice]<< "\n";
                    for (auto iter = TargetIt->second.begin(); iter !=  TargetIt->second.end(); TargetIt++){
                        std::cout << *iter << " ";
                    }
                    std::cout << "\n";
                    auto result = Cengraph->searchKnn(NeighborListCentroid.data(), 1000);
                    for (size_t temp = 0; temp < 1000; temp++){
                        std::cout << result.top().second << " " << result.top().first << " ";
                        result.pop();
                    }
                    std::cout << "\n";
                    exit(0);
                }
                NLIndice ++;
            }
        }
        TRecorder.print_time_usage("Compute the alpha for all neighbor lists");

        // All information for Neighbor List is acquired, save the neighbor list index to disk
        std::ofstream NeighborListOutput(PathNeighborList, std::ios::binary);

        NeighborListOutput.write((char *) & NumNeighborList, sizeof(uint32_t));
        NLIndice = 0;
        float MinAlpha = 1;
        float MaxAlpha = 0;
        uint32_t MaxIndice = 0;
        uint32_t MinIndice = 0;

        for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end(); NNIt++){

            uint32_t NNClusterID = NNIt->first;

            for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end();TargetIt++){

                uint32_t TargetClusterID = TargetIt->first;

                uint32_t NLSize = TargetIt->second.size();
                std::vector<uint32_t> NListCopy (TargetIt->second.begin(), TargetIt->second.end());

                // Sort the indices of items in neighbor list
                NeighborListOutput.write((char *) & TargetClusterID, sizeof(uint32_t));
                NeighborListOutput.write((char *) & NNClusterID, sizeof(uint32_t));
                NeighborListOutput.write((char *) & NLSize , sizeof(uint32_t));
                NeighborListOutput.write((char *) (NeighborListAlpha.data() + NLIndice), sizeof(float));
                NeighborListOutput.write((char *) NListCopy.data(), NLSize * sizeof(uint32_t));
                if (NeighborListAlpha[NLIndice] < MinAlpha){MinAlpha = NeighborListAlpha[NLIndice]; MinIndice = NLIndice;}
                if (NeighborListAlpha[NLIndice] > MaxAlpha){MaxAlpha = NeighborListAlpha[NLIndice]; MaxIndice = NLIndice;}
                NLIndice ++;
            }
        }
        assert(NumNeighborList == NLIndice);
        NeighborListOutput.close();
        std::cout << "The neighborlist size: " << BoundaryConflictMap.size() << "\n";
        std::cout << "Max alpha: " << MaxAlpha << " Indice: " << MaxIndice << " Min alpha: " << MinAlpha << " Min Indice: "<< MinIndice << "\n";
        exit(0);
    }


    //std::cout << "The total number of boundary conflict record: " << NeighborConflictMap.size() << " The total number of centroids: " << nc << "\n";
    //TRecorder.print_record_time_usage(RecordFile, "Train the vectors with neighbor info ");
    float prop = 0;
    //float NumOptimization = 1;

    std::vector<uint32_t> TrainLabels(nt);
    std::vector<float> TrainDists(nt);
    std::vector<std::vector<uint32_t>> VectorOutIDs(nc);
    std::vector<std::vector<uint32_t>> TrainIds(nc);
    size_t KOptimization = 5; assert(KOptimization > 0);
    //auto NeighborConflictMap = neighborkmeans(BaseSet.data(), Dimension, nt, nc, prop, Nlevel, NumOptimization, KOptimization, NeighborNum, Centroids.data(), Verbose, UseOptimize, TrainIds, VectorOutIDs, Lambda, OptSize, UseGraph, 30, TrainLabels.data(), TrainDists.data());

    /***************************************************************************/
/*
    size_t TestSize = nq;
    size_t TestNgt = TargetK;
    size_t TestClusterNum = NeighborNum;
    std::vector<float> TestQuery(TestSize * Dimension);
    std::ifstream TestQueryInput(PathQuery, std::ios::binary);
    readXvec<float> (TestQueryInput, TestQuery.data(),Dimension, TestSize, true, true); 
    hnswlib::HierarchicalNSW * TestGraph = new hnswlib::HierarchicalNSW(Dimension, nt);
    for (size_t i = 0;i < nt; i++){TestGraph->addPoint(TrainSet.data() + i * Dimension);}
    std::vector<uint32_t> TestGt(TestNgt * TestSize);
#pragma omp parallel for
    for (size_t i = 0; i < TestSize; i++){
        auto result = TestGraph->searchKnn(TestQuery.data() + i * Dimension, TestNgt);
        for (size_t j = 0; j < TestNgt; j++){
            TestGt[i * TestNgt + TestNgt - j - 1] = result.top().second;
            result.pop();
        }
    }
    hnswlib::HierarchicalNSW * TestCenGraph = new hnswlib::HierarchicalNSW(Dimension, nc);
    for (size_t i = 0; i < nc; i++){TestCenGraph->addPoint(Centroids.data() + i * Dimension);}
    float SumVisitedGt = 0;
    float SumVisitedVec = 0;

    for (size_t i = 0; i < TestSize; i++){
        std::unordered_set<uint32_t> GT;
        for (size_t j = 0; j < TestNgt; j++){
            GT.insert(TestGt[i * TestNgt + j]);
        }

        size_t VisitedGt = 0;
        size_t VisitedVec = 0;
        auto result = TestCenGraph->searchKnn(TestQuery.data() + i * Dimension, TestClusterNum);
        std::vector<uint32_t> TestCenID(TestClusterNum);
        for (size_t j = 0; j < TestClusterNum; j++){
            TestCenID[TestClusterNum - j - 1] = result.top().second;
            result.pop();
        }

        for (size_t j = 0; j < TestClusterNum; j++){
            uint32_t ClusterID = TestCenID[j];
            for (size_t k = 0; k < TrainIds[ClusterID].size(); k++){
                if (GT.count( TrainIds[ClusterID][k]) != 0 ){
                    VisitedGt++;
                }
            }

            SumVisitedGt += VisitedGt;
            SumVisitedVec += TrainIds[ClusterID].size();

            if (VisitedGt == TestNgt){
                break;
            }
        }
    }
    std::cout << "The number of visited vectors: " << SumVisitedVec / TestSize << " The Recall@" << TestNgt << " : " << SumVisitedGt / TestSize << "\n";
*/
    /***************************************************************************/


    // Select the neighbor list that save the most search cost
    // Struct: <Total cost saved, boundary dist, target cluster, NN cluster>
    auto comp = [](std::tuple<size_t, float, uint32_t, uint32_t> Element1, std::tuple<size_t, float, uint32_t, uint32_t> Element2){return std::get<0>(Element1) < std::get<0>(Element2);};
    std::priority_queue<std::tuple<size_t, float, uint32_t, uint32_t>, std::vector<std::tuple<size_t, float, uint32_t, uint32_t>>, decltype(comp)> NeighborListQueue(comp);

/*
    for (auto it = NeighborConflictMap.begin(); it != NeighborConflictMap.end(); it++){
        size_t NeighborListSize = 0;
        NeighborListQueue.emplace(std::make_tuple(std::get<0>((*it).second)* std::get<2>((*it).second), std::get<1>((*it).second), (*it).first.first, (*it).first.second ));
    }
    TRecorder.print_record_time_usage(RecordFile,"Build neighbor list queue");
*/

/*
    for(size_t i = 0; i < 1000; i++){
        auto result = NeighborListQueue.top();
        std::cout << " | " << std::get<0>(result) << " " <<  std::get<1>(result) << " " <<  std::get<2>(result) << " " <<  std::get<3>(result);
        NeighborListQueue.pop();
    }
    exit(0);
*/
    size_t MaxNumNeighborList = 1000;
    MaxNumNeighborList = MaxNumNeighborList < NeighborListQueue.size() ? MaxNumNeighborList : NeighborListQueue.size();
    std::map<uint32_t, std::vector<std::pair<uint32_t, float>>> NeighborListMapNNCluster;
    for (size_t i = 0; i < MaxNumNeighborList; i++){
        auto EachList = NeighborListQueue.top();
        uint32_t TargetClusterID = std::get<2>(EachList);
        uint32_t NNClusterID = std::get<3>(EachList);

        auto result = NeighborListMapNNCluster.find(NNClusterID);
        if (result != NeighborListMapNNCluster.end()){
            NeighborListMapNNCluster[NNClusterID].emplace_back(std::make_pair(TargetClusterID, std::get<1>(EachList)));
        }
        else{
            NeighborListMapNNCluster[NNClusterID] = {std::make_pair(TargetClusterID, std::get<1>(EachList))};
        }
        NeighborListQueue.pop();
    }
    TRecorder.print_record_time_usage(RecordFile,"Build inverted neighbor list for base vectors");

/*
    for(auto result = NeighborListMapNNCluster.begin(); result != NeighborListMapNNCluster.end(); result++){
        std::cout << " | " << (*result).first  << ": ";
        for (auto it = (*result).second.begin(); it != (*result).second.end(); it++){
            std::cout << (*it).first << " " << (*it).second << " ";
        }
    }
    exit(0);
*/

/*
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> NeighborList;
    // Assignment of the base vectors: find the nearest trainset vectors
    hnswlib::HierarchicalNSW * GraphHNSW = new hnswlib::HierarchicalNSW(Dimension, nc); for (size_t i = 0; i < nc; i++){GraphHNSW->addPoint(Centroids.data() + i * Dimension);}


    std::vector<uint32_t> BaseIDSeq(nb);
    std::vector<std::vector<uint32_t>> BaseNListSeq(nb);
*/
    // Only search two types of train vector in assignment: Vectors in the target cluster, or vectors that are shifted out from the target cluster
    // Firstly search all vectors that are shifted, then the in the target cluster.


    //hnswlib::HierarchicalNSW * TrainHNSW = new hnswlib::HierarchicalNSW(Dimension, nt, M, 2 * M, 2 * EfConstruction); for (size_t i = 0; i < nt; i++){TrainHNSW->addPoint(TrainSet.data() + i * Dimension);}

#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
/*
        auto baseresult = GraphHNSW->searchKnn(BaseSet.data() + i * Dimension, NeighborNum);
        std::vector<uint32_t> VectorClusterID(NeighborNum);
        std::vector<float> VectorClusterDist(NeighborNum);
        for (size_t j = 0; j < NeighborNum; j++){
            VectorClusterID[NeighborNum - j - 1] = baseresult.top().second;
            VectorClusterDist[NeighborNum - j - 1] = baseresult.top().first;
            baseresult.pop();
        }
        uint32_t ClusterID = VectorClusterID[0];
        BaseIDSeq[i] = ClusterID;
*/      
        /*
        float VectorDist = std::numeric_limits<float>::max();
        
        size_t ShiftSize = VectorOutIDs[ClusterID].size();
        // Firstly check all vectors that are shifted out from the target cluster
        for (size_t j = 0; j < ShiftSize; j++){
            uint32_t NNVectorID = VectorOutIDs[ClusterID][j];

            float TrainVectorDist = faiss::fvec_L2sqr(BaseSet.data() + i  * Dimension, TrainSet.data() + NNVectorID * Dimension, Dimension);

            if (TrainVectorDist < VectorDist){
                VectorDist = TrainVectorDist;
                BaseIDSeq[i] = TrainLabels[NNVectorID];
            }
        }

        // If there is one vector in the target cluster that smaller than all vectors in other clusters
        for (size_t j = 0; j < TrainIds[ClusterID].size(); j++){

            float TrainVectorDist = faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, TrainSet.data() + TrainIds[ClusterID][j] * Dimension, Dimension);

            if (TrainVectorDist < VectorDist){
                VectorDist = TrainVectorDist;
                BaseIDSeq[i] = ClusterID;
                break;
            }
        }
        */

        /*
        auto trainresult = TrainHNSW->searchKnn(BaseSet.data() + i * Dimension, 1);
        BaseIDSeq[i] = TrainLabels[trainresult.top().second];
        */

        /*
        // Incldue the neighbor list that is related to the base vector
        auto MapResult = NeighborListMapNNCluster.find(BaseIDSeq[i]);
        if (MapResult != NeighborListMapNNCluster.end()){
            size_t NListSize = NeighborListMapNNCluster[BaseIDSeq[i]].size();
            for (size_t j = 0; j < NListSize; j++){
                uint32_t TargetClusterID = NeighborListMapNNCluster[BaseIDSeq[i]][j].first;
                float TargetBoundaryDist = NeighborListMapNNCluster[BaseIDSeq[i]][j].second;

                float VectorTargetClusterDist = -1;
                for (size_t k = 0; k < NeighborNum; k++){
                    if (VectorClusterID[k] == TargetClusterID){
                        VectorTargetClusterDist = VectorClusterDist[k];
                        break;
                    }
                }
                if (VectorTargetClusterDist < 0){
                    VectorTargetClusterDist = faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, Centroids.data() + TargetClusterID * Dimension, Dimension);
                }

                if (VectorTargetClusterDist < TargetBoundaryDist){
                    BaseNListSeq[i].emplace_back(TargetClusterID);
                }
            }
        }
        */
    }

    TRecorder.print_record_time_usage(RecordFile,"Assign base vectors and the index with neighor list index");

/*
    for(size_t i = 0; i < 1000; i++){
        
        std::cout << i << ": ";
        for (size_t j = 0; j < BaseNListSeq[i].size(); j++){
            std::cout << BaseNListSeq[i][j] << " ";
        }
        std::cout << "|";
    }
    exit(0);
*/

/*
    size_t TotalNumNLVecs = 0;
    for (uint32_t i = 0; i < nb; i++){

        size_t NumVectorList = BaseNListSeq[i].size();
        TotalNumNLVecs += NumVectorList;
        for (size_t j = 0; j < NumVectorList; j++){
            std::pair<uint32_t, uint32_t> NLIndex = std::make_pair(BaseNListSeq[i][j], BaseIDSeq[i]);
            auto result = NeighborList.find(NLIndex);
            if (result != NeighborList.end()){
                NeighborList[NLIndex].emplace_back(i);
            }
            else{
                NeighborList[NLIndex] = {i};
            }
        }
    }
    TRecorder.print_record_time_usage(RecordFile, "Construct neighbor list index");
*/

/*
    size_t SumVectors = 0;
    for (auto it = NeighborList.begin(); it != NeighborList.end(); it++){
        size_t ListSize = (*it).second.size();
        std::cout << "\n|" << (*it).first.first << " " << (*it).first.second << " : ";
        for (size_t i = 0; i < ListSize; i++){
            std::cout << (*it).second[i] << " ";
        }
        SumVectors += ListSize;
    }
    std::cout << "The total number of lists: " << NeighborList.size() << " The total number of vectors in neighbor lists: " << SumVectors << "\n";
    exit(0);


    std::vector<std::vector<uint32_t>> BaseIds(nc);
    for (uint32_t i = 0; i < nb; i++){
        BaseIds[BaseIDSeq[i]].emplace_back(i);
    }
*/
    // Search the index with NL

/*********************************************************************************************************/

/*
    hnswlib::HierarchicalNSW * BaseGraph = new hnswlib::HierarchicalNSW(Dimension, nb); for (size_t i = 0; i < nb; i++){BaseGraph->addPoint(BaseSet.data() + i * Dimension);}
#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        auto result = BaseGraph->searchKnn(QuerySet.data() + i * Dimension, TargetK);
        for (size_t j = 0; j < TargetK; j++){
            GTSet[i * TargetK + TargetK - j - 1] = result.top().second;
            result.pop();
        }
    }

    size_t SumVisitedGt = 0;
    size_t SumVisitedVec = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<uint32_t> GT;
        for (size_t j = 0; j < TargetK; j++){
            GT.insert(GTSet[i * TargetK + j ]);
        }

        size_t VisitedGt = 0;
        size_t VisitedVec = 0;
        auto result = GraphHNSW->searchKnn(QuerySet.data() + i * Dimension, NeighborNum);
        std::vector<uint32_t> ClusterIDs(NeighborNum);
        for (size_t j = 0; j < NeighborNum; j++){
            ClusterIDs[NeighborNum - j - 1] = result.top().second;
            result.pop();
        }

        for (size_t j = 0; j < NeighborNum; j++){
            uint32_t ClusterID = ClusterIDs[j];
            for (size_t k = 0; k < BaseIds[ClusterID].size(); k++){
                if (GT.count( BaseIds[ClusterID][k]) != 0 ){
                    VisitedGt++;
                }
            }

            SumVisitedGt += VisitedGt;
            SumVisitedVec += BaseIds[ClusterID].size();

            if (VisitedGt == TargetK){
                break;
            }
        }
    }
    std::cout << "The number of visited vectors: " << SumVisitedVec / nq << " The Recall@" << TargetK << " : " << SumVisitedGt / (nq * TargetK) << "\n";
*/

/*********************************************************************************************************/

/*
    std::cout << "The neighbor list size: " << MaxNumNeighborList  << " The num of optimization: " << NumOptimization << " The K OPtimization: " << KOptimization << " The K for search: " << TargetK << " The TotalNumNLVecs: " << TotalNumNLVecs << "\n";
    for (size_t ParaIdx = 0; ParaIdx < NumPara; ParaIdx++){
        float SumVisitedGt = 0;
        float SumVisitedVec = 0;
        float SumnNLSize = 0;
        size_t NumCluster = EfSearch[ParaIdx];
        std::cout << "Search " << NumCluster << " number of clusters \n";
        for (size_t i = 0; i < nq; i++){

            std::unordered_set<uint32_t> GT;
            for (size_t j = 0; j < TargetK; j++){
                GT.insert(GTSet[i * TargetK + j]);
            }


            size_t VisitedGt = 0;
            std::vector<uint32_t> ClusterID(NumCluster);
            std::vector<float> ClusterDist(NumCluster);
            auto result = GraphHNSW->searchKnn(QuerySet.data() + i * Dimension, NumCluster);

            for (size_t j = 0; j < NumCluster; j++){
                ClusterID[NumCluster - j - 1] = result.top().second;
                ClusterDist[NumCluster - j - 1] = result.top().first;
                result.pop();
            }

            // Check the vectors in the neighbor cluster (not in the neighbor list) and the vectors in the neighbor list
            uint32_t TargetClusterID = ClusterID[0];
            // For the target cluster, search all vectors in the cluster
            for (size_t j = 0; j < BaseIds[TargetClusterID].size(); j++){
                if (GT.count(BaseIds[TargetClusterID][j]) != 0){
                    VisitedGt ++;
                }
            }
            SumVisitedVec += BaseIds[TargetClusterID].size();
            

            // For the other clusters, if there is neighbor list, only search the vectors in neighbor list
            for (size_t j = 1; j < NumCluster; j++){
                uint32_t NNClusterID = ClusterID[j];
                std::pair<uint32_t, uint32_t> NLIndex = std::make_pair(TargetClusterID, NNClusterID);
                auto result = NeighborList.find(NLIndex);
                if (result != NeighborList.end()){
                    size_t NeighborListSize = (*result).second.size();
                    for (size_t k = 0; k < NeighborListSize; k++){
                        if (GT.count((*result).second[k]) != 0){
                            VisitedGt ++;
                        }
                    }
                    SumnNLSize += NeighborListSize;
                    SumVisitedVec += NeighborListSize;
                }
                else{
                    for (size_t k = 0; k < BaseIds[NNClusterID].size(); k++){
                        if (GT.count(BaseIds[NNClusterID][k]) != 0){
                            VisitedGt ++;
                        }
                    }
                    SumVisitedVec += BaseIds[NNClusterID].size();
                }
            }

            SumVisitedGt += VisitedGt;
        }
        std::cout << "The average Recall of " << TargetK << " nearest neighbors: " << SumVisitedGt / (nq * TargetK) << " Number of clusters: " << NumCluster << " Num of vectors: " << SumVisitedVec / nq << " Num of NL vectors: " << SumnNLSize / nq << "\n";       
    }
*/

}


/*
    std::ifstream QuerySetInput(PathQuery, std::ios::binary);
    std::vector<float> QuerySetTest(nq * Dimension);
    readXvecFvec<float> (QuerySetInput, QuerySetTest.data(), Dimension, nq, true, true);
    std::ifstream QueryGtSetInput(PathGt, std::ios::binary);
    std::vector<uint32_t> QueryGtSetTest(nq * ngt);
    readXvec<uint32_t> (QueryGtSetInput, QueryGtSetTest.data(), ngt, nq, true, true);
    for (size_t i = 0; i < 10; i++){
        std::cout << "Query Cluster ID:";
        uint32_t TestK = 50;
        auto result = Cengraph->searchKnn(QuerySetTest.data() + i * Dimension, TestK);
        std::vector<uint32_t> IDResult(TestK);
        for (size_t j = 0; j < TestK; j++){
            IDResult[TestK - j- 1] = result.top().second;
            result.pop();
        }
        for (size_t j = 0; j < TestK; j++){
            std::cout << IDResult[j] << " ";
        }
        std::cout << "\n";
        std::cout << "QueryGt: ";
        for (size_t j = 0; j < 10; j++){
            std::cout << QueryGtSetTest[i * ngt + j] << " ";
        }
        std::cout << "The Cluster ID: ";
        for (size_t j = 0; j < 10; j++){
            std::cout << BaseAssignment[QueryGtSetTest[i * ngt + j]] << " ";
        } 
        std::cout << "\n"; 
        std::cout << "The BeNN ID :\n";
        for (size_t j = 0; j < 10; j++){
            std::cout << QueryGtSetTest[i * ngt + j] << ": ";
            for (size_t k = 0; k < BaseBeNN[QueryGtSetTest[i * ngt + j]].size(); k++){
                std::cout << BaseBeNN[QueryGtSetTest[i * ngt + j]][k] << " ";
            }
            std::cout << "\nThe BeNN Cluster:";
            for (size_t k = 0; k < BaseBeNN[QueryGtSetTest[i * ngt + j]].size(); k++){
                std::cout << BaseAssignment[BaseBeNN[QueryGtSetTest[i * ngt + j]][k]] << " ";
            }
            std::cout << "\n\n";
        }

        for (size_t temp = 0; temp < 10; temp ++){

        std::cout << "\nThe " << temp << " th NN: ";
        for (size_t j = 0; j < 10; j++){
            std::cout << BaseQueryNNSeq[QueryGtSetTest[i * ngt + temp] * SearchK + j] << " ";
        }
        std::cout << "The Cluster ID: ";
        for (size_t j = 0; j < 10; j++){
            std::cout << BaseAssignment[BaseQueryNNSeq[QueryGtSetTest[i * ngt + temp] * SearchK + j]] << " ";
        } 
        std::cout << "\n"; 
        }
        std::cout << "\n\n";
    }
    exit(0);
*/

/*
    uint32_t NumShift = 0;
    // Optimize the assignment based on the collected information
    for (uint32_t temp0 = 0; temp0 < nb; temp0++){
        for (size_t temp1 = 0; temp1 < TargetK; temp1++){
            uint32_t NNID = BaseQueryNNSeq[temp0 * SearchK + temp1];
            if (BaseAssignment[temp0] != BaseAssignment[NNID]){
                
                std::unordered_map<uint32_t, size_t> NNBeNNClusterIDCount;

                
                //std::cout << "Be NN IDs: ";
                for (size_t j = 0; j < BaseBeNN[NNID].size(); j++){
                    uint32_t BeNNCluserID = BaseAssignment[BaseBeNN[NNID][j].first];
                    //std::cout << BeNNCluserID << " ";
                    if (NNBeNNClusterIDCount.find(BeNNCluserID) != NNBeNNClusterIDCount.end()){
                        NNBeNNClusterIDCount[BeNNCluserID] += float(TargetK - (BaseBeNN[NNID][j].second)) / TargetK;
                    }
                    else{
                        NNBeNNClusterIDCount[BeNNCluserID] = float(TargetK - (BaseBeNN[NNID][j].second)) / TargetK;
                    }
                }


                //std::cout << "NN IDs: ";
                for (size_t j = 0; j < TargetK; j++){
                    uint32_t NNClusterID = BaseAssignment[BaseQueryNNSeq[NNID * SearchK + j]];
                    //std::cout << NNClusterID << " ";
                    if (NNBeNNClusterIDCount.find(NNClusterID) != NNBeNNClusterIDCount.end()){
                        NNBeNNClusterIDCount[NNClusterID] += float(TargetK - (j)) / TargetK;
                    }
                    else{
                        NNBeNNClusterIDCount[NNClusterID] = float(TargetK - (j)) / TargetK;
                    }
                }

                //std::cout << " \nThe accumulation result: \n";
                int MaxResultID = -1; int MaxResultNum = 0;
                for (auto it = NNBeNNClusterIDCount.begin(); it != NNBeNNClusterIDCount.end(); it++){
                    if (it->second > MaxResultNum){
                        MaxResultNum = it->second;
                        MaxResultID = it->first;
                    }
                    //std::cout << it->first << " " << it->second << " | ";
                }

                //std::cout << "Origin Cluster ID: " <<  BaseAssignment[NNID] << "\n";
                if (MaxResultID != BaseAssignment[NNID]){
                    BaseAssignment[NNID] = MaxResultID;
                    NumShift++;
                }
            }
        }
        std::cout << "Processed " << temp0 << " / " << nb << " conflicts\r";
    }
    std::cout << "\nThe total number of shift: " << NumShift << "\n";


    memcpy(BaseAssignment.data(), OriginAssignmentID.data(), nb * sizeof(uint32_t));
*/