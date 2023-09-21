#include "AblationNL.h"

/*
Note: With more neighbor vectors (for exampe, include the NN vectors in NL not just the BeNN vectors, or include the vectors that locate close enough to the boundary / target centroids),
we have better accuracy at long search time
Otherwise, we get better results on small search time

Optimization: 
Build a multi-layer quantization in all neighbor list: record the distance between the vector and the boundary
Quantize the dataset space based on the boundary distance, gradually enlarge the quantization distance, include more vectors until all vectors are included

Final Index of a neighbor list: 
[Number of quantizations in this list, Number of vectors in 1st quantization, XXX in 2nd quantization, ..., XXX in the final quantization]
[[Vector indices in the 1st quantization], [Vector indices in the 2nd quantization], ..., [Vector indices in the final quantization]]


Todo:
1. Accelerate the selection on neighbor list (make it consume least time): Label the vectors in the overlapping area in index: larger index -> faster search
2. Add the integer compression part
3. Add the gt test function for showing the loss from quantization
4. Test the neighbor list with quantization
5. Test the OPQ setting
6. Test the parameter setting on SearchK, BaseSet and Train Set
7. Test the performance on million scale
*/

int main(){
    std::string PathRecord = PathFolder + Dataset + "/RecordNL.txt";
    std::ofstream RecordFile;

    if (Recording){
        RecordFile.open(PathRecord, std::ios::app);
        time_t now = std::time(0); char * dt = ctime(&now);
        RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
        char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
        RecordFile << "Server Node: " << hostname << std::endl;
        RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    time_recorder TRecorder = time_recorder();
    memory_recorder MRecorder = memory_recorder();

    PrepareFolder(PathNLFolder.c_str());
    assert(NLTargetK <= SearchK);

    size_t Datasetsize;
    std::string PathNNDataset;
    bool BaseConflict;
    bool DistPrune;
    std::string PathDatasetNeighborID;
    std::string PathDatasetNeighborDist;

    if (NNDatasetName == "Train"){
        BaseConflict = false;
        DistPrune = true;
        Datasetsize = nt;
        PathNNDataset = PathLearn;
        PathDatasetNeighborID = PathNLFolder + "TrainNeighborID_" +  std::to_string(Datasetsize) + "_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
        PathDatasetNeighborDist = PathNLFolder + "TrainNeighborDist_" +  std::to_string(Datasetsize) + "_" + std::to_string(nc) + "_" + std::to_string(NeighborNum);
    }
    else if(NNDatasetName == "Base"){
        BaseConflict = true;
        DistPrune = true;  // Can change this setting
        Datasetsize = nb;
        PathNNDataset = PathBase;
        PathDatasetNeighborID = PathBaseNeighborID;
        PathDatasetNeighborDist = PathBaseNeighborDist;
    }
    else{
        std::cout << " NN Test dataset error: please choose from Base or Train\n";
        exit(0);
    }

    std::string PathNeighborList = PathNLFolder + "NeighborList_" + std::to_string(Datasetsize) + "_" + std::to_string(nc) + "_" + std::to_string(NLTargetK) + "_" + "_" + NNDatasetName + "_" + std::to_string(DistPrune) + "_" + std::to_string(UseQuantize);
    std::string PathNeighborListInfo = PathNeighborList + "_info";
    
    if (false){
        PathBaseIDSeq = "/data/yujian/Dataset/SIFT1B/precomputed_idxs_sift1b.ivecs";
        PathBaseIDInv = "/data/yujian/Dataset/SIFT1B/precomputed_idxs_sift1b.inv";

        std::cout << "Loading the baseid structure \n";

        std::vector<uint32_t> BaseAssignment(nb);
        std::vector<std::vector<uint32_t>> BaseIds(nc);

        std::ifstream BaseIDInput(PathBaseIDSeq, std::ios::binary);
        readXvec<uint32_t>(BaseIDInput, BaseAssignment.data(), 1000000, nb / 1000000, false, true);
        std::cout << "Generate the baseid index\n";

        /*
            std::vector<uint32_t> ClusterSize(nc, 0);
            std::vector<uint32_t> VectorIndice(nb, 0);
            for (size_t i = 0; i < nb; i++){
                std::cout << i << " / " << nb << "\r";
                VectorIndice[i] = ClusterSize[BaseAssignment[i]];
                ClusterSize[i] ++;
            }

            for (size_t i = 0; i < nc; i++){
                BaseIds[i].resize(ClusterSize[i]);
            }
        */
        for (uint32_t i = 0; i < nb; i++){
            assert(BaseAssignment[i] < nc);

            if ((i % 100) == 0) std::cout << float(i) /  float(nb) << " Completed\r";

            //BaseIds[BaseAssignment[i]][VectorIndice[i]] = i;
            BaseIds[BaseAssignment[i]].emplace_back(i);
        }

        std::ofstream BaseIDInvOutput(PathBaseIDInv, std::ios::binary);

        for (size_t i = 0; i < nc; i++){
            uint32_t ClusterSize = BaseIds[i].size();
            BaseIDInvOutput.write((char *) & ClusterSize, sizeof(uint32_t));
            BaseIDInvOutput.write((char *) BaseIds[i].data(), sizeof(uint32_t) * ClusterSize);
        }
        BaseIDInvOutput.close();
    }

    /*-----------------------------------------------------------------------------------------------------------------*/
    // Train the centroids for inverted index
    PathCentroid = PathNLFolder + "Centroids_" + std::to_string(nc) + ".fvecs";
    //PathCentroid = PathFolder  + Dataset + "/" + "centroids_sift1b.fvecs";

    std::vector<float> Centroids(nc * Dimension);
    if (!exists(PathCentroid)){
        std::vector<float> TrainSet(nt * Dimension);
        std::ifstream TrainInput(PathBase, std::ios::binary);
        readXvec<float>(TrainInput, TrainSet.data(), Dimension, nt, true, true);
        hierarkmeans(TrainSet.data(), Dimension, nt, nc, Centroids.data(), Nlevel, UseOptimize, UseGraph, OptSize, Lambda);

        std::ofstream CentroidsOuput(PathCentroid, std::ios::binary);
        uint32_t D = Dimension;
        for (size_t i = 0; i < nc; i++){
            CentroidsOuput.write((char *) & D,  sizeof(uint32_t));
            CentroidsOuput.write((char *) (Centroids.data() + i * Dimension), Dimension * sizeof(float));
        }
    }
    else{
        std::ifstream CentroidInput(PathCentroid, std::ios::binary);
        readXvecFvec<float>(CentroidInput, Centroids.data(), Dimension, nc, true, true);
    }

/*
    std::vector<float> CNorms(nc);
    if (exists(PathCentroidNorm)){
        std::ifstream CentroidNormInput(PathCentroidNorm, std::ios::binary);
        CentroidNormInput.read((char * )CNorms.data(), nc * sizeof(float));
    }
    else{
        std::ofstream CentroidNormOutput(PathCentroidNorm, std::ios::binary);
        faiss::fvec_norms_L2sqr(CNorms.data(), Centroids.data(), Dimension, nc);
        CentroidNormOutput.write((char * ) CNorms.data(), nc * sizeof(float));
        CentroidNormOutput.close();
    }
*/

    // Assign the base set, and optimize the assignment with minimizing the search cost:
    hnswlib::HierarchicalNSW * Cengraph;
    if (exists(PathCenGraphInfo) && exists(PathCenGraphEdge)){
        Cengraph = new hnswlib::HierarchicalNSW(PathCenGraphInfo, PathCentroid, PathCenGraphEdge);
    }
    else{
        Cengraph = new hnswlib::HierarchicalNSW(Dimension, nc, M, 2*M, EfConstruction);
        for (size_t i = 0; i < nc; i++){Cengraph->addPoint(Centroids.data() + i * Dimension);}
        Cengraph->SaveEdges(PathCenGraphEdge);
        Cengraph->SaveInfo(PathCenGraphInfo);
    }

    std::string PathBaseIDSeq    = PathNLFolder + "BaseID_nc" + NCString + "_Seq";
    std::cout << "Check whether the neighbor id exists: " << exists(PathBaseNeighborID) << " " << exists(PathBaseNeighborDist) << " " << exists(PathBaseIDSeq) << " ";
    std::cout << PathBaseNeighborID << "\n" << PathBaseNeighborDist << "\n" << PathBaseIDSeq << "\n";
    //exit(0);

    /*
        if (!exists(PathBaseNeighborID) || !exists(PathBaseNeighborDist) || !exists(PathBaseIDSeq)){
            std::cout << "Assign the base vectors\n";
            std::ifstream BaseInput = std::ifstream(PathBase, std::ios::binary);
            std::ofstream BaseNeighborIDOutput(PathBaseNeighborID, std::ios::binary);
            std::ofstream BaseNeighborDistOutput(PathBaseNeighborDist, std::ios::binary);
            std::ofstream BaseIDSeqOutput(PathBaseIDSeq, std::ios::binary);

            std::vector<float> BaseAssignBatch(Assign_batch_size * Dimension);
            std::vector<uint32_t> BaseNeighborClusterIDBatch(Assign_batch_size * NeighborNum);
            std::vector<float> BaseNeighborClusterDistBatch(Assign_batch_size * NeighborNum);

            for (size_t i = 0; i < Assign_num_batch; i++){
                readXvecFvec<DataType> (BaseInput, BaseAssignBatch.data(), Dimension, Assign_batch_size, true, true);

                #pragma omp parallel for
                for (size_t j = 0; j < Assign_batch_size; j++){
                    auto result = Cengraph->searchKnn(BaseAssignBatch.data() + j * Dimension, NeighborNum);
                    for (size_t k = 0; k < NeighborNum; k++){
                        BaseNeighborClusterIDBatch[j * NeighborNum + NeighborNum - k - 1] = result.top().second;
                        BaseNeighborClusterDistBatch[j * NeighborNum + NeighborNum - k - 1] = result.top().first;
                        result.pop();
                    }
                }
                BaseNeighborIDOutput.write((char *) BaseNeighborClusterIDBatch.data(), Assign_batch_size * NeighborNum * sizeof(uint32_t));
                BaseNeighborDistOutput.write((char *) BaseNeighborClusterDistBatch.data(), Assign_batch_size * NeighborNum * sizeof(float)); 
                for (size_t j = 0; j < Assign_batch_size; j++){
                    BaseIDSeqOutput.write((char *) & BaseNeighborClusterIDBatch[j * NeighborNum], sizeof(uint32_t));
                }
                TRecorder.print_time_usage("Assign the " + std::to_string(i + 1) + " th batch in " + std::to_string(Assign_num_batch) + " total batches");
            }
            BaseNeighborIDOutput.close();
            BaseNeighborDistOutput.close();
        }
    */
    /*------------------------------------*/
    bool NeighborTest = false;
    if (NeighborTest){
        std::cout << "Test the vector cost on neighboring partition\n";
        //size_t nq = 10;

        std::vector<float> Query (nq * Dimension);
        std::vector<uint32_t> GT(nq * ngt);
        std::ifstream GTInput(PathGt, std::ios::binary);
        readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
        std::ifstream QueryInput(PathQuery, std::ios::binary);
        readXvecFvec<DataType>(QueryInput, Query.data(), Dimension, nq, true, true);
        GTInput.close(); QueryInput.close();
        size_t Ef = 10000;

        std::vector<uint32_t> BaseAssignment(nb);
        std::vector<std::vector<uint32_t>> BaseIds(nc);

        PathBaseIDSeq = PathFolder  + Dataset + "/" + "precomputed_idxs_sift1b.ivecs";
        PathBaseIDInv = PathFolder + Dataset + "/" + "precomputed_idxs_sift1b.inv";

        std::cout << "Loading the baseid structure \n";
        if (!exists(PathBaseIDInv)){

            std::ifstream BaseIDInput(PathBaseIDSeq, std::ios::binary);
            readXvec<uint32_t>(BaseIDInput, BaseAssignment.data(), 1000000, nb / 1000000, false, true);
            std::cout << "Generate the baseid index\n";

            std::vector<uint32_t> ClusterSize(nc, 0);
            std::vector<uint32_t> VectorIndice(nb, 0);
            for (size_t i = 0; i < nb; i++){
                std::cout << i << " / " << nb << "\r";
                VectorIndice[i] = ClusterSize[BaseAssignment[i]];
                ClusterSize[i] ++;
            }

            for (size_t i = 0; i < nc; i++){
                BaseIds[i].resize(ClusterSize[i]);
            }

            for (uint32_t i = 0; i < nb; i++){
                //assert(BaseAssignment[i] < nc);
                std::cout << i << " / " << nb << "\r";
                BaseIds[BaseAssignment[i]][VectorIndice[i]] = i;
            }

            std::ofstream BaseIDInvOutput(PathBaseIDInv, std::ios::binary);

            for (size_t i = 0; i < nc; i++){
                uint32_t ClusterSize = BaseIds[i].size();
                BaseIDInvOutput.write((char *) & ClusterSize, sizeof(uint32_t));
                BaseIDInvOutput.write((char *) BaseIds[i].data(), sizeof(uint32_t) * ClusterSize);
            }
            BaseIDInvOutput.close();
        }
        else{
            std::ifstream BaseIDInvInput(PathBaseIDInv, std::ios::binary);

            uint32_t ClusterSize = 0;
            for (size_t i = 0; i < nc; i++){
                if ((i % 1000) == 0) std::cout << i << " / " << nc << "\r";
                BaseIDInvInput.read((char *) & ClusterSize, sizeof(uint32_t));
                BaseIds[i].resize(ClusterSize);
                BaseIDInvInput.read((char *) BaseIds[i].data(), ClusterSize * sizeof(uint32_t));
            }
        }


        for (size_t i = 0; i < 100; i++){
            std::cout << BaseIds[nc - 1 - i].size() << " ";
        }
        std::cout << "\n";

        std::cout << "Start the evaluation\n";
        TRecorder.reset();

        for (size_t KInEval = 1; KInEval <= 10; KInEval++){
        
            std::vector<std::pair<uint32_t, uint32_t>> VectorCost(nq, std::pair<uint32_t, uint32_t>{0, 0}); 

            //std::vector<uint32_t> GtNum(nq, 0);

            #pragma omp parallel for
            for (size_t i = 0; i < nq; i++){
                //std::cout << "Processing " << i << " / " << nq << "\r";
                size_t VisitedGt = 0;
                size_t VisitedVec = 0;

                std::vector<uint32_t> QCID(Ef);
                std::vector<float> QCDist(Ef);
                std::unordered_set<uint32_t> QueryGT;
                for (size_t j = 0; j < KInEval; j++){
                    QueryGT.insert(GT[i * ngt + j]);
                }

                auto result = Cengraph->searchBaseLayer(Query.data() + i * Dimension, Ef);
                for (size_t j = 0; j < Ef; j++){
                    QCID[Ef - j - 1] = result.top().second;
                    QCDist[Ef - j - 1] = result.top().first;
                    result.pop();
                    //std::cout << QCID[Ef - j - 1] << " ";
                }

                //std::cout << "Visiting the partitions\n";
                VectorCost[i].first = BaseIds[QCID[0]].size();
                for (size_t j =0; j < Ef; j++){
                    //std::cout << j << "/ " << Ef << "\r";
                    uint32_t ClusterID = QCID[j];

                    VisitedVec += BaseIds[ClusterID].size();
                    for (size_t k = 0; k < BaseIds[ClusterID].size(); k++){
                        if (QueryGT.find(BaseIds[ClusterID][k]) != QueryGT.end()){
                            VisitedGt ++;
                        }
                    }
                    assert(VisitedGt <= KInEval);
                    if (VisitedGt == KInEval){
                        break;
                    }

                    //GtNum[i] = VisitedGt;
                    //break;
                }
                VectorCost[i].second = VisitedVec;

            //std::cout << VectorCost[i].first << " " << VectorCost[i].second << " " << BaseIds[QCID[0]].size() << " " << VisitedVec <<"\n";
            }
            /*
                    float TotalGt = 0;
                    for (size_t i = 0; i < nq; i++){
                        TotalGt += GtNum[i];
                    }
                    std::cout << "Recall in central partition: " << float(TotalGt) / float(nq * KInEval) << " for K = " << KInEval << "\n";
                    continue;
            */
            float Ratio = 0;
            for (size_t i = 0; i < nq; i++){
                //std::cout << VectorCost[i].second << " " << VectorCost[i].first << " | ";
                Ratio += float(VectorCost[i].second - VectorCost[i].first) / VectorCost[i].second;
            }
            std::cout << "The ratio for K = " << KInEval << " is: " << Ratio / nq << "\n";
            TRecorder.print_time_usage("");
        }
        exit(0);
    }
    /*------------------------------------*/


    /*---------------------------------------------------------------------------------------------------------------------*/
    std::vector<float> CentroidsNorm(nc);

    if (! exists(PathCentroidNLNorm)){
        std::cout << "Computing the distance between centroid vectors \n";
#pragma omp parallel for
        for (size_t i = 0; i < nc; i++){
            CentroidsNorm[i] = faiss::fvec_norm_L2sqr(Cengraph->getDataByInternalId(i), Dimension);
        }
        std::ofstream CentroidDistOutput(PathCentroidNLNorm, std::ios::binary);
        CentroidDistOutput.write((char *) CentroidsNorm.data(), nc * sizeof(float));
        CentroidDistOutput.close();
    }
    else{
        std::ifstream CentroidDistInput(PathCentroidNLNorm, std::ios::binary);
        CentroidDistInput.read((char *) CentroidsNorm.data(), nc * sizeof(float));
        CentroidDistInput.close();
    }


    /*---------------------------------*/
    //ComputeNN(SearchK, Datasetsize, TRecorder, MRecorder, PathSubGraphFolder, PathNNDataset, PathDatasetNN);

    /*---------------------------------------------------------------------------------------------------------------------*/

    BuildNeighborList(SearchK, Assign_num_batch, Datasetsize, NeighborNum, NLTargetK, BaseConflict, DistPrune, UseQuantize, QuantBatchSize, Cengraph,
                        TRecorder, PathDatasetNN, PathNeighborList, PathNeighborListInfo, PathNNDataset, PathDatasetNeighborID, PathDatasetNeighborDist, PathBaseNeighborID, PathBaseNeighborDist);

/*-------------------------------------------------------------------------------------*/
    /*
    faiss::ProductQuantizer * PQ = new faiss::ProductQuantizer;
    faiss::LinearTransform * OPQ = new faiss::LinearTransform;
    std::vector<uint8_t> BaseCodes;
    std::vector<float> BaseNorms;
    TrainQuantizer(TRecorder, PQTrainSize, PathLearn, PathPQ, PathOPQ, Cengraph);
    PQ = faiss::read_ProductQuantizer(PathPQ.c_str());
    if (UseOPQ)OPQ = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(PathOPQ.c_str()));
    QuantizeBaseset(Assign_num_batch, Assign_batch_size, PathBase, PathBaseIDSeq, PathBaseCode, PathBaseNorm, PathOPQCentroids, BaseCodes, BaseNorms, Cengraph, PQ, OPQ);
    */
/*----------------------------------------------------------------------------------------*/

    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::pair<std::vector<uint32_t>, std::vector<float>>>> QuanNeighborList;
    // The Alpha value of neighbor lists
    float * NeighborListAlpha = nullptr;
    float * NeighborListAlphaNorm = nullptr;
    // The target cluster ID of neighbor lists
    uint32_t * NeighborListAlphaTar = nullptr;
    // 
    uint32_t * NeighborListAlphaIndice = nullptr;
    // The end indice of each comp neighor list ( not included )
    uint32_t * NeighborListAlphaCompIndice = nullptr;
    // The vector ID in neighbor list of multiple clusters
    uint32_t * NeighborListVec = nullptr;
    // The end idnice of each neighbor list
    uint32_t * NeighborListVecIndice = nullptr;
    // The number of target cluster of different vectors
    uint32_t * NeighborListNumTar = nullptr;
    // The indice of number of target cluster
    uint32_t * NeighborListNumTarIndice = nullptr;
    //
    uint32_t * NeighborListTarID = nullptr;
    //
    uint32_t * NeighborListTarIDIndice = nullptr;

    uint32_t NumAlphaList, NumVecID, NumTarList, MaxVecListSize, MaxTarListSize, MaxAlphaListSize, 
            CompNumAlpha, CompNumVecID, CompNumTar, CompTarID;

    FastPForLib::CODECFactory factory;
    FastPForLib::IntegerCODEC & codec = * factory.getFromName("simdfastpfor128");

    if (UseQuantize){
        uint32_t NumNeighborList = 0;
        std::ifstream NeighborListInput(PathNeighborList, std::ios::binary);
        NeighborListInput.read((char *) & NumNeighborList, sizeof(uint32_t));
        std::cout << "The number of neighbor list is: " << NumNeighborList << "\n";
        for (size_t i = 0; i < NumNeighborList; i++){
            uint32_t TargetClusterID, NNClusterID, IDSize, AlphaSize;
            NeighborListInput.read((char *) & TargetClusterID, sizeof(uint32_t));
            NeighborListInput.read((char *) & NNClusterID, sizeof(uint32_t));
            NeighborListInput.read((char *) & IDSize, sizeof(uint32_t));
            NeighborListInput.read((char *) & AlphaSize, sizeof(uint32_t));

            if (QuanNeighborList.find(TargetClusterID) != QuanNeighborList.end()){
                QuanNeighborList[TargetClusterID][NNClusterID] = std::make_pair(std::vector<uint32_t>(), std::vector<float>());
            }
            else{
                QuanNeighborList[TargetClusterID] = std::unordered_map<uint32_t, std::pair<std::vector<uint32_t>, std::vector<float>>>();
                QuanNeighborList[TargetClusterID][NNClusterID] =std::pair<std::vector<uint32_t>, std::vector<float>>();
            }

            QuanNeighborList[TargetClusterID][NNClusterID].first.resize(IDSize);
            QuanNeighborList[TargetClusterID][NNClusterID].second.resize(AlphaSize);
            NeighborListInput.read((char *) QuanNeighborList[TargetClusterID][NNClusterID].first.data(), IDSize * sizeof(uint32_t));
            NeighborListInput.read((char *) QuanNeighborList[TargetClusterID][NNClusterID].second.data(), AlphaSize * sizeof(float));
        }
        TRecorder.print_time_usage("Load the neighbor list index with quantization");
    }
    else{
        // struct:  targetclusterID,     NNclusterID,      TargetVecNNclusterDist   vector list on one boundary
        // Load the neighbor list for futher search

        std::ifstream NeighborListInput(PathNeighborList, std::ios::binary);
        std::ifstream NeighborListInfoInput(PathNeighborListInfo, std::ios::binary);

        NeighborListInfoInput.read((char *) & NumAlphaList, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & NumVecID, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & NumTarList, sizeof(uint32_t));

        NeighborListInfoInput.read((char *) & MaxVecListSize, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & MaxTarListSize, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & MaxAlphaListSize, sizeof(uint32_t));

        NeighborListInfoInput.read((char *) & CompNumAlpha, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & CompNumVecID, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & CompNumTar, sizeof(uint32_t));
        NeighborListInfoInput.read((char *) & CompTarID, sizeof(uint32_t));

        // The Alpha value of neighbor lists, not compressed
        NeighborListAlpha = new float[NumAlphaList];
        NeighborListAlphaNorm = new float[NumAlphaList];
        // The target cluster ID of neighbor lists
        NeighborListAlphaTar = new uint32_t[CompNumAlpha];
        // 
        NeighborListAlphaIndice = new uint32_t[nc];
        // The end indice of each neighor list ( not included )
        NeighborListAlphaCompIndice = new uint32_t[nc];
        // The vector IDs in neighbor list of diffeent clusters
        NeighborListVec = new uint32_t[CompNumVecID];
        // The end idnice of each neighbor list
        NeighborListVecIndice = new uint32_t[nc];

        NeighborListNumTar = new uint32_t[CompNumTar];
        NeighborListNumTarIndice = new uint32_t[nc];

        NeighborListTarID = new uint32_t[CompTarID];
        NeighborListTarIDIndice = new uint32_t[nc];
        
        NeighborListInput.read((char *) NeighborListAlpha, NumAlphaList * sizeof(float));
        NeighborListInput.read((char *) NeighborListAlphaNorm, NumAlphaList * sizeof(float));
        NeighborListInput.read((char *) NeighborListAlphaIndice, nc * sizeof(uint32_t));
        NeighborListInput.read((char *) NeighborListAlphaTar, CompNumAlpha * sizeof(uint32_t));        
        NeighborListInput.read((char *) NeighborListAlphaCompIndice, nc * sizeof(uint32_t));

        NeighborListInput.read((char *) NeighborListVec, CompNumVecID * sizeof(uint32_t));
        NeighborListInput.read((char *) NeighborListVecIndice, nc * sizeof(uint32_t));

        NeighborListInput.read((char *) NeighborListNumTar, CompNumTar * sizeof(uint32_t));
        NeighborListInput.read((char *) NeighborListNumTarIndice, nc * sizeof(uint32_t));

        NeighborListInput.read((char *) NeighborListTarID, CompTarID * sizeof(uint32_t));
        NeighborListInput.read((char *) NeighborListTarIDIndice, nc * sizeof(uint32_t));


        std::cout << "The neighbor list info: NumAlphaList: " << NumAlphaList << " CompNumAlpha: " << CompNumAlpha <<
                    " NumVecID: " << NumVecID << " CompNumVecID: " << CompNumVecID << " CompNumTar: " << CompNumTar << 
                    " NumTarList: " << NumTarList << " CompTarID: " << CompTarID <<   " MaxVecListSize: " << 
                    MaxVecListSize << " MaxTarListSize: " << MaxTarListSize << " MaxAlphaListSize: " << MaxAlphaListSize << "\n";

        TRecorder.print_time_usage("Load the neighbor list index with no quantization");
    }

    // Do the query and evaluate
    std::vector<std::vector<uint32_t>> BaseCompIds(nc);
    uint32_t BaseIdCompIndice[nc];
    size_t MaxClusterSize = 0;
    if (!exists(PathBaseAssignComp)){
        std::vector<uint32_t> BaseAssignment(nb);
        std::ifstream BaseIDInput(PathBaseIDSeq, std::ios::binary);
        BaseIDInput.read((char *) BaseAssignment.data(), nb * sizeof(uint32_t));
        for (uint32_t i = 0; i < nb; i++){
            assert(BaseAssignment[i] < nc);
            BaseCompIds[BaseAssignment[i]].emplace_back(i);
        }

        for (uint32_t i = 0; i < nc; i++){
            if (BaseCompIds[i].size() > MaxClusterSize){
                MaxClusterSize = BaseCompIds[i].size();
            }
            std::vector<uint32_t> CompressedIds(BaseCompIds[i].size() + 1024);
            size_t CompressedSize = CompressedIds.size();
            codec.encodeArray(BaseCompIds[i].data(), BaseCompIds[i].size(), CompressedIds.data(), CompressedSize);
            BaseIdCompIndice[i] = i == 0 ? CompressedSize : BaseIdCompIndice[i - 1] + CompressedSize;
            memcpy(BaseCompIds[i].data(), CompressedIds.data(), CompressedSize * sizeof(uint32_t));
            BaseCompIds[i].resize(CompressedSize);
        }
        std::ofstream BaseAssignCompOutput (PathBaseAssignComp, std::ios::binary);
        BaseAssignCompOutput.write((char *) & MaxClusterSize, sizeof(size_t));
        BaseAssignCompOutput.write((char *)BaseIdCompIndice, nc * sizeof(uint32_t));
        for (uint32_t i = 0; i < nc; i++){
            BaseAssignCompOutput.write((char *) BaseCompIds[i].data(), BaseCompIds[i].size() * sizeof(uint32_t));
        }
        BaseAssignCompOutput.close();
    }
    else{
        std::ifstream BaseAssignCompInput(PathBaseAssignComp, std::ios::binary);
        BaseAssignCompInput.read((char *) & MaxClusterSize, sizeof(size_t));
        BaseAssignCompInput.read((char *) BaseIdCompIndice, nc * sizeof(uint32_t));
        for (uint32_t i = 0; i < nc; i++){
            BaseCompIds[i].resize(i == 0 ? BaseIdCompIndice[0] : BaseIdCompIndice[i] - BaseIdCompIndice[i - 1]);
            BaseAssignCompInput.read((char *) BaseCompIds[i].data(), BaseCompIds[i].size() * sizeof(uint32_t));
        }
    }

    uint32_t DeCompVecID[MaxVecListSize + 1024];
    uint32_t DeCompNumTar[MaxVecListSize + 1024];
    uint32_t DeCompTarID[MaxTarListSize + 1024];
    uint32_t DeCompAlphaTarID[MaxAlphaListSize + 1024];
    uint32_t DeCompBaseIDs[MaxClusterSize + 1024];

    bool NeighborListOnly = true;
    bool UseList = true;

    bool SearchQuant = false;
    //nq = 1;

    //std::cout << "Start Evaluate the query" << std::endl;
    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<DataType>(QueryInput, Query.data(), Dimension, nq, true, true);
    GTInput.close(); QueryInput.close();
    std::string SearchMode = "NonParallel";

    // Validation on the nieghbor list for upper bound
    bool Validation = true;
    bool QualityTest = true;
    size_t EvalK = 1;

    if(Validation){
    for (size_t ParaIdx = 0;  ParaIdx < NumPara; ParaIdx++){
        float SumVisitedGt = 0;
        float SumVisitedVec = 0;
        float SumVisitedNLNum = 0;
        float SumVisitedNLSize = 0;
        size_t NumCluster = EfSearch[ParaIdx];
        size_t MaxElements = MaxItem[ParaIdx];

        std::vector<std::pair<uint32_t, uint32_t>> QualityList(MaxElements);

        std::vector<uint32_t> QCID(nq * NumCluster);
        std::vector<float> QCDist(nq * NumCluster);
        size_t CompSize = 0;

        for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){

            std::unordered_set<uint32_t> VisitedVecSet;
            size_t VisitedVec = 0;
            size_t VisitedGt = 0;
            size_t VisitedNLSize = 0;
            size_t VisitedNLNum = 0;

            std::unordered_set<uint32_t> QueryGT;
            for (size_t j = 0; j < EvalK; j++){
                QueryGT.insert(GT[QueryIdx * ngt + j]);
            }

            auto result = Cengraph->searchBaseLayer(Query.data() + QueryIdx * Dimension, NumCluster);

            for (size_t j = 0; j < NumCluster; j++){
                QCID[QueryIdx * NumCluster + NumCluster - j - 1] = result.top().second;
                QCDist[QueryIdx * NumCluster + NumCluster - j - 1] = result.top().first;
                result.pop();
            }

            // Check the vectors in the neighbor cluster (not in the neighbor list) and the vectors in the neighbor list
            uint32_t CenClusterID = QCID[QueryIdx * NumCluster];
            //std::cout << "\n Target Cluster ID: " << TargetClusterID << "\n";
            // For the target cluster, search all vectors in the cluster

            CompSize = MaxClusterSize + 1024;
            codec.decodeArray(BaseCompIds[CenClusterID].data(), BaseCompIds[CenClusterID].size(), DeCompBaseIDs, CompSize);
            for (size_t j = 0; j < CompSize; j++){
                VisitedVec ++;

                if (QueryGT.find(DeCompBaseIDs[j]) != QueryGT.end()){
                    VisitedGt ++;
                }

                if (QualityTest){

                    QualityList[VisitedVec - 1].first ++;
                    QualityList[VisitedVec - 1].second += VisitedGt;
                }

                if (VisitedVec >= MaxElements){
                    SumVisitedVec += VisitedVec;
                    SumVisitedGt += VisitedGt;
                    break;
                }
            }

            if (VisitedVec >= MaxElements){
                continue;
            }

            //std::cout << "Compute distance between query and Neighor List\n";

            // Scan the close cluster and sort based on the distance to the query
            if (UseList){
                assert(EfNList[ParaIdx] <= NumCluster * (NumCluster - 1));
                int64_t NL2SearchID[EfNList[ParaIdx]];
                float NL2SearchDist[EfNList[ParaIdx]];
                faiss::maxheap_heapify(EfNList[ParaIdx], NL2SearchDist, NL2SearchID);

                uint32_t NumFoundNL = 0;
                size_t CompStartIndice, CompIndice, CompSize1, CompSize2, CompSize3, StartIndice, EndIndice;

                // Compute the distance between the vector and neighbor lists
                for (size_t i = 1; i < NumCluster; i++){
                    uint32_t NNClusterID = QCID[QueryIdx * NumCluster + i];

                    // Decompress the alpha target ID with a given NNCluster ID
                    CompStartIndice = NNClusterID == 0 ? 0 : NeighborListAlphaCompIndice[NNClusterID - 1];
                    CompIndice = NeighborListAlphaCompIndice[NNClusterID] - CompStartIndice;
                    if (CompIndice == 0){continue;}
                    CompSize1 = MaxAlphaListSize + 1024;

                    //memcpy(DeCompAlphaTarID, NeighborListAlphaTar + CompStartIndice, CompIndice * sizeof(uint32_t));
                    //CompSize1 = CompIndice;

                    codec.decodeArray(NeighborListAlphaTar + CompStartIndice, CompIndice, DeCompAlphaTarID, CompSize1);

                    StartIndice = NNClusterID == 0 ? 0 : NeighborListAlphaIndice[NNClusterID - 1];
                    EndIndice = NeighborListAlphaIndice[NNClusterID];

                    assert(CompSize1 == (EndIndice - StartIndice));

                    for (uint32_t j = 0; j < NumCluster; j++){
                        if (i == j) continue;
                        uint32_t TargetClusterID = QCID[QueryIdx * NumCluster + j];

                        for (size_t temp = 0; temp < CompSize1; temp++){
                            if (TargetClusterID == DeCompAlphaTarID[temp]){
                                NumFoundNL ++;
                                float Alpha = NeighborListAlpha[StartIndice + temp];
                                float AlphaNorm = NeighborListAlphaNorm[StartIndice + temp];

                                float QueryNLDist = (1 - Alpha) * (QCDist[QueryIdx * NumCluster + i] - CentroidsNorm[NNClusterID]) + Alpha * (QCDist[QueryIdx * NumCluster + j] - CentroidsNorm[TargetClusterID]) + AlphaNorm;

                                //float CentroidDist = faiss::fvec_L2sqr(Cengraph->getDataByInternalId(NNClusterID), Cengraph->getDataByInternalId(TargetClusterID), Dimension);
                                //float QueryNLDist_ = FetchQueryNLDist(QCDist[QueryIdx * NumCluster + j], QCDist[QueryIdx * NumCluster + i], CentroidDist, Alpha);

                                std::vector<float> AlphaCentroid(Dimension);
                                for (size_t k = 0; k < Dimension; k++){
                                    AlphaCentroid[k] = Cengraph->getDataByInternalId(NNClusterID)[k] * (1 - Alpha) + Cengraph->getDataByInternalId(TargetClusterID)[k] * (Alpha);
                                }
                                //float QueryNLCorrect = faiss::fvec_L2sqr(AlphaCentroid.data(), Query.data() + QueryIdx * Dimension, Dimension);

                                /*
                                std::cout << Alpha << " " << QCDist[QueryIdx * NumCluster + i] << " " << NNClusterID << " " << CentroidsNorm[NNClusterID] << " " << 
                                QCDist[QueryIdx * NumCluster + j] << " " << CentroidsNorm[TargetClusterID] << " " << AlphaNorm << "\n";
                                std::cout << QueryNLDist << " " << QueryNLDist_ << " " << QueryNLCorrect << "\n";
                                

                                std::cout << "Check 1: " << faiss::fvec_norm_L2sqr(AlphaCentroid.data(), Dimension) - (1 - Alpha) * CentroidsNorm[NNClusterID] - Alpha * CentroidsNorm[TargetClusterID];
                                std::cout << "Check 2: " << Alpha * (1 - Alpha) * CentroidDist << "\n";
                                */

                                //assert(QueryNLDist > QueryNLDist);
                                int64_t NLIndice = i * NumCluster + j;
                                if (QueryNLDist < NL2SearchDist[0]){
                                    faiss::maxheap_pop(EfNList[ParaIdx], NL2SearchDist, NL2SearchID);
                                    faiss::maxheap_push(EfNList[ParaIdx],NL2SearchDist, NL2SearchID, QueryNLDist, NLIndice);
                                }

                                break;
                            }
                        }
                    }
                    //exit(0);
                }
                /*
                    std::cout << "Form the target cluster set\n";
                    for (size_t i = 0; i < EfNList[ParaIdx]; i++){
                        std::cout << int64_t(NL2SearchID[i]) << " " << NL2SearchDist[i] << " | ";
                    }
                    std::cout << "\n";
                    std::cout << "The found number of NL: " << NumFoundNL << "\n";
                */
                int64_t ClusterIDSet[NumCluster - 1][NumCluster]; 
                uint32_t ClusterIDSetIndice[NumCluster - 1];

                for (size_t i = 0; i < NumCluster -1; i++){
                    for (size_t j = 0; j < NumCluster; j++){
                        ClusterIDSet[i][j] = -1;
                    }
                    ClusterIDSetIndice[i] = 0;
                }

                for (size_t i = 0; i < EfNList[ParaIdx]; i++){
                    if (NL2SearchID[i] < 0){continue;}
                    uint32_t NNClusterIndice = NL2SearchID[i] / NumCluster;
                    uint32_t TargetClusterIndice = NL2SearchID[i] % NumCluster;

                    assert(NNClusterIndice < NumCluster && NNClusterIndice > 0);
                    assert(TargetClusterIndice < NumCluster);
                    uint32_t TargetClusterID = QCID[QueryIdx * NumCluster + TargetClusterIndice];

                    ClusterIDSet[NNClusterIndice - 1][ClusterIDSetIndice[NNClusterIndice - 1]] = TargetClusterID;
                    ClusterIDSetIndice[NNClusterIndice - 1] ++;
                }

                for (size_t i = 0; i < NumCluster - 1; i++){
                    std::sort(ClusterIDSet[i], ClusterIDSet[i] + ClusterIDSetIndice[i]);
                }
                // Search the vectors in these NNClusters
                for (size_t i = 0; i < NumCluster - 1; i++){
                    if (VisitedVec >= MaxElements){
                        break;
                    }

                    if (ClusterIDSetIndice[i] == 0){continue;}

                    uint32_t NNClusterID = QCID[QueryIdx * NumCluster + i + 1];
                    // Decompress the vector ID and its number of targetcluster, targetcluster ID

                    CompStartIndice = NNClusterID == 0 ? 0 : NeighborListVecIndice[NNClusterID - 1];
                    CompIndice = NeighborListVecIndice[NNClusterID] - CompStartIndice;
                    CompSize1 = MaxVecListSize + 1024;

                    //memcpy(DeCompVecID, NeighborListVec + CompStartIndice, CompIndice * sizeof(uint32_t));
                    //CompSize1 = CompIndice;

                    codec.decodeArray(NeighborListVec + CompStartIndice, CompIndice, DeCompVecID, CompSize1);

                    CompStartIndice = NNClusterID == 0 ? 0 : NeighborListNumTarIndice[NNClusterID - 1];
                    CompIndice = NeighborListNumTarIndice[NNClusterID] - CompStartIndice;
                    CompSize2 = MaxVecListSize + 1024;
                    
                    //memcpy(DeCompNumTar, NeighborListNumTar + CompStartIndice, CompIndice * sizeof(uint32_t));
                    //CompSize2 = CompIndice;
                    codec.decodeArray(NeighborListNumTar + CompStartIndice, CompIndice, DeCompNumTar, CompSize2);
                    CompStartIndice = NNClusterID == 0 ? 0 : NeighborListTarIDIndice[NNClusterID - 1];
                    CompIndice = NeighborListTarIDIndice[NNClusterID] - CompStartIndice;
                    CompSize3 = MaxTarListSize + 1024;

                    //memcpy(DeCompTarID, NeighborListTarID + CompStartIndice, CompIndice * sizeof(uint32_t));
                    //CompSize3 = CompIndice;

                    codec.decodeArray(NeighborListTarID + CompStartIndice, CompIndice, DeCompTarID, CompSize3);
                    assert(CompSize1 == CompSize2);

                    uint32_t Indice = 0;
                    for (size_t temp = 0; temp < CompSize1; temp++){
                        VisitedNLNum ++;
                        bool CheckFlag = Intersect(DeCompTarID + Indice, DeCompTarID + Indice + DeCompNumTar[temp], ClusterIDSet[i], ClusterIDSet[i] + ClusterIDSetIndice[i]);

                        if (CheckFlag){
                            
                            VisitedNLSize += 1;

                            VisitedVec += 1;

                            uint32_t VectorID = DeCompVecID[temp];
                            VisitedVecSet.insert(VectorID);

                            if (QueryGT.find(VectorID) != QueryGT.end()){
                                VisitedGt ++;
                            }

                            if (QualityTest){
                                QualityList[VisitedVec - 1].first ++;
                                QualityList[VisitedVec - 1].second += VisitedGt;
                            }

                            if (VisitedVec >= MaxElements){
                                break;
                            }
                        }
                        Indice += DeCompNumTar[temp];
                    }

                    //Trecorder.recordTimeConsumption3();
                    //std::cout << "\n\n";
                }
            }
            //std::cout << "\n";

            if (!NeighborListOnly){
                for (size_t j = 1; j < NumCluster; j++){

                    if (VisitedVec >= MaxElements){break;}

                    uint32_t NNClusterID = QCID[QueryIdx * NumCluster + j];

                    CompSize = MaxClusterSize + 1024;
                    codec.decodeArray(BaseCompIds[NNClusterID].data(), BaseCompIds[NNClusterID].size(), DeCompBaseIDs, CompSize);

                    // The NNCluster is partly searched in the NL, search the remaining part
                    for (size_t k = 0; k < CompSize; k++){

                        if (VisitedVecSet.find(DeCompBaseIDs[k]) ==  VisitedVecSet.end()){
                            VisitedVecSet.insert(DeCompBaseIDs[k]);

                            VisitedVec ++;
                            if (QueryGT.find(DeCompBaseIDs[k]) != QueryGT.end()){
                                //std::cout << "Gt in Cluster: " << DeCompBaseIDs[k] << " " << j << " | ";
                                //std::cout << DeCompBaseIDs[k] << " found\n";
                                VisitedGt ++;
                            }

                            if (QualityTest){
                                QualityList[VisitedVec - 1].first ++;
                                QualityList[VisitedVec - 1].second += VisitedGt;
                            }

                            if (VisitedVec >= MaxElements){
                                break;
                            }
                        }
                    }
                }
            }

            SumVisitedNLNum += VisitedNLNum;
            SumVisitedNLSize += VisitedNLSize;
            SumVisitedGt += VisitedGt;
            SumVisitedVec += VisitedVec;       
        }

        std::cout << SumVisitedGt << "\n";
        std::cout << "\nThe average Recall of " << EvalK << " nearest neighbors: " << SumVisitedGt / (nq * EvalK) << " Number of clusters: " << NumCluster << " Num of vectors: " << SumVisitedVec / nq << " Num of NL vectors: " << SumVisitedNLSize / nq << " Total number of NL: " << SumVisitedNLNum / (nq) << " Number of NList: " << EfNList[ParaIdx] << " Number of clusters: " << NumCluster << " MaxElements: " << MaxElements << "\n";     
        //break;

        
        if (QualityTest)
        for (size_t i = 0; i < QualityList.size(); i++){
            if ((i + 1) % 1000 == 0){
                std::cout << i << " " << float(QualityList[i].second) << " " << float(QualityList[i].first) << " " << float(QualityList[i].second) / float(QualityList[i].first * EvalK) << " | ";
            }
        }
/*
        std::cout << "[";
        for (size_t i = 0; i < QualityList.size(); i++){
            std::cout << float(QualityList[i].second) / float(QualityList[i].first * EvalK) << ", ";
        }
        std::cout << "]";
*/
        exit(0);
    }
    exit(0);
    }

    //std::vector<float> PQTable(nq * PQ->ksub * PQ->M);
    recall_recorder Rrecorder = recall_recorder();
    std::string Result = "";
    RecordFile << "Search Results with UseList: " << UseList << " and UseListOnly: " << NeighborListOnly << " Search with Quantization Data: " << SearchQuant << "\n";

    std::vector<float> TimeConsumption(NumPara);
    std::vector<float> RecallResult(NumPara);
    std::vector<float> Recall1Result(NumPara);
    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInput = std::ifstream(PathBase, std::ios::binary);
    readXvec<float>(BaseInput, BaseVectors.data(), Dimension, nb, true, true);

    for (size_t i = 0; i < NumRecall; i++){

        std::vector<float> QueryDists(nq * RecallK[i]);
        std::vector<int64_t>QueryIds(nq * RecallK[i]);

        for (size_t j = 0; j < NumPara; j++){
            std::cout << "Do the retrieval for Recall@" << RecallK[i] << " with MaxItem " << MaxItem[j] << " and EfSearch " << EfSearch[j] << std::endl;
            float CenSearchTime = 0;
            float TableTime = 0;
            float VectorSearchTime = 0;
            float WholeTime = 0;
            uint32_t NumVisitedItem = 0;
           // Complete the search by stages

            std::vector<float> QCDist(nq * EfSearch[j]);
            std::vector<uint32_t> QCID(nq * EfSearch[j]);
            int64_t ClusterIDSet[(EfSearch[j] - 1) * EfSearch[j]]; 
            uint32_t ClusterIDSetIndice[EfSearch[j] - 1];

            int64_t NL2SearchID[EfNList[j]];
            float NL2SearchDist[EfNList[j]];

            TRecorder.TempDuration1 = 0; TRecorder.TempDuration2 = 0; TRecorder.TempDuration3 = 0;
            time_recorder TempRecorder = time_recorder();
            if (SearchQuant){
                //NumVisitedItem = SearchMulti(TRecorder, nq, RecallK[i], QuantBatchSize, Query.data(), QueryIds.data(), QueryDists.data(), EfSearch[j], MaxItem[j], EfNList[j], UseList, UseQuantize, NeighborListOnly, PQ, OPQ, Cengraph, CentroidsDist.data(), QCDist.data(), QCID.data(), PQTable.data(),  CNorms.data(), BaseNorms.data(), 
                                        //BaseCodes.data(), BaseIds, QuanNeighborList, NeighborListAlpha, NeighborListAlphaTar, NeighborListAlphaIndice, NeighborListVecIndice, NeighborListVecIndice);
            }
            else{
                /*
                NumVisitedItem = SearchMultiFull(TRecorder, nq, RecallK[i], Query.data(), QueryIds.data(), QueryDists.data(), 
                                                 EfSearch[j], MaxItem[j], EfNList[j], UseList, Cengraph,
                                                 QCDist.data(), QCID.data(), BaseCompIds, CentroidsNorm.data(), BaseIdCompIndice, NeighborListAlpha, NeighborListAlphaNorm, NeighborListAlphaTar,
                                                 NeighborListAlphaIndice, NeighborListAlphaCompIndice, NeighborListVec, 
                                                 NeighborListVecIndice, NeighborListNumTarIndice, NeighborListNumTar, NeighborListTarIDIndice,
                                                 NeighborListTarID, NL2SearchID, NL2SearchDist, BaseVectors.data(), DeCompAlphaTarID, 
                                                 DeCompVecID, DeCompNumTar, DeCompTarID, ClusterIDSet, ClusterIDSetIndice, codec, 
                                                 MaxAlphaListSize, MaxVecListSize, MaxTarListSize);
                */
                NumVisitedItem = SearchMultiFull(TRecorder, nq, RecallK[i], Query.data(), QueryIds.data(), QueryDists.data(),
                                                 EfSearch[j], MaxItem[j], EfNList[j], UseList, Cengraph, QCDist.data(), QCID.data(), 
                                                 BaseCompIds, CentroidsNorm.data(), NeighborListAlpha, NeighborListAlphaNorm,
                                                 NeighborListAlphaTar, NeighborListAlphaIndice, NeighborListAlphaCompIndice, NeighborListVec, 
                                                 NeighborListVecIndice, NeighborListNumTarIndice, NeighborListNumTar, NeighborListTarIDIndice, 
                                                 NeighborListTarID, NL2SearchID, NL2SearchDist, BaseVectors.data(), DeCompBaseIDs, DeCompAlphaTarID,
                                                 DeCompVecID, DeCompNumTar, DeCompTarID, ClusterIDSet, ClusterIDSetIndice, codec, MaxAlphaListSize, 
                                                 MaxVecListSize, MaxTarListSize);        
            }

            WholeTime = TempRecorder.getTimeConsumption();
            CenSearchTime = TRecorder.TempDuration1;
            TableTime = TRecorder.TempDuration2;
            VectorSearchTime = TRecorder.TempDuration3;
            Result +=  "Multi  Whole Time (ms): " + std::to_string((WholeTime) / (1000*nq)) + " CenTime:" + std::to_string(CenSearchTime / (1000 * nq)) + " TableTime: " + std::to_string(TableTime/(nq * 1000)) + " VecTime: " +  std::to_string(VectorSearchTime / (1000 * nq)) +" NumVisitedItem " + std::to_string(float(NumVisitedItem) / nq) +  " EfSearch: " + std::to_string(EfSearch[j]) + " MaxItem: " + std::to_string(MaxItem[j]) + " \n";
            size_t Correct = 0;
            size_t Correct1 = 0;
            for (size_t k = 0; k < nq; k++){
                std::unordered_set<int64_t> GtSet;
                std::unordered_set<int64_t> GtSet1;

                GtSet1.insert(GT[ngt * k]);
                for (size_t m = 0; m < RecallK[i]; m++){
                    GtSet.insert(GT[ngt * k + m]);
                    //std::cout << GT[ngt * k + m] << " ";
                }
                //std::cout << std::endl;
                assert(GtSet1.size() == 1 && GtSet.size() == RecallK[i]);

                for (size_t m = 0; m < RecallK[i]; m++){
                    //std::cout << QueryIds[k * RecallK[i] + m] <<" ";
                    if (GtSet.count(QueryIds[k * RecallK[i] + m]) != 0){
                        Correct ++;
                    }
                    if (GtSet1.count(QueryIds[k * RecallK[i] + m]) != 0){
                        Correct1 ++;
                    }
                }
                if (!(Correct <= GtSet.size() * (k + 1) && Correct1 <= GtSet1.size() * (k + 1))){
                    std::cout << "Correct: " << Correct << " Correct1: " << Correct1 << " k: " << k << "\n";
                    exit(0);
                }
                //std::cout << std::endl;
            }
            float Recall1 = float(Correct) / (nq * RecallK[i]);
            float Recall = float(Correct1) / (nq);
            TimeConsumption[j] = WholeTime;
            RecallResult[j] = Recall;
            Recall1Result[j] = Recall1;
            Result += "Recall" + std::to_string(RecallK[i]) + "@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall1) + " Recall@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall) + "\n";
            Rrecorder.print_recall_performance(nq, Recall1, RecallK[i], SearchMode, EfSearch[j], MaxItem[j]);
            //break;
        }

        Result += "\nNC = " + std::to_string(nc) + "\n";
        Result += "Time = [";
        for (size_t j = 0; j < NumPara; j++){
            Result += std::to_string(TimeConsumption[j] / (1000*nq)) + " , ";
        }
        Result += "]\nRecall1 = [";
        for (size_t j = 0; j < NumPara; j++){
            Result += std::to_string(Recall1Result[j]) + " , ";
        }
        Result += "]\nRecall = [";
        for (size_t j = 0; j < NumPara; j++){
            Result += std::to_string(RecallResult[j]) + " , ";
        }
        Result += "]\n\n";
        
        break;
    }

    std::cout << "Search Completed: " << Result << "\n";
    RecordFile << Result;


    free(NeighborListAlpha);
    free(NeighborListAlphaNorm);
    free(NeighborListAlphaTar);
    free(NeighborListAlphaIndice);
    free(NeighborListAlphaCompIndice);
    free(NeighborListVec);
    free(NeighborListVecIndice);
    free(NeighborListNumTar);
    free(NeighborListNumTarIndice);
    free(NeighborListTarID);
    free(NeighborListTarIDIndice);

    RecordFile.close();
    exit(0);
}