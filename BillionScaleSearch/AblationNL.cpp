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

    /*-----------------------------------------------------------------------------------------------------------------*/
    // Train the centroids for inverted index
    PathCentroid = PathNLFolder + "Centroids_" + std::to_string(nc) + ".fvecs";
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


    std::cout << exists(PathBaseNeighborID) << " " << exists(PathBaseNeighborDist) << " " << exists(PathBaseIDSeq) << " ";
    std::cout << PathBaseNeighborID << "\n" << PathBaseNeighborDist << "\n" << PathBaseIDSeq << "\n";
    //exit(0);

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

    /*------------------------------------*/
    bool NeighborTest = true;
    if (NeighborTest){
        std::cout << "Test the vector cost on neighboring partition\n";
        std::vector<float> Query (nq * Dimension);
        std::vector<uint32_t> GT(nq * ngt);
        std::ifstream GTInput(PathGt, std::ios::binary);
        readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
        std::ifstream QueryInput(PathQuery, std::ios::binary);
        readXvecFvec<DataType>(QueryInput, Query.data(), Dimension, nq, true, true);
        GTInput.close(); QueryInput.close();
        size_t Ef = 500;
        nq = 2000;
        std::vector<uint32_t> BaseAssignment(nb);
        std::vector<std::vector<uint32_t>> BaseIds(nc);
        std::ifstream BaseIDInput(PathBaseIDSeq, std::ios::binary);
        BaseIDInput.read((char *) BaseAssignment.data(), nb * sizeof(uint32_t));
        for (uint32_t i = 0; i < nb; i++){
            assert(BaseAssignment[i] < nc);
            BaseIds[BaseAssignment[i]].emplace_back(i);
        }

        std::cout << "Start the evaluation\n";
        TRecorder.reset();

        for (size_t KInEval = 1; KInEval <= 10; KInEval++){
        
        std::vector<std::pair<uint32_t, uint32_t>> VectorCost(nq); 


//#pragma omp parallel for
        for (size_t i = 0; i < nq; i++){
            std::cout << "Processing " << i << " / " << nq << "\n";
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
            }

            VectorCost[i].first = BaseIds[QCID[0]].size();
            for (size_t j =0; j < Ef; j++){
                uint32_t ClusterID = QCID[j];

                VisitedVec += BaseIds[ClusterID].size();
                for (size_t k = 0; k < BaseIds[ClusterID].size(); j++){
                    if (QueryGT.find(BaseIds[ClusterID][k]) != QueryGT.end()){
                        VisitedGt ++;
                    }
                }
                assert(VisitedGt <= KInEval);
                if (VisitedGt == KInEval){
                    break;
                }
            }
            VectorCost[i].second = VisitedVec;
        }

        float Ratio = 0;
        for (size_t i = 0; i < nq; i++){
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
    ComputeNN(Graph_num_batch, SearchK, Datasetsize, TRecorder, MRecorder, NNDatasetName, PathNNDataset, PathDatasetNN, PathSubGraphFolder);

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

                                float CentroidDist = faiss::fvec_L2sqr(Cengraph->getDataByInternalId(NNClusterID), Cengraph->getDataByInternalId(TargetClusterID), Dimension);
                                float QueryNLDist_ = FetchQueryNLDist(QCDist[QueryIdx * NumCluster + j], QCDist[QueryIdx * NumCluster + i], CentroidDist, Alpha);

                                std::vector<float> AlphaCentroid(Dimension);
                                for (size_t k = 0; k < Dimension; k++){
                                    AlphaCentroid[k] = Cengraph->getDataByInternalId(NNClusterID)[k] * (1 - Alpha) + Cengraph->getDataByInternalId(TargetClusterID)[k] * (Alpha);
                                }
                                float QueryNLCorrect = faiss::fvec_L2sqr(AlphaCentroid.data(), Query.data() + QueryIdx * Dimension, Dimension);

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
    std::vector<DataType> BaseVectors(nb * Dimension);
    std::ifstream BaseInput = std::ifstream(PathBase, std::ios::binary);
    readXvec<DataType>(BaseInput, BaseVectors.data(), Dimension, nb, true, true);

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
                                                 BaseCompIds, CentroidsNorm.data(), BaseIdCompIndice, NeighborListAlpha, NeighborListAlphaNorm,
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