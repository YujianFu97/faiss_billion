#include "BIndex.h"

void ComupteResidual(size_t Dimension, float * Subtrahend, float * Minuend, float * Residual){
    faiss::fvec_madd(Dimension, Subtrahend, -1.0, Minuend, Residual);
    return;
}

void ReverseResidual(size_t Dimension, float * Addend1, float * Addend2, float * Result){
    faiss::fvec_madd(Dimension, Addend1, 1.0, Addend2, Result);
    return;
}

// Generate a list of size n with numbers 0 - n-1 in random sort
void rand_perm (uint32_t *perm, size_t n, int64_t seed)
{
    for (size_t i = 0; i < n; i++) perm[i] = i;
    faiss::RandomGenerator rng (seed);
    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.mt() % (n - i);
        std::swap(perm[i], perm[i2]);
    }
}

template<typename T>
void SampleSet(float * ResultData, std::string PathLearn, size_t LoadSize, size_t TotalSize, size_t Dimension){
    

    if (LoadSize == TotalSize){
        std::ifstream TrainInput(PathLearn, std::ios::binary);
        readXvecFvec<T>(TrainInput, ResultData, Dimension, LoadSize, true, true);
    }
    else{
        long RandomSeed = 1234;
        std::vector<uint32_t> RandomId(TotalSize); faiss::RandomGenerator RNG(RandomSeed);

        rand_perm(RandomId.data(), TotalSize, RandomSeed+1);
#pragma omp parallel for
        for (size_t i = 0; i < LoadSize; i++){
            std::ifstream TrainInput(PathLearn, std::ios::binary);
            std::vector<T> OriginData(Dimension);
            TrainInput.seekg(RandomId[i] * (Dimension * sizeof(T) + sizeof(uint32_t)) + sizeof(uint32_t), std::ios::beg);
            TrainInput.read((char *) OriginData.data(), Dimension * sizeof(T));
            for (size_t j = 0; j < Dimension; j++){
                ResultData[i * Dimension + j] = 1.0 * OriginData[j];
            }
            TrainInput.close();
        }
        std::cout << "Check the sampled vectors: \n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << ResultData[i] << " ";
        }
        std::cout << "\n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << ResultData[(LoadSize-1) * Dimension + i] << " ";
        }
        std::cout << "\n";
    }
    return;
}

BIndex::BIndex(const size_t Dimension, const size_t nb, const size_t nc, const size_t nt, const bool Saving, const bool Recording,
    const bool Retrain, const bool UseOPQ, const size_t M_PQ, const size_t CodeBits): Dimension(Dimension), nb(nb), nc(nc), nt(nt), Saving(Saving), Recording(Recording),
    Retrain(Retrain), UseOPQ(UseOPQ), M_PQ(M_PQ), CodeBits(CodeBits){
        Mrecorder = memory_recorder();
        Trecorder = time_recorder();
    }

float BIndex::TestDistBound(size_t K, size_t ngt, size_t nq, std::string PathQuery, std::string PathGt, std::string PathBase){
    std::vector<float> Query (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<DataType>(QueryInput, Query.data(), Dimension, nq, true, true);
    GTInput.close(); QueryInput.close();
    std::ifstream BaseInput(PathBase, std::ios::binary);

    float DistBound;
    //for (K = 1; K <= 50; K++){
    DistBound = 0;
    std::vector<float> GtVector(Dimension);
    std::vector<DataType> GtVectorOrigin(Dimension);

    for (size_t i = 0; i < nq; i++){
        uint32_t BaseId = GT[i * ngt + K - 1];
        BaseInput.seekg(BaseId * (Dimension * sizeof(DataType) + sizeof(uint32_t)) + sizeof(uint32_t), std::ios::beg);
        BaseInput.read((char *) GtVectorOrigin.data(), Dimension * sizeof(DataType));
        for (size_t j = 0; j < Dimension; j++){GtVector[j] = GtVectorOrigin[j];}
        DistBound += faiss::fvec_L2sqr(Query.data() + i * Dimension, GtVector.data(), Dimension);
    }

    return DistBound / (nq);
}


// Require the trainset, store the centroids on disk if saving
float BIndex::TrainCentroids(size_t CenTrainSize, std::string PathLearn, std::string PathCentroid, std::string PathCentroidNorm, bool Optimize, bool UseGraph, size_t Nlevel, size_t OptSize, float Lambda){
    // Check if the centroid exists
    Trecorder.reset();
    if (!Retrain && exists(PathCentroid)){
        std::cout << "Reading the Centroids Norms from file " << PathCentroid << " on disk " << std::endl;
        std::ifstream CentroidNormInput(PathCentroidNorm, std::ios::binary);
        CNorms.resize(nc);
        CentroidNormInput.read((char * )CNorms.data(), nc * sizeof(float));
        Trecorder.print_time_usage("Read Centroids Norms into main memory");
        return 0;
    }

    std::vector<float> TrainSet(Dimension * CenTrainSize);

    SampleSet<DataType>(TrainSet.data(), PathLearn, CenTrainSize, nt, Dimension);
    Trecorder.print_time_usage("Sampled subset for training");

    std::vector<float> Centroids(nc * Dimension);
    float Err;
    std::vector<float> TrainDist(CenTrainSize);
    std::vector<uint32_t> TrainLabel(CenTrainSize);
    Err = hierarkmeans(TrainSet.data(), Dimension, CenTrainSize, nc, Centroids.data(), Nlevel, Optimize, UseGraph, OptSize, Lambda, 30, true, TrainLabel.data(), TrainDist.data());

    CNorms.resize(nc);
    faiss::fvec_norms_L2sqr(CNorms.data(), Centroids.data(), Dimension, nc);

    if (Saving){
        std::cout << "Writing the Centroids and Norms to file " << PathCentroid << " on disk" << std::endl;
        std::ofstream CentroidOutput(PathCentroid, std::ios::binary);
        std::ofstream CentroidNormOutput(PathCentroidNorm, std::ios::binary);
        uint32_t Dim = Dimension;
        for (size_t i = 0; i < nc; i++){
            CentroidOutput.write((char *) & Dim, sizeof(uint32_t));
            CentroidOutput.write((char *) (Centroids.data() + i * Dimension), Dimension * sizeof(float));
        }
        CentroidOutput.close();
        CentroidNormOutput.write((char * ) CNorms.data(), nc * sizeof(float));
    }
    Trecorder.print_time_usage("Centroid Training Completed");
    return Err;
}

// Input parameters: (1) scalar value (2) data vector (3) path (4) point, data structure
uint32_t BIndex::LearnCentroidsINI(
    size_t CenTrainSize, size_t nq, bool Optimize, size_t MinNC, size_t MaxNC, float TargetRecall, float MaxCandidateSize, size_t RecallK, size_t MaxFailureTimes, float CheckProp, size_t LowerBound,
    size_t NCBatch, size_t PQ_M, size_t nbits, size_t PQ_TrainSize, size_t GraphM, size_t GraphEf, size_t NLevel, size_t OptSize, float Lambda, size_t ngt,
    std::string Path_folder, std::string Path_GT, std::string Path_base, std::string Path_query, std::string Path_learn,
    std::ofstream &RecordFile
){
    Trecorder.reset();
    std::string Path_folder_recall = Path_folder + std::to_string(MaxNC) + "_" + std::to_string(NCBatch)  + "_" + std::to_string(MaxFailureTimes) + "_" + std::to_string(M_PQ) + "_Recall_" + std::to_string(size_t(TargetRecall* 100 / RecallK)) + "_" + std::to_string(RecallK) + "/";
    PrepareFolder(Path_folder_recall.c_str());
    std::string Path_opt_file = Path_folder_recall + "OptRecord.txt";
    std::ifstream OptInput(Path_opt_file, std::ios::app);
    std::string s;
    bool IndexPreTrained = false;
    std::vector<std::string> OptString;
    while(getline(OptInput, s)){
        if (s == "Training process completed with failure times: " + std::to_string(MaxFailureTimes)){
            IndexPreTrained = true;
        }
        else{
            IndexPreTrained = false;
            OptString = StringSplit(s, " ");
        }
    }
    OptInput.close();

    size_t OptNC = 0;
    nc = MinNC;
    std::vector<uint32_t> Base_ID_seq(nb);
    std::string Path_centroid = Path_folder + "Centroids_" + std::to_string(nc) + ".fvecs";
    uint32_t Assignment_num_batch = 20; assert(nb % Assignment_num_batch == 0);
    uint32_t Assignment_batch_size = nb / Assignment_num_batch;

    if(!IndexPreTrained){
    // Parameters requried

    std::ofstream OptOutput; OptOutput.open(Path_opt_file, std::ios::app);

    uint32_t Assignment_indice = 0;
    uint32_t Assignment_num = 20;

    // 1. Initialize the centroids

    std::cout << "Initializing the cluster centroids\n";
    std::vector<float>  Centroids(MaxNC * Dimension);

    if (!Retrain && exists(Path_centroid)){
        std::ifstream IniCentroidInput(Path_centroid, std::ios::binary);
        readXvec<float>(IniCentroidInput, Centroids.data(), Dimension, nc, true, true);
        std::cout << "Load ini pretrained centroids from " << Path_centroid << "\n";
    }
    else{
        std::vector<float> TrainSet(Dimension * CenTrainSize);
        std::cout << "Load the train vectors from: " << Path_learn << std::endl;
        SampleSet<DataType>(TrainSet.data(), Path_learn, CenTrainSize, nt, Dimension);
        Trecorder.print_record_time_usage(RecordFile, "Sampled subset for training");

        hierarkmeans(TrainSet.data(), Dimension, CenTrainSize, nc, Centroids.data(), NLevel, Optimize, true, OptSize, Lambda, 30);
        std::ofstream IniCentroidOutput(Path_centroid, std::ios::binary);
        for (size_t i = 0; i < nc; i++){
            IniCentroidOutput.write((char *) & Dimension, sizeof(uint32_t));
            IniCentroidOutput.write((char *) (Centroids.data() + i * Dimension), Dimension * sizeof(float));
        }
        IniCentroidOutput.close();
    }
    Trecorder.print_record_time_usage(RecordFile, "Prepared the initialization centroids");

    // 2. Construct the graph for base vector assignment
    std::string Path_Graph_info = Path_folder + "Graph_" + std::to_string(nc) + "_" + std::to_string(GraphM) + "_" + std::to_string(GraphEf) + ".info"; 
    std::string Path_Graph_Edge = Path_folder + "Graph_" +std::to_string(nc) + "_" + std::to_string(GraphM) + "_" + std::to_string(GraphEf) + ".edge"; 

    hnswlib::HierarchicalNSW * HNSWGraph;
    if (!Retrain && exists(Path_Graph_info) && exists(Path_Graph_Edge)){
        HNSWGraph = new hnswlib::HierarchicalNSW(Path_Graph_info, Path_centroid, Path_Graph_Edge);
    }
    else{
        HNSWGraph = new hnswlib::HierarchicalNSW(Dimension, nc, GraphM, 2 * GraphM, GraphEf);
        for (size_t i = 0; i < nc; i++){HNSWGraph->addPoint(Centroids.data() + i * Dimension);}
        RecordFile << "Graph data: efConstruction " << HNSWGraph->efConstruction_ << " efSearch: " << HNSWGraph->efSearch << " M: " << HNSWGraph->maxM_ << " Num of nodes: " << HNSWGraph->maxelements_ << "\n";
        HNSWGraph->SaveEdges(Path_Graph_Edge); HNSWGraph->SaveInfo(Path_Graph_info);
    }
    Trecorder.print_record_time_usage(RecordFile, "Construct the graph and save graph index");

    // 3.1. Assign the billion-scale dataset in batch and in parallel, we can run this assignment process on multiple machines
    std::ifstream BaseInput(Path_base, std::ios::binary);
    std::string Path_ID_seq  = Path_folder + "BaseID_" + std::to_string(nc) + ".seq";
    for (uint32_t batch_idx = 0; batch_idx < Assignment_num; batch_idx++){
        std::string Batch_ID_path = Path_folder + "BaseID_" +  std::to_string(batch_idx + 1) + "_" + std::to_string(Assignment_num_batch) + ".ivecs";
        if (exists(Path_ID_seq) || exists(Batch_ID_path)){continue;}

        std::cout << "Assigning the " << Assignment_indice + batch_idx + 1 << " / " << Assignment_num_batch << " batch of the dataset and save ID to" << Batch_ID_path << " Time consumption: " << Trecorder.get_time_usage() << " s\n";
        BaseInput.seekg((Assignment_indice + batch_idx) * Assignment_batch_size * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
        std::vector<float> Base_batch(Assignment_batch_size * Dimension);
        readXvecFvec<DataType>(BaseInput, Base_batch.data(), Dimension, Assignment_batch_size, true, true);
        std::vector<uint32_t> Base_batch_id(Assignment_batch_size, 0);
#pragma omp parallel for
        for (uint32_t data_idx = 0; data_idx < Assignment_batch_size; data_idx++){
            auto Result = HNSWGraph->searchKnn(Base_batch.data() + data_idx * Dimension, 1);
            Base_batch_id[data_idx] = Result.top().second;
        }
        std::ofstream Batch_ID_output(Batch_ID_path, std::ios::binary);
        Batch_ID_output.write((char *) & Assignment_batch_size, sizeof(uint32_t));
        Batch_ID_output.write((char *) Base_batch_id.data(), Assignment_batch_size * sizeof(uint32_t));
        Batch_ID_output.close();
    }
    BaseInput.close();
    Trecorder.print_record_time_usage(RecordFile, "Assign the base vectors in batches");

    // 3.2. Load the base vector ID to main memory, ensure the ID_batch_file exists
    for (uint32_t i = 0; i < Assignment_num_batch; i++){
        std::string Batch_ID_path = Path_folder + "BaseID_" +  std::to_string(i + 1) + "_" + std::to_string(Assignment_num) + ".ivecs";
        if(!exists(Batch_ID_path) && !exists(Path_ID_seq) ){std::cout << "Base vector ID file " << Batch_ID_path << " not exists, cannot continue the training process\n"; exit(0);}
    }

    if (exists(Path_ID_seq)){
        std::ifstream BaseIDSeqINput(Path_ID_seq, std::ios::binary);
        BaseIDSeqINput.read((char *) Base_ID_seq.data(), nb * sizeof(uint32_t));
        BaseIDSeqINput.close();
        std::cout << "The 10 base IDs in the top and the end:\n";
        for (size_t i = 0; i < 10; i++){
            std::cout << Base_ID_seq[i] << " ";
        }
        for (size_t i = 0; i < 10; i++){
            std::cout << Base_ID_seq[nb - 10 + i] << " ";
        }
        std::cout << "\n";
    }
    else{
        for (uint32_t i = 0; i < Assignment_num_batch; i++){
            std::string Batch_ID_path = Path_folder + "BaseID_" +  std::to_string(i + 1) + "_" + std::to_string(Assignment_num) + ".ivecs";
            std::ifstream Batch_ID_input(Batch_ID_path, std::ios::binary);
            readXvec<uint32_t>(Batch_ID_input, Base_ID_seq.data() + i * Assignment_batch_size, Assignment_batch_size, 1);
            Batch_ID_input.close();
        }
        std::ofstream BaseIDSeqOutput(Path_ID_seq, std::ios::binary);
        BaseIDSeqOutput.write((char * ) Base_ID_seq.data(), nb * sizeof(uint32_t));
        BaseIDSeqOutput.close();
        std::cout << "Save vector ID in seq index to " << Path_ID_seq << "\n";
    }

    // 3.3 Save the ID to inverted format
    BaseIds.resize(nc); 
    for (uint32_t i = 0; i < nb; i++){
        assert(Base_ID_seq[i] < nc);
        BaseIds[Base_ID_seq[i]].emplace_back(i);
    }
    std::cout << "The top 10 cluster size: \n";
    for (size_t i = 0; i < 10; i++){
        std::cout << BaseIds[i].size() << " ";
    }
    std::cout << "\n";
    Trecorder.print_record_time_usage(RecordFile, "Load base ID to index and merge the ID");

    // 4. Load the query and groundtruth of the queries on baseset for recall check
    std::ifstream GTInput(Path_GT, std::ios::binary);
    std::ifstream QueryInput(Path_query, std::ios::binary);
    std::vector<uint32_t> QueryGT(nq * ngt);
    readXvec<uint32_t> (GTInput, QueryGT.data(), ngt, nq, true, true);
    std::vector<float>  QuerySet(nq * Dimension);
    readXvecFvec<DataType>(QueryInput, QuerySet.data(), Dimension, nq, true, true);
    QueryInput.close(); GTInput.close();
    Trecorder.print_record_time_usage(RecordFile, "Load the queries and groundtruth");

    // 5. Prepare and load the trainset for PQ and train the quantizer, reuse the trainset for centroids
    std::vector<int> RandomId(nb);
    faiss::rand_perm(RandomId.data(), nb, 1234+1);
    std::vector<float> SubResidual(PQ_TrainSize * Dimension, 0);
    std::vector<float> TrainSet(PQ_TrainSize * Dimension);


    BaseInput.open(Path_base, std::ios::binary);
    for (size_t i =0; i < PQ_TrainSize; i++){
        BaseInput.seekg(RandomId[i] * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
        readXvecFvec<DataType>(BaseInput, TrainSet.data() + i * Dimension, Dimension, 1);
    }
    BaseInput.close();

    Trecorder.print_record_time_usage(RecordFile, "Load the subtrainset for PQ training");

    // 6. Build the inverted index for base vectors
    bool ContinueSplit = true;
    std::vector<size_t> NCRecord; std::vector<float> CenTimeRecord; std::vector<float> VecTimeRecord; std::vector<float> NumClusterRecord; std::vector<float> CansizeRecord;
    std::vector<float> ClusterVectorCost;

    auto comp = [](std::pair<float, uint32_t> Element1, std::pair<float, uint32_t> Element2){return Element1.first < Element2.first;};
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(comp)> ClusterCostQueue(comp);

    float OptPerTime = std::numeric_limits<float>::max();
    size_t FailureTimes = 0;
    
    // The loop for changing the M
    while(ContinueSplit){
        std::cout << "\nRun cluster split with batch: " << NCBatch << " and nc: " << nc << "\n";
        // 1.1. Compute the centroid norm
        CNorms.resize(nc);

        std::string Path_cen_norm = Path_folder_recall + "Centroids_" + std::to_string(nc)+".norm";

///*
        if (!Retrain && exists(Path_cen_norm)){
            std::ifstream CenNormInput(Path_cen_norm, std::ios::binary);
            size_t nc_r;
            CenNormInput.read((char *) & nc_r, sizeof(size_t));
            assert(nc_r == nc);
            CenNormInput.read((char *) CNorms.data(), nc * sizeof(float));
            CenNormInput.close();
        }
        else{
//*/
            std::ofstream CenNormOutput(Path_cen_norm, std::ios::binary);
            faiss::fvec_norms_L2sqr(CNorms.data(), Centroids.data(), Dimension, nc);
            CenNormOutput.write((char * ) & nc, sizeof(size_t));
            CenNormOutput.write((char * ) CNorms.data(), nc * sizeof(float));
            CenNormOutput.close();
       }

        // 1.2. Build PQ quantizer with the constructed centroids and the a subset of the baseset, Save the PQ quantizer to disk for further evaluation
        std::string Path_PQ = Path_folder_recall + "PQ_" + std::to_string(nc)+"_"+std::to_string(PQ_M) + "_" + std::to_string(nbits)+".pq";
///*
        if (!Retrain && exists(Path_PQ)){
            PQ = faiss::read_ProductQuantizer(Path_PQ.c_str());
        }
        else{
//*/
            for (size_t i = 0; i < PQ_TrainSize; i++){
                uint32_t ClusterID = Base_ID_seq[RandomId[i]];
                faiss::fvec_madd(Dimension, TrainSet.data() + i * Dimension, -1.0,  HNSWGraph->getDataByInternalId(ClusterID), SubResidual.data() + i * Dimension);
            }
            PQ = new faiss::ProductQuantizer(Dimension, PQ_M, nbits); PQ->verbose = true;
            PQ->train(PQ_TrainSize, SubResidual.data());
            faiss::write_ProductQuantizer(PQ, Path_PQ.c_str());

/*
            std::vector<uint8_t> SubResidualCode(PQ_TrainSize * PQ->code_size);
            PQ->compute_codes(SubResidual.data(), SubResidualCode.data(), PQ_TrainSize);
            std::vector<float> RecoverResidual(PQ_TrainSize * Dimension);
            PQ->decode(SubResidualCode.data(), RecoverResidual.data(), PQ_TrainSize);

            for (size_t i = 0; i < PQ_TrainSize; i++){
                std::cout << faiss::fvec_norm_L2sqr(SubResidual.data() + i * Dimension, Dimension) << " " << faiss::fvec_norm_L2sqr(RecoverResidual.data() + i * Dimension, Dimension) << " " << faiss::fvec_L2sqr(SubResidual.data() + i * Dimension, RecoverResidual.data() + i * Dimension, Dimension) << " | "; 
            }
            std::cout << "\n\n\n";
*/
        }

        Trecorder.print_record_time_usage(RecordFile, "Train the PQ quantizer");

        // 2. Update the search performance




    time_recorder TRecorder = time_recorder();
    // Accumulate the vector quantization in the clusters to be visited by the queries
    std::cout << "Check the recall performance on different num clusters to be visited\n";
    std::vector<bool> QuantizeLabel(nc, false);
    std::vector<std::vector<uint8_t>> BaseCodeSubset(nc);
    std::vector<std::vector<float>> BaseRecoverNormSubset(nc);
    std::cout << "-2\n";

    std::vector<int64_t> ResultID(RecallK * nq, 0);
    std::vector<float> ResultDist(RecallK * nq, 0);

    std::vector<std::unordered_set<uint32_t>> GtSets(nq);
    for (size_t i = 0; i < nq; i++){
        for (size_t j = 0; j < RecallK; j++){
            GtSets[i].insert(QueryGT[ngt * i + j]);
        }
    }
    std::cout << "-1\n";

    bool ValidResult = false;
    float MinimumCoef = 0.95;
    size_t MaxRepeatTimes = 3;
    
    while(!ValidResult){
        size_t ClusterNum = size_t(MaxCandidateSize / (2 * (nb / nc)));
        size_t ClusterBatch = std::ceil(float(ClusterNum) / 10);
        std::vector<float> ClusterNumList;
        std::vector<float> CanLengthList;
        std::vector<float> CenSearchTime;
        std::vector<float> VecSearchTime;



        bool AchieveTargetRecall = true;
        bool UpdateClusterNum = true;
        bool IncreaseClusterNum = false;
        bool DecreaseClusterNum = false;
        float ResultIndice = -1;
        float VisitedVec = 0;
        size_t PreviousClusterNum = ClusterNum;
        float PreviousRecordTime1 = 0;
        float PreviousRecordTime3 = 0;
        size_t RepeatTimes = 0;
        int nt = omp_get_max_threads();

        // Change different ClusterNum
        
        while (UpdateClusterNum)
        {
            
            // Record the time of graph search on centroids
            std::cout << "0: \n";
            std::vector<float> QueryDist(nq * ClusterNum);
            std::vector<uint32_t> QueryLabel(nq * ClusterNum);
            TRecorder.reset();
            
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                auto result = HNSWGraph->searchBaseLayer(QuerySet.data() + QueryIdx * Dimension, ClusterNum);
                for (size_t i = 0; i < ClusterNum; i++){
                    QueryLabel[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().second;
                    QueryDist[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().first;
                    result.pop();
                }
            }

            TRecorder.recordTimeConsumption1();
            std::cout << "1: \n";


            size_t NumLoadCluster = 0;
            TRecorder.reset();
            // Load and quantize the vectors to be visited
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                // Do parallel for the search result of one query, as there will be no repeat

                for (size_t i = 0; i < ClusterNum; i++){
                    if (!QuantizeLabel[QueryLabel[QueryIdx * ClusterNum + i]]){
                        NumLoadCluster ++;
                        uint32_t ClusterLabel = QueryLabel[QueryIdx * ClusterNum + i];
                        // The vectors are not quantized, read and quantize the base vectors
                        QuantizeLabel[ClusterLabel] = true;
                        BaseCodeSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size() * PQ->code_size);
                        BaseRecoverNormSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size());

                        
                        size_t ClusterSize = BaseIds[ClusterLabel].size();

                        size_t StartIndice = 0;
                        size_t EndIndice = StartIndice + (nt < ClusterSize ? nt : ClusterSize);
                        bool FlagContinue = true;

                        while(FlagContinue){
#pragma omp parallel for
                            for (size_t j = StartIndice; j < EndIndice; j++){
                                std::ifstream BaseInput(Path_base, std::ios::binary);
                                std::vector<float> BaseVector(Dimension);
                                BaseInput.seekg(BaseIds[ClusterLabel][j] * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);

                                readXvecFvec<DataType>(BaseInput, BaseVector.data(), Dimension, 1);
                                std::vector<float> BaseResidual(Dimension);
                                std::vector<float> RecoverResidual(Dimension);
                                faiss::fvec_madd(Dimension, BaseVector.data(), -1.0,  HNSWGraph->getDataByInternalId(ClusterLabel), BaseResidual.data());
                                PQ->compute_code(BaseResidual.data(), BaseCodeSubset[ClusterLabel].data() + j * PQ->code_size);
                                PQ->decode(BaseCodeSubset[ClusterLabel].data() + j * PQ->code_size, RecoverResidual.data());
                                std::vector<float> RecoverVector(Dimension);
                                faiss::fvec_madd(Dimension, RecoverResidual.data(), 1.0, HNSWGraph->getDataByInternalId(ClusterLabel), RecoverVector.data());
                                BaseRecoverNormSubset[ClusterLabel][j] = faiss::fvec_norm_L2sqr(RecoverVector.data(), Dimension);

                                //std::cout << faiss::fvec_norm_L2sqr(BaseResidual.data(), Dimension) << " " << faiss::fvec_norm_L2sqr(RecoverResidual.data(), Dimension) << " " << faiss::fvec_L2sqr(BaseResidual.data(), RecoverResidual.data(), Dimension) << " | "; 
                                BaseInput.close();
                            }
                            std::cout << nq << " " << QueryIdx << " " << i << " " << ClusterSize << "\r";
                            if (EndIndice == ClusterSize){FlagContinue = false;}
                            StartIndice = EndIndice;
                            EndIndice = EndIndice + nt <= ClusterSize ? EndIndice + nt : ClusterSize;
                        }
                    }
                }
            }
            TRecorder.print_time_usage("Load and quantize the base vectors in |" + std::to_string(NumLoadCluster) + "| clusters ");
        }
        exit(0);
    }





        std::cout << "Get into the recall performance estimation process\n";









        auto RecallResult = BillionUpdateRecall(nb, nq, Dimension, nc, RecallK, TargetRecall, MaxCandidateSize, ngt, QuerySet.data(), QueryGT.data(), CNorms.data(), Path_base, RecordFile, HNSWGraph, PQ, BaseIds);

        delete PQ;
        Trecorder.print_record_time_usage(RecordFile, "Update the search recall performance");

        // 3.1. Check the search performance and determine whether the centroids should be splitted and report the search performance
        float PerTime = std::get<3>(RecallResult) + std::get<4>(RecallResult);
        NCRecord.emplace_back(nc); CenTimeRecord.emplace_back(std::get<3>(RecallResult)); VecTimeRecord.emplace_back(std::get<4>(RecallResult)); CansizeRecord.emplace_back(std::get<2>(RecallResult));
        RecordFile << "\nThis is index with |" << nc << "| centroids with search time: |" << PerTime << "| PerCentroidTime: |" << std::get<3>(RecallResult) << "| Per Vecor Time: |" << std::get<4>(RecallResult) <<  "| ClusterNum: " << std::get<1>(RecallResult) << " Candidate List Size: " << std::get<2>(RecallResult) << std::endl;
        std::cout << "\nThis is index with |" << nc << "| centroids with search time: |" << PerTime << "| PerCentroidTime: |" << std::get<3>(RecallResult) << "| Per Vecor Time: |" << std::get<4>(RecallResult) << "| ClusterNum: " << std::get<1>(RecallResult) << " Candidate List Size: " << std::get<2>(RecallResult) << std::endl;

        RecordFile << "\nNC = [";  for (size_t i = 0; i < NCRecord.size(); i++){RecordFile << NCRecord[i] << ", ";}
        RecordFile << "]\nCanlist = [";for (size_t i = 0; i < CansizeRecord.size(); i++){RecordFile << CansizeRecord[i] << ", ";}
        RecordFile << "]\nCenTime = [";for (size_t i = 0; i < CenTimeRecord.size(); i++){RecordFile << CenTimeRecord[i] << ", ";}
        RecordFile << "]\nVecTime = [";for (size_t i = 0; i < VecTimeRecord.size(); i++){RecordFile << VecTimeRecord[i] << ", ";}
        RecordFile << "]\nSumTime = [";for (size_t i = 0; i < VecTimeRecord.size(); i++){RecordFile << CenTimeRecord[i] + VecTimeRecord[i] << ", ";}
        RecordFile << "]\n";

        // 3.2. Check the quality of centroids
        // This iteration offers new optimal centroids
        if (std::get<0>(RecallResult) && PerTime < OptPerTime){
            OptOutput << "New opt NC: " << nc << " Centroid Path: " << Path_centroid << " PQ Path: " << Path_PQ << " Centroid Norm Path: " << Path_cen_norm << 
            " Graph edge path: " << Path_Graph_Edge << " Graph info path: " << Path_Graph_info << " Path ID_seq: " << Path_ID_seq <<
             " Search Time: " << PerTime << " Target Recall: " << TargetRecall << "\n";
            OptOutput.close();
            OptOutput.open(Path_opt_file, std::ios::app);

            OptPerTime = PerTime;
            OptNC = nc;
            FailureTimes = 0;
            RecordFile << "This is new optimal index \n\n";
            std::cout << "This is new optimal index \n\n";
        }
        //  This iteration cannot achieve the target recall
        else if(! std::get<0>(RecallResult)){
            OptOutput << "New opt NC: " << nc << " Centroid Path: " << Path_centroid << " PQ Path: " << Path_PQ << " Centroid Norm Path: " << Path_cen_norm << 
            " Graph edge path: " << Path_Graph_Edge << " Graph info path: " << Path_Graph_info << " Path ID_seq: " << Path_ID_seq <<
             " Search Time: " << PerTime << " Target Recall: " << TargetRecall << "\n";
            OptOutput.close();
            OptOutput.open(Path_opt_file, std::ios::app);

            OptPerTime = PerTime;
            OptNC = nc;
            RecordFile << "This is *below recall target* index\n\n";
            std::cout << "This is *below recall target* index\n\n";
        }
        // This iteration offer non-optimal centroids
        else{
            FailureTimes ++;
            RecordFile << "This is *not* optimal index, the optimal NC is: |" << OptNC << "|, Optimal search time: |" << OptPerTime << "| Failure Time: " << FailureTimes << " / " << MaxFailureTimes << "\n\n"; 
            std::cout << "This is *not* optimal index, the optimal NC is: |" << OptNC << "|, Optimal search time: |" << OptPerTime << "| Failure Time: " << FailureTimes << " / " << MaxFailureTimes <<"\n\n"; 
        }

        // 3.3. Determine whether the index should be further splitted
        // Stop increasing NC
        if (nc + NCBatch > MaxNC || FailureTimes == MaxFailureTimes){
            ContinueSplit = false;
        }

        // If the centroids should be further splitted, update the cluster search cost
        float SumClusterCost = 0;
        if (ContinueSplit){
            // Update the candidate list size
            ClusterVectorCost.resize(nc, 0);
            BillionUpdateCost(std::get<1>(RecallResult), nc, CheckProp, LowerBound, Dimension, ClusterVectorCost.data(), Path_base, BaseIds, HNSWGraph);
            for (uint32_t i = 0; i < nc; i++){
                ClusterCostQueue.emplace(std::make_pair(ClusterVectorCost[i], i));
                SumClusterCost += ClusterVectorCost[i];
            }
            delete HNSWGraph;
            HNSWGraph = nullptr;
            Trecorder.print_record_time_usage(RecordFile, "Update the cluster search cost for further split");

            // Split the clusters to multiple small clusters and update the index information
            std::vector<uint32_t> ClusterIDBatch(NCBatch, 0);
            std::vector<float> ClusterCostBatch(NCBatch, 0);
            for (size_t i = 0; i < NCBatch; i++){
                ClusterIDBatch[i] = ClusterCostQueue.top().second;
                ClusterCostBatch[i] = ClusterCostQueue.top().first;
                ClusterCostQueue.pop();
            }
            while(!ClusterCostQueue.empty()){
                ClusterCostQueue.pop();
            }

            std::cout << "The top 50 clusters to be further split: \n";
            for (size_t i = 0; i < 50; i++){
                std::cout << ClusterIDBatch[i] << " " << ClusterCostBatch[i] << " " << BaseIds[ClusterIDBatch[i]].size() << " | ";
            }
            std::cout << "\n";

            BillionUpdateCentroids(Dimension, NCBatch, SumClusterCost / nc, Optimize, Lambda, OptSize, nc, ClusterCostBatch.data(), ClusterIDBatch.data(), Centroids.data(), Base_ID_seq.data(), Path_base, BaseIds);
            Trecorder.print_record_time_usage(RecordFile, "Split the clusters and update the vector IDs");

            // Save the base set ID
            Path_ID_seq = Path_folder_recall + "BaseID_" + std::to_string(nc) + ".seq"; 
            std::ofstream BaseIDSeqOutput(Path_ID_seq, std::ios::binary);
            BaseIDSeqOutput.write((char * ) Base_ID_seq.data(), nb * sizeof(uint32_t));
            BaseIDSeqOutput.close();
            std::cout << "Save vector ID in seq index to " << Path_ID_seq << "\n";

            HNSWGraph = new hnswlib::HierarchicalNSW(Dimension, nc, GraphM, 2 * GraphM, GraphEf);
            for (size_t i = 0; i < nc; i++){HNSWGraph->addPoint(Centroids.data() + i * Dimension);}
            Path_Graph_info = Path_folder_recall + "Graph_" + std::to_string(nc) + "_" + std::to_string(GraphM) + "_" + std::to_string(GraphEf) + ".info"; 
            Path_Graph_Edge = Path_folder_recall + "Graph_" + std::to_string(nc) + "_" + std::to_string(GraphM) + "_" + std::to_string(GraphEf) + ".edge"; 
            HNSWGraph->SaveEdges(Path_Graph_Edge); HNSWGraph->SaveInfo(Path_Graph_info);
            Path_centroid = Path_folder_recall + "Centroid_" + std::to_string(nc) + ".fvecs"; std::ofstream CentroidOutput(Path_centroid, std::ios::binary);
            uint32_t Dim = Dimension;
            for (size_t i = 0; i < nc; i++){
                CentroidOutput.write((char *) & Dim, sizeof(uint32_t));
                CentroidOutput.write((char *) (Centroids.data() + i * Dimension), Dimension * sizeof(float));
            }
            CentroidOutput.close();
            Trecorder.print_record_time_usage(RecordFile, "Save the base ID and updated graph information");
        }
    }
    OptOutput << "Training process completed with failure times: " + std::to_string(MaxFailureTimes) + "\n"; OptOutput.close();
    }

    OptInput.open(Path_opt_file);
    IndexPreTrained = false;
    while(getline(OptInput, s)){
        if (s == "Training process completed with failure times: " + std::to_string(MaxFailureTimes)){
            IndexPreTrained = true;
        }
        else{
            IndexPreTrained = false;
            OptString = StringSplit(s, " ");
        }
    }
    OptInput.close(); assert(IndexPreTrained);


    std::cout << "\n\nTraining Completed. load the index infomation with the following path\n";

    // Load the centroids and the graph
    nc = std::stoul(OptString[3]);
    std::cout << "The optimal number of cluster is: " << nc << "\n";
    std::cout << "Load Centroid file from: " << OptString[6] << "\n";
    std::cout << "Load Graph edge file from: " << OptString[17] << "\n";
    std::cout << "Load Graph info file from: " << OptString[21] << "\n";

    assert(exists(OptString[6])); assert(exists(OptString[17])); assert(exists(OptString[21])); 

    CentroidHNSW = new hnswlib::HierarchicalNSW(OptString[21], OptString[6], OptString[17]);
    // Load the centroid norms
    std::cout << "Load Centroid norm file from: " << OptString[13] << "\n";
    assert(exists(OptString[13]));
    std::ifstream CenNormInput(OptString[13]);
    CNorms.resize(nc);
    CenNormInput.read((char * ) & nc, sizeof(size_t)); assert(nc == std::stoul(OptString[3]));
    CenNormInput.read((char * ) CNorms.data(), nc * sizeof(float));
    CenNormInput.close();

/*
    for (size_t i = 0; i < nc; i++){
        if (abs(faiss::fvec_norm_L2sqr(CentroidHNSW->getDataByInternalId(i), Dimension) - CNorms[i]) > 1e-5){
            std::cout << "Centroid Norm Error: " << i << " th centroid: correct norm: " << (faiss::fvec_norm_L2sqr(CentroidHNSW->getDataByInternalId(i), Dimension)) << " recorded norm: " << CNorms[i]  << " Error: " << faiss::fvec_norm_L2sqr(CentroidHNSW->getDataByInternalId(i), Dimension) - CNorms[i] << "\n";
            exit(0);
        }
    }
*/
    // Load the PQ quantizer
    std::cout << "Load Product quantizer file from: " << OptString[9] << "\n";
    assert(exists(OptString[9]));
    PQ = faiss::read_ProductQuantizer(OptString[9].c_str());

    // Load the base ID and transfer to inverted index
    std::cout << "Load Base ID in sequence from: " << OptString[24] << "\n";
    assert(exists(OptString[24]));
    std::ifstream BaseIDInput(OptString[24], std::ios::binary);
    BaseIds.resize(nc); for (size_t i = 0; i < nc; i++){BaseIds.resize(0);}

    BaseIDInput.read((char *) Base_ID_seq.data(), nb * sizeof(uint32_t));
    BaseIDInput.close();
    for (uint32_t i = 0; i < nb; i++){
        BaseIds[Base_ID_seq[i]].emplace_back(i);
    }

    // Quantize the base vectors to vector code and save tthe code to index

    std::string Path_base_code = Path_folder_recall + "Base_code_" + std::to_string(nc) + "_" + std::to_string(PQ_M) +  ".code";
    std::string Path_base_norm = Path_folder_recall + "Base_" + std::to_string(nc) + "_" + std::to_string(PQ_M) +  ".norm";

    std::cout << "Save Base code file to: " << Path_base_code << " \nSave Base norm file to: " << Path_base_norm << "\n";

    Retrain = false;

    QuantizeBaseset(Assignment_num_batch, Assignment_batch_size, Path_base, OptString[24], Path_base_code, Path_base_norm);
    return nc;
}


void BIndex::BuildGraph(size_t M, size_t efConstruction, std::string PathGraphInfo, std::string PathGraphEdge, std::string PathCentroid){
    Trecorder.reset();

    if (!Retrain &&exists(PathGraphEdge) && exists(PathGraphInfo)){
        std::cout << "Reading Graph from file" << PathGraphInfo << " from disk" << std::endl;
        CentroidHNSW = new hnswlib::HierarchicalNSW(PathGraphInfo, PathCentroid, PathGraphEdge);
        std::cout << "The first centroid: \n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << CentroidHNSW->getDataByInternalId(0)[i] << " ";
        }
        std::cout << "\n";
        std::cout << "The last centroid: \n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << CentroidHNSW->getDataByInternalId(nc-1)[i] << " ";
        }
        std::cout << "\n";

        return;
    }

    std::cout << "Training Graph Index" << std::endl;
    std::vector<float> Centroids(nc * Dimension);
    std::ifstream CentroidInput(PathCentroid, std::ios::binary);
    std::cout << "Load centroids from " << PathCentroid << " for graph construction\n";
    readXvec<float>(CentroidInput, Centroids.data(), Dimension, nc, true, true);

    CentroidHNSW = new hnswlib::HierarchicalNSW(Dimension, nc, M, 2* M, efConstruction);
    for (size_t i = 0; i < nc; i++){
        if (i % size_t(nc / 10) == 0){
            std::cout << "Graph Constructed " << i << " in " << nc << " Centroids\n" ;
        }
        CentroidHNSW->addPoint(Centroids.data() + i * Dimension);
    }

    if (Saving){
        CentroidHNSW->SaveInfo(PathGraphInfo);
        CentroidHNSW->SaveEdges(PathGraphEdge);
    }
    Trecorder.print_time_usage("Graph Construction Completed");
    return;
}



void BIndex::AssignBaseset(size_t NumBatch, size_t BatchSize, std::string PathBase, std::string PathBaseIDInv, std::string PathBaseIDSeq){
    
    if (!Retrain && exists(PathBaseIDSeq) && exists(PathBaseIDInv)){
        std::cout << "Load Baseset ID from " << PathBaseIDInv << std::endl;
        std::ifstream BaseIDInput(PathBaseIDInv, std::ios::binary);
        BaseIds.resize(nc);
        uint32_t ClusterSize;
        for (size_t i = 0; i < nc; i++){
            BaseIDInput.read((char *) & ClusterSize, sizeof(uint32_t));
            BaseIds[i].resize(ClusterSize);
            BaseIDInput.read((char *) BaseIds[i].data(), ClusterSize * sizeof(uint32_t));
        }
        BaseIDInput.close();
        return;
    }

    std::cout << "Assign Base Vectors Index" << std::endl;
    Trecorder.reset();
    std::vector<float> BaseSet(Dimension * BatchSize);
    std::ifstream BaseInput(PathBase, std::ios::binary);
    std::vector<uint32_t> AssignIds(NumBatch * BatchSize);
    for (size_t i = 0; i < NumBatch; i++){
        std::cout << "BaseSet Assignment Completed " << i << " in " << NumBatch << " Batches\n";
        readXvecFvec<DataType>(BaseInput, BaseSet.data(), Dimension, BatchSize, true, true);
        AssignVector(BatchSize, BaseSet.data(), AssignIds.data() + i * BatchSize);
        Trecorder.print_time_usage("Assign Batch: " + std::to_string(i + 1) + " / " + std::to_string(NumBatch));
    }

    BaseIds.resize(nc);
    for (uint32_t i = 0; i < NumBatch * BatchSize; i++){
        if (AssignIds[i] >= nc){
            std::cout << "Wrong Assignment " << AssignIds[i] << " in " << nc << " clusters" << std::endl;
            exit(0);
        }
        BaseIds[AssignIds[i]].emplace_back(i);
    }

    Trecorder.print_time_usage("Load Assign Id to Index");
    std::ofstream BaseIDInvOutput(PathBaseIDInv, std::ios::binary);
    for (size_t i = 0; i < nc; i++){
        uint32_t ClusterSize = BaseIds[i].size();
        BaseIDInvOutput.write((char *) & ClusterSize, sizeof(uint32_t));
        BaseIDInvOutput.write((char *) BaseIds[i].data(), ClusterSize * sizeof(uint32_t));
    }
    BaseIDInvOutput.close();
    std::cout << "Save vector ID in inverted index to " << PathBaseIDInv << "\n";
    std::ofstream BaseIDSeqOutput(PathBaseIDSeq, std::ios::binary);
    BaseIDSeqOutput.write((char * ) AssignIds.data(), NumBatch * BatchSize * sizeof(uint32_t));
    BaseIDInvOutput.close();
    std::cout << "Save vector ID in seq index to " << PathBaseIDSeq << "\n";
    return;
}


std::string BIndex::ClusterFeature(){
    std::cout << "The feature of cluster size: " << std::endl;
    double MaxSize, MinSize, SumSize, DiffSize, AvgSize, SearchCost = 0;
    MaxSize = 0; MinSize = nb; AvgSize = nb / nc; SumSize = 0; DiffSize = 0;

    for (size_t i = 0 ; i < nc; i++){
        if (BaseIds[i].size() > MaxSize){
            MaxSize = BaseIds[i].size();
        }
        if (BaseIds[i].size() < MinSize){
            MinSize = BaseIds[i].size();
        }
        SumSize += BaseIds[i].size();
        DiffSize += std::abs(double(BaseIds[i].size()) - AvgSize);
        SearchCost += BaseIds[i].size() * BaseIds[i].size();
        //std::cout << BaseIds[i].size() << " ";
    }
    std::cout << std::endl;

    if(SumSize != nb){
        std::cout << "Sum Size: " << SumSize << " is different from nb: " << nb << std::endl;
    };
    std::string Result = "nc: " + std::to_string(nc) + " MaxSize: " + std::to_string(size_t(MaxSize)) + " MinSize: " + std::to_string(size_t(MinSize)) + " Avg Size: " + std::to_string(size_t(SumSize / nc )) + " AvgDiffProp: " + std::to_string(DiffSize / nc) + "/" + std::to_string(size_t(AvgSize)) + " InnerSearchCost: " + std::to_string(size_t(SearchCost / nb)) + "\n";

    std::cout << Result << std::endl;
    return Result;
}


void BIndex::TrainQuantizer(size_t PQTrainSize, std::string PathLearn, std::string PathPQ, std::string PathOPQ){
    Trecorder.reset();

    if (!Retrain && exists(PathPQ) && ((UseOPQ && exists(PathOPQ)) || (!UseOPQ)) ){
        std::cout << "PQ and OPQ Quantizer Exist, Load PQ from " << PathPQ <<  std::endl;
        PQ = faiss::read_ProductQuantizer(PathPQ.c_str());
        if (UseOPQ){
            std::cout << " Read OPQ from " << PathOPQ << std::endl;
            OPQ = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(PathOPQ.c_str()));
        }

        Trecorder.print_time_usage("Load PQ and OPQ Quantizer");
        return;
    }

    std::cout << "Training PQ and OPQ quantizer" << std::endl;
    std::vector<float> PQTrainSet(PQTrainSize * Dimension);
    std::cout << "Sampling " << PQTrainSize << " from " << nt << " for PQ training" << std::endl;
    SampleSet<DataType>(PQTrainSet.data(), PathLearn, PQTrainSize, nt, Dimension);

    std::vector<uint32_t> DataID(PQTrainSize);

    AssignVector(PQTrainSize, PQTrainSet.data(), DataID.data());
    std::vector<float> Residual(PQTrainSize * Dimension);

#pragma omp parallel for
    for (size_t i = 0; i < PQTrainSize; i++){
        ComupteResidual(Dimension, PQTrainSet.data() + i * Dimension, CentroidHNSW->getDataByInternalId(DataID[i]), Residual.data() + i * Dimension);
    }

    if (UseOPQ){
        faiss::ProductQuantizer * PQTrain  = new faiss::ProductQuantizer(Dimension, M_PQ, CodeBits);


        faiss::OPQMatrix * OPQTrain = new faiss::OPQMatrix(Dimension, M_PQ);
        OPQTrain->verbose = false;
        OPQTrain->niter = 50;
        OPQTrain->pq = PQTrain;
        OPQTrain->train(PQTrainSize, Residual.data());
        OPQ = OPQTrain;
        DoOPQ(PQTrainSize, Residual.data());
        if (Saving){
            faiss::write_VectorTransform(OPQTrain, PathOPQ.c_str());
        }
        OPQ = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(PathOPQ.c_str()));
        if(Saving){
            faiss::write_ProductQuantizer(PQTrain, PathPQ.c_str());
        }
        PQ = faiss::read_ProductQuantizer(PathPQ.c_str());
        Trecorder.print_time_usage("OPQ Trainining Completed");
    }

    else{
        faiss::ProductQuantizer * PQTrain  = new faiss::ProductQuantizer(Dimension, M_PQ, CodeBits);

        PQTrain->verbose = true;
        PQTrain->train(PQTrainSize, Residual.data());

        if(Saving){
            faiss::write_ProductQuantizer(PQTrain, PathPQ.c_str());
        }
        PQ = faiss::read_ProductQuantizer(PathPQ.c_str());
        std::cout << "Save PQ quantizer to: " << PathPQ << std::endl;
        Trecorder.print_time_usage("Train PQ Quantizer");
    }
    return;
}


void BIndex::QuantizeBaseset(size_t NumBatch, size_t BatchSize, std::string PathBase, std::string PathBaseIDSeq, std::string PathBaseCode, std::string PathBaseNorm, std::string PathOPQCentroids){
    if (!Retrain && exists(PathBaseCode) && exists(PathBaseNorm) && ((UseOPQ && exists(PathOPQCentroids)) || (!UseOPQ)) ){
        std::cout << "Loading Base Vector PQ Codes" << std::endl;
        std::ifstream BaseCodeInput(PathBaseCode, std::ios::binary);
        std::ifstream BaseNormInput(PathBaseNorm, std::ios::binary);

        BaseCodes.resize(nb * PQ->code_size);
        BaseCodeInput.read((char * ) BaseCodes.data(), nb * PQ->code_size * sizeof(uint8_t));
        BaseCodeInput.close();

        BaseNorms.resize(nb);
        BaseNormInput.read((char *) BaseNorms.data(), nb * sizeof(float));
        BaseNormInput.close();

        if (UseOPQ){
            std::ifstream OPQCentroidsInput(PathOPQCentroids, std::ios::binary);
            uint32_t Dim;
            for (size_t i = 0; i < nc; i++){
                OPQCentroidsInput.read((char * ) & Dim, sizeof(uint32_t)); assert(Dim == Dimension);
                OPQCentroidsInput.read((char *) CentroidHNSW->getDataByInternalId(i), Dimension * sizeof(float));
            }
            OPQCentroidsInput.close();
        }
        std::cout << "Load Vector Codes Completed" << std::endl;
        return;
    }

    BaseCodes.resize(nb * PQ->code_size);
    BaseNorms.resize(nb);

    std::vector<float> OPQCentroids;
    if (UseOPQ){
        OPQCentroids.resize(nc * Dimension);
        std::cout << "Appling OPQ Rotation" << std::endl;
        for (size_t i = 0; i < nc; i++){
            OPQ->apply_noalloc(1, CentroidHNSW->getDataByInternalId(i), OPQCentroids.data() + i * Dimension);
        }
    }

    std::ifstream BaseInput(PathBase, std::ios::binary);
    std::vector<float> BaseSet(Dimension *BatchSize);
    std::ifstream BaseIDSeqInput(PathBaseIDSeq, std::ios::binary);
    std::vector<uint32_t> BaseSetID(BatchSize);
    std::cout << "Quantizing the base vectors " << std::endl;

/*
    std::vector<float> BaseSetTest(nb * Dimension);
    std::vector<uint32_t> BaseIDTest(nb);
    std::ifstream BaseInputTest(PathBase, std::ios::binary);
    std::ifstream BaseIDSeqInputTest(PathBaseIDSeq, std::ios::binary);
    readXvecFvec<float>(BaseInputTest, BaseSetTest.data(), Dimension, nb, false, false);
    BaseIDSeqInputTest.read((char *) BaseIDTest.data(), nb * sizeof(uint32_t));
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
        auto result = CentroidHNSW->searchKnn(BaseSetTest.data() + i * Dimension, 1);
        assert(BaseIDTest[i] == result.top().second);
    }
*/

    for (size_t i = 0; i <  NumBatch; i++){
        std::cout << "Quantize BaseSet Completed " << i << " batch in " << NumBatch << " Batches\n";
        readXvecFvec<DataType>(BaseInput, BaseSet.data(), Dimension, BatchSize, true, true);
        BaseIDSeqInput.read((char *)BaseSetID.data(), BatchSize * sizeof(uint32_t));
        QuantizeVector(BatchSize, BaseSet.data(), BaseSetID.data(), BaseCodes.data() + i * BatchSize * PQ->code_size, BaseNorms.data() + i * BatchSize, OPQCentroids.data());
    }

    std::ofstream BaseCodeOutput(PathBaseCode, std::ios::binary);
    BaseCodeOutput.write((char * )BaseCodes.data(), nb * PQ->code_size * sizeof(uint8_t));
    BaseCodeOutput.close();
    std::ofstream BaseNormOutput(PathBaseNorm, std::ios::binary);
    BaseNormOutput.write((char *)BaseNorms.data(), nb * sizeof(float));
    BaseNormOutput.close();

    if (UseOPQ){
        std::ofstream OPQCentroidOutput(PathOPQCentroids, std::ios::binary);
        for (size_t i = 0; i < nc; i++){
            OPQCentroidOutput.write((char *) & Dimension, sizeof(uint32_t));
            OPQCentroidOutput.write((char *) (OPQCentroids.data() + i * Dimension), Dimension * sizeof(float));
            memcpy(CentroidHNSW->getDataByInternalId(i), OPQCentroids.data() + i * Dimension, Dimension * sizeof(float));
        }
        OPQCentroidOutput.close();
    }
    return;
}


std::string BIndex::QuantizationFeature(std::string PathBase, std::string PathBaseIDSeq){
    size_t MaxTestSize = 10e4;
    std::cout <<  std::endl << "Test the Quantization Error of constructed PQ Quantizer" << std::endl;
    std::vector<float> RecoverBase(MaxTestSize * Dimension);
    std::vector<float> RecoverResidual(MaxTestSize * Dimension);
    std::vector<float> BaseSet(Dimension * MaxTestSize);
    std::vector<float> Residual(Dimension * MaxTestSize);

    std::vector<uint32_t> BaseSetID(MaxTestSize);
    std::ifstream BaseIDSeqInput(PathBaseIDSeq, std::ios::binary);
    std::ifstream BaseInput(PathBase, std::ios::binary);

    BaseIDSeqInput.read((char *)BaseSetID.data(), MaxTestSize * sizeof(uint32_t));
    readXvecFvec<DataType>(BaseInput, BaseSet.data(), Dimension, MaxTestSize, true, true);
    if (UseOPQ) DoOPQ(MaxTestSize, BaseSet.data());
    PQ->decode(BaseCodes.data(), RecoverResidual.data(), MaxTestSize);
    for (size_t i = 0; i < MaxTestSize; i++){
        faiss::fvec_madd(Dimension, RecoverResidual.data() + i * Dimension, 1.0, CentroidHNSW->getDataByInternalId(BaseSetID[i]), RecoverBase.data()+i * Dimension);
        faiss::fvec_madd(Dimension, BaseSet.data() + i * Dimension, -1.0, CentroidHNSW->getDataByInternalId(BaseSetID[i]), Residual.data() + i * Dimension);
    }
    for (size_t i = 0; i < 5; i++){
        for (size_t j = 0; j < Dimension; j++){
            std::cout << RecoverResidual[i * Dimension + j] << " ";
        }
        std::cout << std::endl;
    }
    for (size_t i = 0; i < 5; i++){
        for (size_t j = 0; j < Dimension; j++){
            std::cout << Residual[i * Dimension + j] << " ";
        }
        std::cout << std::endl;
    }
    float BCDist = 0; float AvgDist = 0; double AvgBaseNorm = 0; double AvgResidualDist = 0;
    for (size_t i = 0; i < MaxTestSize; i++){
        BCDist += faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, CentroidHNSW->getDataByInternalId(BaseSetID[i]), Dimension);
        AvgDist += faiss::fvec_L2sqr(BaseSet.data()+ i * Dimension, RecoverBase.data()+ i * Dimension, Dimension);
        AvgBaseNorm += faiss::fvec_norm_L2sqr(BaseSet.data() + i * Dimension, Dimension);
        AvgResidualDist += faiss::fvec_L2sqr(Residual.data() + i * Dimension, RecoverResidual.data() + i * Dimension, Dimension);
    }
    std::string Result = "Avg Base Norm: " + std::to_string( AvgBaseNorm / MaxTestSize) + " Avg QC Dist: " + std::to_string( BCDist / MaxTestSize) + " Avg Dist Error: " + std::to_string( AvgDist / MaxTestSize) + " Avg Residual Dist: " + std::to_string( AvgResidualDist / MaxTestSize);
    std::cout << Result << std::endl;
    return Result;
}


void BIndex::QuantizeVector(size_t N, float * BaseData, uint32_t * BaseID, uint8_t * BaseCode, float * BaseNorm, float * OPQCentroids){
    std::vector<float> Residual(N * Dimension);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        ComupteResidual(Dimension, BaseData + i * Dimension, CentroidHNSW->getDataByInternalId(BaseID[i]), Residual.data() + i * Dimension);
    }

    if (UseOPQ){
        DoOPQ(N, Residual.data());
    }

    PQ->compute_codes(Residual.data(), BaseCode, N);

    std::vector<float> RecoveredResidual(N * Dimension);
    PQ->decode(BaseCode, RecoveredResidual.data(), N);
    std::vector<float> RecoveredBase(N * Dimension);

    // The loss in PQ quantization
    uint32_t ErrorSample = 10000;
    std::cout << "\nThe base vector residual norm: " << faiss::fvec_norm_L2sqr(Residual.data(), Dimension * ErrorSample) / ErrorSample << std::endl;
    std::cout << "The PQ quantization error of residual: " << faiss::fvec_L2sqr(RecoveredResidual.data(), Residual.data(), Dimension * ErrorSample) / ErrorSample << "\n\n";


    /*
    for (size_t i = 0; i < 5; i++){
        for (size_t j = 0; j < Dimension; j++){
            std::cout << RecoveredResidual[i * Dimension + j] << " ";
        }
        std::cout << std::endl;
    }

    for (size_t i = 0; i < 5; i++){
        for (size_t j = 0; j < Dimension; j++){
            std::cout << Residual[i * Dimension + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    if (UseOPQ){
#pragma omp parallel for
        for (size_t i = 0; i < N; i++){
            ReverseResidual(Dimension, RecoveredResidual.data() + i * Dimension, OPQCentroids + BaseID[i] * Dimension, RecoveredBase.data() + i * Dimension);
        }
    }
    else{
#pragma omp parallel for
        for (size_t i = 0; i < N; i++){
            ReverseResidual(Dimension, RecoveredResidual.data() + i * Dimension, CentroidHNSW->getDataByInternalId(BaseID[i]), RecoveredBase.data() + i * Dimension);
        }
    }
    faiss::fvec_norms_L2sqr(BaseNorm, RecoveredBase.data(), Dimension, N);


    /*
    std::vector<float> TempBaseNorm(N);
    std::vector<float> TempRecoverResidual(N * Dimension);
    std::vector<float> TempBase(N * Dimension);
    OPQ->reverse_transform(N, OPQResidual.data(), TempRecoverResidual.data());
    for (size_t i = 0; i < N; i++){
        ReverseResidual(Dimension, TempRecoverResidual.data() + i * Dimension, CentroidHNSW->getDataByInternalId(BaseID[i]), TempBase.data() + i * Dimension);
    }
    faiss::fvec_norms_L2sqr(TempBaseNorm.data(), TempBase.data(), Dimension, N);

    std::vector<float> CorrectNorm(N);
    faiss::fvec_norms_L2sqr(CorrectNorm.data(), BaseData, Dimension, N);

    std::cout << "Computed Norms: ";
    for (size_t i = 0; i < 20; i++){
        std::cout << BaseNorm[i] << " ";
    }

    std::cout << std::endl << "Recovered Computed Norms: ";

    for (size_t i = 0; i < 20; i++){
    std::cout << "Computed Norms: ";
    for (size_t i = 0; i < 20; i++){
        std::cout << BaseNorm[i] << " ";
    }

    std::cout << std::endl << "Recovered Computed Norms: ";

    for (size_t i = 0; i < 20; i++){
        std::cout << TempBaseNorm[i] << " ";
    }

    std::cout << std::endl << "Correct Norms: ";
    for (size_t i = 0; i < 20; i++){
        std::cout << CorrectNorm[i] << " ";
    }
    exit(0);
    */
}


// For inverted index based construction, the search mechanism on compressed vectors can be expressed as:
// ||Q - （C + PQ）||^2 = ||(Q - C) - PQ||^2 = ||Q - C||^2 - 2 * PQ * (Q - C) + ||PQ||^2 
// = ||Q - C||^2 + ||PQ|||^2 + 2 * PQ * C + ||C||^2 - 2 * PQ * Q - ||C||^2 = ||Q - C||^2 + ||PQ + C||^2 - ||C||^2 - 2 * Q * PQ
// With OPQ, we do OPQ to the query

uint32_t BIndex::Search(size_t K, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem){
    const std::string BaseFile = "/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_base.fvecs";
    std::ifstream BaseInput(BaseFile, std::ios::binary);

    if (UseOPQ){
        OPQ->apply(1, Query);
    }

    Trecorder.reset();
    auto ResultQueue = CentroidHNSW->searchBaseLayer(Query, EfSearch);
    
    for (size_t i = 0; i < EfSearch; i++){
        QCDist[EfSearch - i - 1] = ResultQueue.top().first;
        QCID[EfSearch - i - 1] = ResultQueue.top().second;
        ResultQueue.pop();
    }
    Trecorder.recordTimeConsumption1();

    faiss::maxheap_heapify(K, QueryDists, QueryIds);

    PQ->compute_inner_prod_table(Query, PQTable.data());
    Trecorder.recordTimeConsumption2();
    
    VisitedItem = 0;


    for (size_t i = 0; i < EfSearch; i++){
        
        uint32_t ClusterID = QCID[i];

        float CDist = QCDist[i] - CNorms[ClusterID];
        
        for (size_t j = 0; j < BaseIds[ClusterID].size(); j++){
            uint32_t BaseID = BaseIds[ClusterID][j];
            
            float VNorm = BaseNorms[BaseID];
            float ProdDist = 0;
            for (size_t k = 0; k < PQ->code_size; k++){
                ProdDist += PQTable[PQ->ksub * k + BaseCodes[BaseID * PQ->code_size + k]];
            }
            float Dist = CDist + VNorm - 2 * ProdDist;


            std::vector<float> BaseVector(Dimension);
            BaseInput.seekg(BaseID * (Dimension * sizeof(float) + sizeof(uint32_t)) + sizeof(uint32_t), std::ios::beg);
            BaseInput.read((char *) BaseVector.data(), Dimension * sizeof(float));
            auto Result = CentroidHNSW->searchKnn(BaseVector.data(), 1);
            std::cout << "ID: " << BaseID << " ClusterID: " << ClusterID << " Correct ClusterID: " << Result.top().second << "\n";
            std::cout << "Correct QC Dist: " << faiss::fvec_L2sqr(Query, CentroidHNSW->getDataByInternalId(ClusterID), Dimension) << " Result: " << QCDist[i] << std::endl;
            std::cout << "Correct CNorm: " << faiss::fvec_norm_L2sqr(CentroidHNSW->getDataByInternalId(ClusterID), Dimension) << " Result: " << CNorms[ClusterID] << std::endl; 

            std::vector<float> Residual(Dimension);
            std::vector<uint8_t> ResidualCode(PQ->code_size);
            faiss::fvec_madd(Dimension, BaseVector.data(), -1.0, CentroidHNSW->getDataByInternalId(ClusterID), Residual.data());
            PQ->compute_code(Residual.data(), ResidualCode.data());
            for (size_t k = 0; k < PQ->code_size; k++){
                if (ResidualCode[k] != BaseCodes[BaseID * PQ->code_size + k]){
                    std::cout << int(ResidualCode[k]) << " " << int(BaseCodes[BaseID * PQ->code_size + k]) << " | ";
                }
            }

            std::vector<float> RecoveredResidual(Dimension);
            PQ->decode(ResidualCode.data(), RecoveredResidual.data(),1);
            float CorrectProdDist = faiss::fvec_inner_product(RecoveredResidual.data(), Query, Dimension);
            std::cout << "Correct Prod Dist: " << CorrectProdDist << " Result ProdDist: " << ProdDist << std::endl;
            std::vector<float> OPQBase(Dimension);
            faiss::fvec_madd(Dimension, RecoveredResidual.data(), 1.0, CentroidHNSW->getDataByInternalId(ClusterID), OPQBase.data());
            float CorrectBaseNorm = faiss::fvec_norm_L2sqr(OPQBase.data(), Dimension);
            float CorrectResultDist = faiss::fvec_L2sqr(OPQBase.data(), Query, Dimension);

            std::cout << "Precise Base Norm " << faiss::fvec_norm_L2sqr(BaseVector.data(), Dimension) << " Recorded Base Norm " << BaseNorms[BaseIds[QCID[i]][j]] <<  " Correct Base Norm: " << CorrectBaseNorm << std::endl;
            std::cout << "Precise Dist " << faiss::fvec_L2sqr(Query, BaseVector.data(), Dimension) << " Result Dist: " << Dist <<  " Correct Result Dist: " << CorrectResultDist << std::endl;
            std::cout << "Norm of the QC residual: " << faiss::fvec_norm_L2sqr(Residual.data(), Dimension) << " Norm of the PQ residual: " << faiss::fvec_norm_L2sqr(RecoveredResidual.data(), Dimension) << " Distance between two residual: " << faiss::fvec_L2sqr(Residual.data(), RecoveredResidual.data(), Dimension) << "\n";

            if (Dist < QueryDists[0]){
                faiss::maxheap_pop(K, QueryDists, QueryIds);
                faiss::maxheap_push(K, QueryDists, QueryIds, Dist, BaseID);
            }
        }
        exit(0);
        

        VisitedItem += BaseIds[ClusterID].size();
        if (VisitedItem > MaxItem){
            break;
        }
    }
    Trecorder.recordTimeConsumption3();
    return VisitedItem;
}

uint32_t BIndex::SearchMulti(size_t nq, size_t K, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem){
    uint32_t SumVisitedItem = 0;
    Trecorder.reset();

    if (UseOPQ){
        OPQ->apply(nq, Query);
    }
    for(size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto ResultQueue = CentroidHNSW->searchBaseLayer(Query + QueryIdx * Dimension, EfSearch);
        for (size_t i = 0; i < EfSearch; i++){
            QCDist[QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().first;
            QCID  [QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().second;
            ResultQueue.pop();
        }
    }
    Trecorder.recordTimeConsumption1();

    Trecorder.reset();
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        PQ->compute_inner_prod_table(Query + QueryIdx * Dimension, PQTable.data() + QueryIdx * PQ->ksub * PQ->M);
        faiss::maxheap_heapify(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
    }
    Trecorder.recordTimeConsumption2();
    

    Trecorder.reset();
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        
        VisitedItem = 0;
        for (size_t i = 0; i < EfSearch; i++){
            uint32_t ClusterID = QCID[QueryIdx * EfSearch + i];

            float CDist = QCDist[QueryIdx * EfSearch + i] - CNorms[ClusterID];
            
            for (size_t j = 0; j < BaseIds[ClusterID].size(); j++){
                uint32_t BaseID = BaseIds[ClusterID][j];
                float VNorm = BaseNorms[BaseID];
                float ProdDist = 0;
                for (size_t k = 0; k < PQ->code_size; k++){
                    ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodes[BaseID * PQ->code_size + k]];
                }
                float Dist = CDist + VNorm - 2 * ProdDist;
                
                if (Dist < QueryDists[QueryIdx * K]){
                    faiss::maxheap_pop(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
                    faiss::maxheap_push(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K, Dist, BaseID);
                }
            }
            VisitedItem += BaseIds[ClusterID].size();
            if (VisitedItem > MaxItem){
                break;
            }
        }
        SumVisitedItem += VisitedItem;
    }
    Trecorder.recordTimeConsumption3();

    return SumVisitedItem;
}





std::string BIndex::Eval(std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, 
size_t NumRecall, size_t NumPara, size_t * RecallK, size_t * MaxItem, size_t * EfSearch){
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

    //PQTable.resize(PQ->ksub * PQ->M);
    PQTable.resize(nq * PQ->ksub * PQ->M);

    recall_recorder Rrecorder = recall_recorder();
    std::string Result = "";

    std::vector<float> TimeConsumption(NumPara);
    std::vector<float> RecallResult(NumPara);
    std::vector<float> Recall1Result(NumPara);

    // Cannot change the query in retrieval
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

            // Complete the search by queries
/*
            QCDist.resize(EfSearch[j]);
            QCID.resize(EfSearch[j]);
            for (size_t k = 0; k < nq; k++){
                NumVisitedItem += Search(RecallK[i], Query.data() + k * Dimension, QueryIds.data() + k * RecallK[i], QueryDists.data() + k * RecallK[i], EfSearch[j], MaxItem[j]);
                WholeTime += Trecorder.TempDuration3;
                CenSearchTime += Trecorder.TempDuration1;
                TableTime += Trecorder.TempDuration2 - Trecorder.TempDuration1;
                VectorSearchTime += Trecorder.TempDuration3 - Trecorder.TempDuration2;
            }
            Result +=  "Single Whole Time (ms): " + std::to_string(WholeTime / (1000*nq)) + " CenTime:" + std::to_string(CenSearchTime / (1000 * nq)) + " TableTime: " + std::to_string(TableTime/(nq * 1000)) + " VecTime: " +  std::to_string(VectorSearchTime / (1000 * nq)) +" NumVisitedItem " + std::to_string(float(NumVisitedItem) / nq) +  " EfSearch: " + std::to_string(EfSearch[j]) + " MaxItem: " + std::to_string(MaxItem[j]) + " \n";
*/

           // Complete the search by stages
///*
            QCDist.resize(nq * EfSearch[j]);
            QCID.resize(nq * EfSearch[j]);
            time_recorder TempRecorder = time_recorder();

            NumVisitedItem = SearchMulti(nq, RecallK[i], Query.data(), QueryIds.data(), QueryDists.data(), EfSearch[j], MaxItem[j]);
            
            WholeTime = TempRecorder.getTimeConsumption();
            CenSearchTime = Trecorder.TempDuration1;
            TableTime = Trecorder.TempDuration2;
            VectorSearchTime = Trecorder.TempDuration3;
            Result +=  "Multi  Whole Time (ms): " + std::to_string((WholeTime) / (1000*nq)) + " CenTime:" + std::to_string(CenSearchTime / (1000 * nq)) + " TableTime: " + std::to_string(TableTime/(nq * 1000)) + " VecTime: " +  std::to_string(VectorSearchTime / (1000 * nq)) +" NumVisitedItem " + std::to_string(float(NumVisitedItem) / nq) +  " EfSearch: " + std::to_string(EfSearch[j]) + " MaxItem: " + std::to_string(MaxItem[j]) + " \n";
//*/
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
            float Recall1 = float(Correct) / (nq * RecallK[i]);
            float Recall = float(Correct1) / (nq);
            TimeConsumption[j] = WholeTime;
            RecallResult[j] = Recall;
            Recall1Result[j] = Recall1;
            Result += "Recall" + std::to_string(RecallK[i]) + "@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall1) + " Recall@" + std::to_string(RecallK[i]) + ": " + std::to_string(Recall) + "\n";
            Rrecorder.print_recall_performance(nq, Recall, RecallK[i], SearchMode, EfSearch[j], MaxItem[j]);
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
    }

    return Result;
}


void BIndex::AssignVector(size_t N, float * BaseData, uint32_t * DataID){
#pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        auto Result = CentroidHNSW->searchKnn(BaseData + i * Dimension, 1);
        DataID[i] = Result.top().second;
    }
    return;
}

void BIndex::DoOPQ(size_t N, float * Dataset){
    std::vector<float> CopyDataset(N * Dimension);
    memcpy(CopyDataset.data(), Dataset, N * Dimension * sizeof(float));
    OPQ->apply_noalloc(N, CopyDataset.data(), Dataset);
    return;
}

float BIndex::QuantizationError(faiss::ProductQuantizer * PQ, float * TestVector, size_t N){
    std::vector<uint8_t> VectorCodes(PQ->code_size * N);
    PQ->compute_codes(TestVector, VectorCodes.data(), N);
    std::vector<float> RecoverVector(N * Dimension);
    PQ->decode(VectorCodes.data(), RecoverVector.data(), N);
    float Dist = 0;
    for (size_t i = 0; i < N * Dimension; i++){Dist += (TestVector[i] - RecoverVector[i]) * (TestVector[i] - RecoverVector[i]);}
    return Dist / N;
}


std::string BIndex::NeighborCost(size_t Scale, std::string PathQuery, std::string PathGt, size_t nq, size_t ngt, size_t EfSearch, float TargetRecall){
    std::cout << "Start Evaluate the query" << std::endl;
    std::vector<float> Queryset (nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream GTInput(PathGt, std::ios::binary);
    readXvec<uint32_t>(GTInput, GT.data(), ngt, nq, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<float>(QueryInput, Queryset.data(), Dimension, nq, true, true);

    GTInput.close(); QueryInput.close();
    size_t RecallK = 10;
    float NumClusters = nc / 1000;
    float SumGt = 0; float SumVisitedVec = 0;
    bool IncreaseDistProp = true;
    std::cout << "Check the cost of query\n";
    while(IncreaseDistProp){
        std::vector<size_t> QueryGt(nq,0);
        std::vector<size_t> QueryVisitedVec(nq, 0);
        SumGt = 0; SumVisitedVec = 0;
#pragma omp parallel for
        for (size_t i = 0; i < nq; i++){
            float * Query = Queryset.data() + i * Dimension;

            std::vector<float> QueryDist(EfSearch);
            std::vector<uint32_t> QueryLabel(EfSearch);
            auto ResultQueue = CentroidHNSW->searchKnn(Query, EfSearch);
            for (size_t j = 0; j < EfSearch; j++){
                QueryDist[EfSearch - j - 1] = ResultQueue.top().first;
                QueryLabel[EfSearch - j - 1] = ResultQueue.top().second;
                ResultQueue.pop();
            }

            std::unordered_set<uint32_t> GtSet;
            for (size_t j = 0; j < RecallK; j++){
                GtSet.insert(GT[ngt * i + j]);
            }

            for (size_t j = 0; j < NumClusters; j++){
                uint32_t ClusterID = QueryLabel[j];
                QueryVisitedVec[i] += BaseIds[ClusterID].size();

                for (size_t k = 0; k < BaseIds[ClusterID].size(); k++){
                    if (GtSet.count(BaseIds[ClusterID][k]) != 0){
                        QueryGt[i] ++;
                    }
                }
            }
        }

        for (size_t i = 0; i < nq; i++){
            SumGt += QueryGt[i];
            SumVisitedVec += QueryVisitedVec[i];
        }

        std::cout <<  "Num of visited clusters: " << (NumClusters) << " Average VisitedGt: "  << (SumGt / nq) << " Candidate List Size: " << (Scale * SumVisitedVec / nq)  <<  "\n";
        NumClusters += 1;
        if (SumGt / nq > TargetRecall){IncreaseDistProp = false;}
    }

    std::string result =  "Final Result: EfSearch: " + std::to_string(EfSearch) + " Num of visited clusters: " + std::to_string(NumClusters) + " Average VisitedGt: "  + std::to_string(SumGt / nq) + " VisitedVec: " + std::to_string(Scale * SumVisitedVec / nq)  + "\n";
    std::cout << result;
    return result;
}



