#include "./Index/BIndex.h"
#include "./parameters/BillionScale/ParameterResults.h"


int main(){

    Retrain = false;
    std::string PathRecord = PathFolder + Dataset + "/CandidateListSize" + NCString + ".txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }

    NCString    = std::to_string(nc);
    PathCentroid     = PathFolder + Dataset + "/Centroids_" + NCString + ".fvecs";
    PathIniCentroid  = PathFolder + Dataset + "/Centroids_" + NCString + "_" + std::to_string(IniM)+"_Ini.fvecs";

    PathBaseIDInv    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Seq";
    PathBaseIDSeq    = PathFolder + Dataset + "/BaseID_nc" + NCString + "_Inv";

    PathCentroidNorm = PathFolder + Dataset + "/CentroidNorms_" + NCString;

    PathLearnCentroidNorm = PathFolder + Dataset + "/CentroidNormsLearn_" + NCString;
    PathGraphInfo    = PathFolder + Dataset + "/HNSW_" + NCString + "_Info";
    PathGraphEdges   = PathFolder + Dataset + "/HNSW_" + NCString + "_Edge";


    std::vector<float> TrainSet(CTrainSize * Dimension);
    std::ifstream LearnInput(PathLearn, std::ios::binary);
    readXvecFvec<DataType> (LearnInput, TrainSet.data(), Dimension, CTrainSize, true, true);
    std::vector<float> QuerySet(nq * Dimension);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<DataType> (QueryInput, QuerySet.data(), Dimension, nq, true, true);
    std::ofstream IDOutput(PathTrainGt, std::ios::binary);
    std::ofstream DistOutput(PathTrainDist, std::ios::binary);

    std::vector<float> QueryVectorDist(CTrainSize);

    time_recorder TRecorder = time_recorder();
    for (size_t i = 0; i < nq; i++){
#pragma omp parallel for
        for (size_t j = 0; j < CTrainSize; j++){
            QueryVectorDist[j] = faiss::fvec_L2sqr(QuerySet.data() + i * Dimension, TrainSet.data() + j * Dimension, Dimension);
        }

        auto comp = [](std::pair<float, uint32_t> Element1, std::pair<float, uint32_t> Element2){return Element1.first < Element2.first;};
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(comp)> QueryDistQueue(comp);

        for (uint32_t j = 0; j < CTrainSize; j++){
            if (QueryDistQueue.size() < ngt){
                QueryDistQueue.emplace(std::make_pair(QueryVectorDist[j], j));
            }
            else if(QueryVectorDist[j] < QueryDistQueue.top().first){
                assert(!QueryDistQueue.empty());
                QueryDistQueue.pop();
                QueryDistQueue.emplace(std::make_pair(QueryVectorDist[j], j));
            }
        }
        IDOutput.write((char *) & ngt, sizeof(uint32_t));
        DistOutput.write((char *) & ngt, sizeof(uint32_t));
        std::vector<uint32_t> QueryGt(ngt);
        std::vector<float> QueryDist(ngt);
        for (size_t j = 0; j < ngt; j++){
            QueryGt[ngt - j - 1] = QueryDistQueue.top().second;
            QueryDist[ngt - j - 1] = QueryDistQueue.top().first;
            QueryDistQueue.pop();
        }

        for (size_t j = 0; j < ngt; j++){
            IDOutput.write((char *) (QueryGt.data() + j), sizeof(uint32_t));
            DistOutput.write((char *) (QueryDist.data() + j), sizeof(float));
        }
        assert(QueryDistQueue.empty());
        if (i % (nq / 10) == 0){
            TRecorder.print_time_usage("Computed " + std::to_string(i) + " th query ");
        }
    }
    IDOutput.close();

    /*
    std::ifstream TrueGTInput(PathGt, std::ios::binary);
    std::ifstream ComputedGTInput(PathTrainGt, std::ios::binary);
    
    std::vector<uint32_t> TrueGT(nq * ngt);
    std::vector<uint32_t> ComputedGT(nq * ngt);

    readXvec<uint32_t>(TrueGTInput, TrueGT.data(), ngt, nq, true, true);
    readXvec<uint32_t>(ComputedGTInput, ComputedGT.data(), ngt, nq, true, true);

    for (size_t i = 0; i < nq; i++){
        for (size_t j = 0; j < ngt; j++){
            if (TrueGT[i * ngt + j] != ComputedGT[i * ngt + j]){
                std::cout << " Query " << i << " th " << j << " th GT is " << TrueGT[i * ngt + j] << " " << ComputedGT[i * ngt + j] << "\n";
            }
        }
    }
    */

    return 1;
}