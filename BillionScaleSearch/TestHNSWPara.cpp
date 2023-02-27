#include "Index/BIndex.h"
#include "parameters/MillionScale/ParameterResults.h"


int main(){

    Retrain = false;
    std::string PathRecord = PathFolder + Dataset + "/RecordHNSWParaTest.txt";
    std::ofstream RecordFile;
    if (Recording){
    RecordFile.open(PathRecord, std::ios::app);
    time_t now = std::time(0); char * dt = ctime(&now);
    RecordFile << std::endl << std::endl << "Time: " << dt << std::endl;
    char hostname[100] = {0}; gethostname(hostname, sizeof(hostname));
    RecordFile << "Server Node: " << hostname << std::endl;
    //RecordFile << "nc: " << nc << " nt: " << nt << " TrainSize " << CTrainSize << " Nlevel: " << Nlevel << " Use Opt: " << UseOptimize << " Lambda: " << Lambda <<  " OptSize: " << OptSize << std::endl;
    }


    std::vector<float> BaseSet(nb * Dimension);
    std::vector<float> QuerySet(nq * Dimension);
    time_recorder Trecorder = time_recorder();

    
    std::ifstream BaseInput(PathBase, std::ios::binary);
    readXvecFvec<float>(BaseInput, BaseSet.data(), Dimension, nb, true, true);
    std::ifstream QueryInput(PathQuery, std::ios::binary);
    readXvecFvec<float>(QueryInput, QuerySet.data(), Dimension, nq, true, true);

    /*
    Trecorder.reset();
    for (size_t i = 0; i < nb; i++){
        for (size_t j = 0; j < nb; j++){
            faiss::fvec_L2sqr(BaseSet.data() + i * Dimension, BaseSet.data() + j * Dimension, Dimension);
        }
    }
    Trecorder.print_time_usage("Search base vectors in square");
    */

    for (size_t TestSize = 1000; TestSize <= 1000000; TestSize *= 2){
        Trecorder.reset();
        hnswlib::HierarchicalNSW * Graph = new hnswlib::HierarchicalNSW(Dimension, TestSize, M, 2 * M, EfConstruction);
        for (size_t i = 0; i < TestSize; i++){
            Graph->addPoint(BaseSet.data() + i * Dimension);
        }

        std::cout << "Build HNSW with " << TestSize << " vectors in " << Trecorder.getTimeConsumption() / 1000000 << " s\n";
        RecordFile << "Build HNSW with " << TestSize << " vectors in " << Trecorder.getTimeConsumption() / 1000000 << " s\n";
        uint32_t ef = 64;
        Trecorder.reset();
        for (size_t i = 0; i < nq; i++){
            auto Result = Graph->searchBaseLayer(QuerySet.data() + i * Dimension, ef);
            for (size_t j = 0; j < ef; j++){
                Result.pop();
            }
        }
        std::cout << "Search " << nq << " vectors with EfSearch = " << ef << " in " << Trecorder.getTimeConsumption() / (nq * 1000) << " ms\n";
        RecordFile << "Search " << nq << " vectors with EfSearch = " << ef << " in " << Trecorder.getTimeConsumption() / (nq * 1000) << " ms\n";
    }

    
    faiss::ProductQuantizer * PQ;
    if (exists(PathPQ)){
        PQ = faiss::read_ProductQuantizer(PathPQ.c_str());
    }
    else{
        PQ->verbose = true;
        PQ->train(1000, BaseSet.data());
        faiss::write_ProductQuantizer(PQ, PathPQ.c_str());
    }
    std::cout << "PQ Train Completed" << std::endl;
    std::vector<float> PQTable(PQ->ksub * PQ->M);
    
    std::vector<float> QDist(RecallK[1]);
    std::vector<int64_t> QLabel(RecallK[1]);
    
   for (size_t pow = 1000; pow <= 10000; pow += 1000){
        Trecorder.reset();
        size_t TestSize = pow;
        for (size_t m = 0; m < nq; m++){
            faiss::maxheap_heapify(RecallK[1],QDist.data(), QLabel.data());
            //PQ->compute_inner_prod_table(QuerySet.data() + m * Dimension, PQTable.data());
            for (size_t i = 0; i < TestSize; i++){
                float VNorm = 0;
                float ProdDist = 0;
                for (size_t j = 0; j < PQ->code_size; j++){
                    ProdDist += PQTable[j];

                    float Dist = VNorm + 2 * ProdDist;
                    if (Dist < QDist[0]){
                        faiss::maxheap_pop(RecallK[1],  QDist.data(), QLabel.data());
                        faiss::maxheap_push(RecallK[1], QDist.data(), QLabel.data(), Dist, m);
                    }
                }
            }
        }
        if(Recording){RecordFile << "Search " << nq << " queries with " << TestSize << " base vectors with time consumption: " << Trecorder.getTimeConsumption() / (nq*1000) << " ms\n";}
        std::cout <<  "Search " << nq << " queries with " << TestSize << " base vectors with time consumption: " << Trecorder.getTimeConsumption() / (nq*1000) << " ms\n";
    }

   RecordFile.close();
    return 1;
}