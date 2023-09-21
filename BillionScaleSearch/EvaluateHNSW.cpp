#include "utils/utils.h"
#include "hnswlib/hnswalg.h"
#include "parametersNL/million/result.h"
#include <iomanip>  // for std::setw, std::left, std::right

#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <stdexcept>
#include <omp.h>

#define EPS 0.0001

/**************************/
bool Reconstruct = false;
size_t EfCons = 100;
size_t HNSW_M = 32;
size_t K = 1;
std::string PathHNSWFolder = PathFolder + Dataset + "/" + "HNSWFiles/";
std::string PathHNSWIndex =  PathHNSWFolder + "HNSW-" + std::to_string(EfCons) + "-" + std::to_string(HNSW_M) + ".index";

std::string PathHNSWNonDataIndex = PathHNSWFolder + "HNSW-" + std::to_string(EfCons) + "-" + std::to_string(HNSW_M) + ".NonDataIndex";
std::vector<uint32_t> EfList = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200};

/***************************/

int main(){
    performance_recorder recorder = performance_recorder("EvaluateHNSW");

    PrepareFolder(PathHNSWFolder.c_str());
    if (!exists(PathHNSWNonDataIndex) || Reconstruct){
        std::cout << "Constructing the HNSW index \n";
        std::vector<float> BaseSubset(Assign_batch_size * Dimension);
        std::ifstream inFile = std::ifstream(PathBase, std::ios::binary);

        hnswlib::L2Space l2space(Dimension);
        hnswlib::HierarchicalNSW<float> * Graph = new hnswlib::HierarchicalNSW<float>(&l2space, nb, HNSW_M, EfCons);

        for (size_t i = 0; i < Assign_num_batch; i++){
            readXvecFvec<DType> (inFile, BaseSubset.data(), Dimension, Assign_batch_size, false, false);
            #pragma omp parallel for
            for (size_t j = 0; j < Assign_batch_size; j++){
                Graph->addPoint(BaseSubset.data() + j * Dimension, i * Assign_batch_size + j);
            }
            recorder.print_performance("Completed " + std::to_string(i + 1) + " th iteration total: " + std::to_string(Assign_num_batch));
        }
        Graph->saveIndexNoData(PathHNSWNonDataIndex);
        delete Graph;
        recorder.print_performance("Construction Completed");
    }

    hnswlib::L2Space l2space(Dimension);
    //hnswlib::HierarchicalNSW<float> * Graph =  new hnswlib::HierarchicalNSW<float>(l2space, PathHNSWIndex, false, nb, false);
    //Graph->saveIndexNoData(PathHNSWNonDataIndex);
    hnswlib::HierarchicalNSW<float> * NewGraph;

    if (exists(PathHNSWNonDataIndex)){
        std::vector<float> BaseSet(nb * Dimension);
        std::ifstream inFile = std::ifstream(PathBase, std::ios::binary);
        readXvecFvec<DType> (inFile, BaseSet.data(), Dimension, nb, true, true);
        NewGraph = new hnswlib::HierarchicalNSW<float>(&l2space, PathHNSWNonDataIndex, BaseSet, false, nb, false);
        BaseSet.clear();
        std::vector<float>().swap(BaseSet);
    }

    recorder.print_performance("Index Loaded");
    omp_set_num_threads(1);
    std::vector<float> QuerySet(nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream inFile(PathQuery, std::ios::binary);
    readXvecFvec<DType>(inFile, QuerySet.data(), Dimension, nq, true, true);
    std::ifstream GTInputStream(PathGt, std::ios::binary);
    if (!GTInputStream) {std::cerr << "Error opening PathGt file: " << PathGt << "\n";exit(1);}
    readXvec<uint32_t>(GTInputStream, GT.data(), ngt, nq, false, false);

    std::cout << "The result for recall = " << K << "\n";
    std::cout << std::left << std::setw(8) << "Ef" << "|"
            << std::setw(12) << "Avg Recall" << "|"
            << std::setw(28) << "Avg Search Time (ms/query)" << "|"
            << std::setw(42) << "Avg Num of Distance Computations/query"
            << std::endl;

    for (size_t i = 0; i < EfList.size(); i++) {
        std::vector<uint32_t> SearchResults(nq * K);
        std::vector<size_t> VisitedItems(nq);
        std::vector<float> queryTimes(nq);
        NewGraph->ef_ = EfList[i];

        size_t TotalCorrect = 0;

        for (size_t j = 0; j < nq; j++) {
            size_t NumDistComp = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            auto Result = NewGraph->searchKnn(QuerySet.data() + j * Dimension, K, NumDistComp);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

            queryTimes[j] = static_cast<float>(duration) / 1000;  // convert to milliseconds

            for (size_t k = 0; k < K; k++) {
                SearchResults[j * K + k] = Result.top().second;
                Result.pop();
            }

            VisitedItems[j] = NumDistComp;
        }

        float avgQueryTime = std::accumulate(queryTimes.begin(), queryTimes.end(), 0.0f) / nq;
        size_t totalDistComp = std::accumulate(VisitedItems.begin(), VisitedItems.end(), 0);

        for (size_t j = 0; j < nq; j++) {
            std::unordered_set<uint32_t> targetSet(GT.begin() + ngt * j, GT.begin() + ngt * j + K);

            for (size_t k = 0; k < K; k++) {
                TotalCorrect += targetSet.count(SearchResults[j * K + k]);
            }
        }

        float Recall = (float)TotalCorrect / (nq * K);
        float avgDistComp = static_cast<float>(totalDistComp) / nq;

        std::cout << std::left << std::setw(8) << EfList[i] << "|"
                << std::right << std::setw(11) << std::fixed << std::setprecision(4) << Recall << " |"
                << std::setw(27) << avgQueryTime << " |"
                << std::setw(41) << avgDistComp
                << std::endl;
    }
    //delete Graph;
    delete NewGraph;
}

