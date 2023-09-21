
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iomanip>  // for std::setw, std::left, std::right

#include "../faiss/Index.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/IndexIVFFlat.h"
#include "../faiss/index_io.h"
#include "parametersNL/million/result.h"
#include "utils/utils.h"

using idx_t = faiss::idx_t;
uint32_t recallK = 10;
size_t Nlist = 1000;
std::vector<uint32_t> ProbeList = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200};
std::string FaissFolder = PathFolder + Dataset + "/" + "FaissFiles/";;
std::string IndexName = FaissFolder + "Faiss-" + std::to_string(Nlist) + ".index";


int main() {
    PrepareFolder(FaissFolder.c_str());

    if (!exists(IndexName)){
        std::vector<float> Baseset(nb * Dimension);
        std::ifstream inFile(PathBase, std::ios::binary);
        readXvecFvec<DType>(inFile, Baseset.data(), Dimension, nb, true, true);
        faiss::IndexFlatL2 quantizer(Dimension); // the other index
        faiss::IndexIVFFlat index(&quantizer, Dimension, nc);
        assert(!index.is_trained);
        index.train(nb, Baseset.data());
        assert(index.is_trained);
        index.add(nb, Baseset.data());
        faiss::write_index(&index, IndexName.c_str());
    }

    std::vector<float> Queryset(nq * Dimension);
    std::vector<uint32_t> GT(nq * ngt);
    std::ifstream inFile(PathQuery, std::ios::binary);
    readXvecFvec<DType>(inFile, Queryset.data(), Dimension, nq, true, true);
    inFile = std::ifstream (PathGt, std::ios::binary);
    readXvec<uint32_t>(inFile, GT.data(), ngt, nq, true, true);
    // Read the index from disk
    faiss::Index* index = faiss::read_index(IndexName.c_str());
    faiss::IndexIVFFlat* ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index);

    std::vector<idx_t> ResultID(nq * recallK);
    std::vector<float> ResultDist(nq * recallK);

    std::cout << "The result for recall = " << recallK << "\n";
    std::cout << std::left << std::setw(8) << "Ef" << "|"
            << std::setw(12) << "Avg Recall" << "|"
            << std::setw(28) << "Avg Search Time (ms/query)" << "|"
            << std::setw(42) << "Avg Num of Distance Computations/query"
            << std::endl;

    omp_set_num_threads(1);
    for (auto probe : ProbeList){
        ivf_index->nprobe = 10;
        float queryTime = 0;
        size_t TotalCorrect = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        ivf_index->search(nq, Queryset.data(), recallK, ResultDist.data(), ResultID.data());
        auto end_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < nq; i++){
            std::unordered_set<uint32_t> targetSet(GT.begin() + ngt * i, GT.begin() + ngt * i + recallK);

            for (size_t j = 0; j < recallK; j++) {
                TotalCorrect += targetSet.count(ResultID[i * recallK + j]);
            }
        }
        float avgQueryTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / nq;
        float Recall = (float)TotalCorrect / (nq * recallK);

        std::cout << std::left << std::setw(8) << probe << "|"
                << std::right << std::setw(11) << std::fixed << std::setprecision(4) << Recall << " |"
                << std::setw(27) << avgQueryTime << " |"
                << std::endl;
    }

    delete index;
    return 0;
}
