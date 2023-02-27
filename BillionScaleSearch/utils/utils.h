#ifndef _UTILS_H
#define _UTILS_H

#include "fstream"
#include "iostream"
#include "../../faiss/utils/random.h"
#include "../../faiss/utils/Heap.h"
#include <queue>
#include <chrono>
#include <string.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>

struct time_recorder{
    std::chrono::steady_clock::time_point start_time;
    float TempDuration1;
    float TempDuration2;
    float TempDuration3;
    public:

        time_recorder(){
            start_time = std::chrono::steady_clock::now();
            TempDuration1 = 0;
            TempDuration2 = 0;
            TempDuration3 = 0;
        }

        inline float getTimeConsumption(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            return (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void recordTimeConsumption1(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            TempDuration1 = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void recordTimeConsumption2(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            TempDuration2 = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void recordTimeConsumption3(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            TempDuration3 = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void reset(){
            start_time = std::chrono::steady_clock::now();
        }

        inline void record_time_usage(std::ofstream & output_record, std::string s){
            output_record << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s " << std::endl;
        }

        inline void print_time_usage(std::string s){
            std::cout << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s "<< std::endl; 
        }

        inline void print_record_time_usage(std::ofstream & output_record, std::string s){
            output_record << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s " << std::endl;
            std::cout << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s "<< std::endl; 
        }

        inline float get_time_usage(){
            return getTimeConsumption() / 1000000;
        }
};

struct memory_recorder{
    public:
        
    void record_memory_usage(std::ofstream & output_record, std::string s){
        rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        output_record << s << " The memory usage: " <<  r_usage.ru_ixrss << " KB / " << r_usage.ru_isrss << " KB / " << r_usage.ru_idrss << " KB / " << r_usage.ru_maxrss <<  " KB " << std::endl;
    }

    void print_memory_usage(std::string s){
        rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        std::cout << s << " The memory usage: " <<  r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;
    }
};

struct recall_recorder{
    public:

    void print_recall_performance(size_t n_query, float recall, size_t recall_k, std::string mode, const size_t nprobe, const size_t MaxItem){
        std::cout << "The recall@ " << recall_k << " for " << n_query << " queries in " << mode << " mode is: " << recall <<  std::endl;
        std::cout << "The search parameters is: ";
        std::cout << "The nprobe and MaxItem: " << nprobe << " " << MaxItem << std::endl;
    }

    void record_recall_performance(std::ofstream & output_record, size_t n_query, float recall, size_t recall_k, std::string mode, const size_t nprobe){
        output_record << "The recall@" << recall_k << " for " << n_query << " queries in " << mode << " mode is: " << recall <<  std::endl;
        output_record << "The search parameters is ";
        output_record << "The nprobe is: " << nprobe << std::endl;
    }
};

template<typename T>
void CheckResult(T * data, const size_t dimension, size_t dataset_size = 2){
    std::cout << "Printing sample (2 vectors) of the dataset " << std::endl;

    for (size_t i = 0; i < dimension; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl << std::endl;
    for (size_t i = 0; i< dimension; i++){
        std::cout << data[(dataset_size-1) * dimension+i] << " ";
    }
    std::cout << std::endl << std::endl;
}

template<typename T>
void readXvec(std::ifstream & in, T * data, const size_t dimension, const size_t n,
                bool CheckFlag = false, bool ShowProcess = false){
    if (ShowProcess)
    std::cout << "Loading data with " << n << " vectors in " << dimension << std::endl;
    uint32_t dim = dimension;
    size_t print_every = n / 10;
    for (size_t i = 0; i < n; i++){
        in.read((char *) & dim, sizeof(uint32_t));
        if (dim != dimension){
            std::cout << dim << " " << dimension << " dimension error \n";
            exit(1);
        }
        in.read((char *) (data + i * dim), dim * sizeof(T));
        if ( ShowProcess && print_every != 0 && i % print_every == 0)
            std::cout << "[Finished loading " << i << " / " << n << "]"  << std::endl; 
    }
    if (CheckFlag)
        CheckResult<T>(data, dimension, n);
}

template<typename T>
void readXvecFvec(std::ifstream & in, float * data, const size_t dimension, const size_t n = 1,
                    bool CheckFlag = false, bool ShowProcess = false){
    if (ShowProcess)
    std::cout << "Loading data with " << n << " vectors in " << dimension << std::endl;
    uint32_t dim = dimension;
    std::vector<T> origin_data(dimension);
    size_t print_every = n / 10;
    for (size_t i = 0; i < n; i++){
        in.read((char * ) & dim, sizeof(uint32_t));
        if (dim != dimension) {
            std::cout << dim << " " << dimension << " dimension error \n";
            exit(1);
        }
        in.read((char * ) origin_data.data(), dim * sizeof(T));
        for (size_t j = 0; j < dimension; j++){
            data[i * dim + j] = 1.0 * origin_data[j];
        }
        if ( ShowProcess && print_every != 0 && i % (print_every) == 0)
            std::cout << "[Finished loading " << i << " / " << n << "]" << std::endl; 
    }
    if (CheckFlag)
        CheckResult<float>(data, dimension, n);
}


inline void PrepareFolder(const char * FilePath){
    if(NULL==opendir(FilePath))
    mkdir(FilePath, S_IRWXU); //Have the right to read, write and execute
}

template<typename T>
void RandomSubset(const T * x, T * output, size_t dimension, size_t n, size_t sub_n){
    long RandomSeed = 1234;
    std::vector<int> RandomId(n);
    faiss::rand_perm(RandomId.data(), n, RandomSeed+1);
    /*
    for (size_t i = 0; i < sub_n; i++){
        std::cout << RandomId[i] << " ";
    }
    std::cout << "\n";
    */

    for (size_t i = 0; i < sub_n; i++){
        memcpy(output + i * dimension, x + RandomId[i] * dimension, sizeof(T) * dimension);
    }
}

inline std::vector<std::string> StringSplit(const std::string &str, const std::string &pattern)
{
    char * strc = new char[strlen(str.c_str())+1];
    strcpy(strc, str.c_str());   //string to C-string
    std::vector<std::string> res;
    char* temp = std::strtok(strc, pattern.c_str());
    while(temp != NULL)
    {
        res.push_back(std::string(temp));
        temp = std::strtok(NULL, pattern.c_str());
    }
    delete[] strc;
    return res;
}

inline bool exists(const std::string FilePath){
    std::ifstream f (FilePath);
    return f.good();
}


// Add inline and there is no error
inline std::pair<float, std::pair< float, float>> LeastSquares(std::vector<float> NV, std::vector<float> ST){
    if (NV.size() ==0){
        std::cout << " Least Square error: No elements to be trained \n";
        return std::make_pair(0, std::make_pair(0, 0));
    }
    assert(NV.size() == ST.size());
    float Lxy = 0; float Lxx = 0; float Lyy = 0;
    float NV_avg = std::accumulate(NV.begin(), NV.end(), 0.0) / NV.size();
    float ST_avg = std::accumulate(ST.begin(), ST.end(), 0.0) / NV.size();

    for (size_t i = 0; i < NV.size(); i++){
        Lxx += (NV[i] - NV_avg) * (NV[i] - NV_avg);
        Lxy += (NV[i] - NV_avg) * (ST[i] - ST_avg);
        Lyy += (ST[i] - ST_avg) * (ST[i] - ST_avg);
    }
    if (Lxy * NV.size() != Lxx){
        float k = Lxy / Lxx;
        float b = ST_avg - k * NV_avg;
        float coef = Lxy / std::sqrt(Lxx * Lyy);
        std::cout << "The correlation coefficient of " << NV.size() << " elements: " << coef << " with k: " << k << " and b: " << b << "\n";
        return std::make_pair(coef, std::make_pair(k, b));
    }
    else{
        std::cout << " Least square error: Unlimited slope\n";
        return std::make_pair(0, std::make_pair(0, 0));
    }
}

#endif