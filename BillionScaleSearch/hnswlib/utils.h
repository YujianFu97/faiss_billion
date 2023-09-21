#ifndef _UTILS_H
#define _UTILS_H

#include "fstream"
#include "iostream"
#include <queue>
#include <chrono>
#include <string.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <random>

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
            TempDuration1 += (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void recordTimeConsumption2(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            TempDuration2 += (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void recordTimeConsumption3(){
            std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
            TempDuration3 += (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        }

        inline void reset(){
            start_time = std::chrono::steady_clock::now();
        }

        inline void record_time_usage(std::ofstream & output_record, std::string s){
            time_t now = std::time(0); char * dt = ctime(&now);
            output_record << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s, now time: " << dt << std::endl;
        }

        inline void print_time_usage(std::string s){
            time_t now = std::time(0); char * dt = ctime(&now);
            std::cout << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s, now time: " << dt << std::endl; 
        }

        inline void print_record_time_usage(std::ofstream & output_record, std::string s){
            time_t now = std::time(0); char * dt = ctime(&now);
            output_record << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s, now time: " << dt << std::endl;
            std::cout << s << " The time usage: |" << getTimeConsumption() / 1000000 << "| s, now time: " << dt << std::endl;
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
        output_record << s << " The memory usage: " << r_usage.ru_maxrss / 1000 <<  " MB " << std::endl;
    }

    void print_memory_usage(std::string s){
        rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        std::cout << s << " The memory usage: " <<  r_usage.ru_maxrss / 1000 << " MB " << std::endl;
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

inline std::vector<uint32_t> sampleList(size_t n, size_t k) {
    std::vector<uint32_t> result(n);
    for (uint32_t i = 0; i < n; ++i) {
        result[i] = i;
    }

    std::mt19937 gen(123);
    
    for (uint32_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<uint32_t> dis(i, n - 1);
        int j = dis(gen);
        std::swap(result[i], result[j]);
    }

    result.resize(k);
    return result;
}

inline bool exists(const std::string FilePath){
    std::ifstream f (FilePath);
    return f.good();
}

#endif