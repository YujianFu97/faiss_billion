#include "fstream"
#include "iostream"
#include <faiss/utils/random.h>
#include <faiss/utils/Heap.h>
#include <queue>
#include <chrono>
#include <string.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>

namespace bslib{
    using idx_t = int64_t;
    struct time_recorder{
        std::chrono::steady_clock::time_point start_time;
        public:
            time_recorder(){
                start_time = std::chrono::steady_clock::now();
            }

            float getTimeConsumption(){
                std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
                return (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
            }

            void reset(){
                start_time = std::chrono::steady_clock::now();
            }

            void record_time_usage(std::ofstream & output_record, std::string s){
                output_record << s << " The time usage: " << getTimeConsumption() / 1000000 << " s " << std::endl;
            }

            void print_time_usage(std::string s){
                std::cout << s << " The time usage: " << getTimeConsumption() / 1000000 << " s "<< std::endl; 
            }

            float get_time_usage(){
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

        void print_recall_performance(size_t n_query, float recall, size_t recall_k, std::string mode, const size_t nprobe){
            std::cout << "The recall@ " << recall_k << " for " << n_query << " queries in " << mode << " mode is: " << recall <<  std::endl;
            std::cout << "The search parameters is: ";
            std::cout << "The nprobe is: " << nprobe << std::endl;
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
    uint32_t GetXvecSize(std::ifstream & in, const size_t dimension){
        in.seekg(0, std::ios::end);
        size_t FileSize = (size_t) in.tellg();
        std::cout << "The file size is " << FileSize / 1000 << " KB " << std::endl;
        size_t DataSize = (unsigned) (FileSize / (dimension * sizeof(T) + sizeof(uint32_t)));
        std::cout << "The data size is " << DataSize << std::endl;
        return DataSize;
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

    template<typename T>
    void writeXvec(std::ofstream & out, T * data, const size_t dimension, const size_t n = 1,
    bool ShowProcess = false){
        uint32_t dim = dimension;
        size_t print_every = 1000;
        for (size_t i = 0; i< n; i++){
            out.write((char *) & dim, sizeof(uint32_t));
            out.write((char *) (data + i * dim), dim * sizeof(T));
            if ( ShowProcess && print_every != 0 && i % print_every == 0)
                std::cout << "[Finished writing " << i << " / " << n << "]"  << std::endl; 
        } 
    }



    inline bool exists(const std::string FilePath){
        std::ifstream f (FilePath);
        return f.good();
    }



    inline void PrintMessage(std::string s){
        std::cout << s << std::endl;
    }

    inline void PrepareFolder(const char * FilePath){
        if(NULL==opendir(FilePath))
        mkdir(FilePath, S_IRWXU); //Have the right to read, write and execute
    }

    template<typename T>
    inline void HashMapping(size_t n, const T * group_ids, T * hash_ids, size_t hash_size){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            hash_ids[i] = group_ids[i] % hash_size;
    }


    inline std::string GetNowTime() {
        time_t setTime;
        time(&setTime);
        tm* ptm = localtime(&setTime);
        std::string time = std::to_string(ptm->tm_year + 1900)
                        + " "
                        + std::to_string(ptm->tm_mon + 1)
                        + " "
                        + std::to_string(ptm->tm_mday)
                        + " "
                        + std::to_string(ptm->tm_hour) + " "
                        + std::to_string(ptm->tm_min) + " "
                        + std::to_string(ptm->tm_sec);		
        return time;	
    }

    /**
     * 
     * This is the function for getting the product between query and base vector
     * 
     * Input:
     * code: the code of base vectors
     * 
     * Output:
     * the product value of query and quantized base vectors
     * 
     **/
    float pq_L2sqr(const uint8_t *code, const float * precomputed_table, size_t code_size, size_t ksub);


    /**
     * 
     * This is the function for keeping k results in m result value
     * 
     * Input:
     * m: the total number of result pairs
     * k: the number of result pairs that to be kept
     * all_dists: the origin results of dists         size: m 
     * all_labels: the origin label results           size: m
     * sub_dists: the kept result of dists            size: k
     * sub_labels: the kept result of labels          size: k   
     * 
     **/
    void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, float * sub_dists, idx_t * sub_labels);

    void keep_k_min_alpha(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, const float * all_alphas, 
    float * sub_dists, idx_t * sub_labels, float * sub_alphas);
}
