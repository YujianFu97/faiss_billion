#include <string>
#include <unistd.h>
#include "VQ_quantizer.h"
#include "LQ_quantizer.h"
#include "PQ_quantizer.h"
#include <unordered_set>
#include <algorithm>
#include "faiss/VectorTransform.h"

#define EPS 0.00001


namespace bslib{

// Change this type for different datasets
//typedef uint8_t learn_data_type;
//typedef uint8_t base_data_type;

typedef float learn_data_type;
typedef float base_data_type;

typedef faiss::Index::idx_t idx_t;
typedef std::pair<size_t, size_t> HNSW_PQ_para;

struct Bslib_Index{
    size_t dimension; // Initialized in constructer
    size_t layers = 0; // Initialized in constructer
    std::vector<std::string> index_type; // Initialized in constructer
    size_t train_size; //Initialized in constructer by 0, assigned in main

    size_t group_HNSW_thres;

    bool * use_VQ_HNSW;
    bool use_OPQ;
    bool use_saving_index;
    bool use_recording;
    bool use_vector_alpha;
    bool VQ_search_type;

    memory_recorder Mrecorder = memory_recorder();
    recall_recorder Rrecorder = recall_recorder();
    time_recorder Trecorder = time_recorder();

    hnswlib::HierarchicalNSW * base_HNSW;

    size_t M_pq; // Initialized by training pq
    size_t nbits; // Initialized by training pq
    size_t code_size; // Initialized by reading PQ
    size_t k_sub;
    size_t final_group_num; // Initialized by compute final nc
    size_t final_graph_num;
    size_t M_HNSW;
    size_t efConstruction;

    
    std::vector<size_t> ncentroids;
    std::vector<VQ_quantizer > vq_quantizer_index; // Initialized in read_quantizer
    std::vector<LQ_quantizer > lq_quantizer_index; // Initialized in read_quantizer
    std::vector<PQ_quantizer > pq_quantizer_index;
    std::vector<float> precomputed_table;

    faiss::ProductQuantizer pq; // Initialized in train_pq
    faiss::LinearTransform opq_matrix;

    std::vector<float> base_norms;
    std::vector<float> centroid_norms;

    std::vector<std::vector<uint8_t>> base_codes;
    std::vector<std::vector<idx_t>> base_sequence_ids;

    std::vector<std::vector<float>> base_alphas;

    std::vector<float> train_data; // Initialized in build_quantizers (without reading)
    std::vector<idx_t> train_data_ids; // Initialized in build_quantizers (without reading)

    std::vector<float> query_centroid_dists;

    explicit Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, 
    const bool saving_index, const bool is_recording,
    bool * use_VQ_HNSW, const bool use_OPQ,
    const size_t train_size, const size_t M_pq, const size_t nbits, const size_t M_HNSW, const size_t efConstruction);

    void do_OPQ(size_t n, float * dataset);
    void reverse_OPQ(size_t n, float * dataset);
    void build_quantizers(const uint32_t * ncentroids, const std::string path_quantizer, const std::string path_learn, const std::string path_edges,
    const std::string path_info, const std::string path_centroids, const size_t * num_train, const std::vector<HNSW_PQ_para> HNSW_paras, const std::vector<HNSW_PQ_para> PQ_paras, const size_t * LQ_type, std::ofstream & record_file);
    
    void add_vq_quantizer(size_t nc_upper, size_t nc_per_group, bool use_VQ_HNSW_layer = false, size_t M = 4, size_t efConstruction = 10);
    void add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, 
    const float * upper_nn_centroid_dists, size_t LQ_type);
    void add_pq_quantizer(size_t nc_upper, size_t M, size_t nbits);

    void read_train_set(const std::string path_learn, size_t total_size, size_t train_set_size);

    void train_pq(const std::string path_pq, const std::string path_learn, const std::string path_OPQ, const size_t train_set_size);

    void encode(size_t n, const float * data, const idx_t * encoded_ids, float * encoded_data, const float * alphas = nullptr);
    void decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data, const float * alphas);
    void assign(const size_t n, const float * assign_data, idx_t * assigned_ids, size_t assign_layer, float * alphas = NULL);
    
    void add_batch(size_t n, const float * data, const idx_t * sequence_ids, const idx_t * group_ids, 
    const size_t * group_positions, const bool base_norm_flag, const bool alpha_flag, const float * vector_alphas);

    void get_final_num();
    void compute_centroid_norm(std::string path_centroid_norm);
    void search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, const size_t nprobe, std::string path_base);
    size_t get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, std::pair<idx_t, float> & result_idx_dist);
    float pq_L2sqr_search(const uint8_t *code);
    //float pq_L2sqr(const uint8_t *code, const float * precomputed_table);

    void read_quantizers(const std::string path_quantizers);
    void write_quantizers(const std::string path_quantizers);
    void read_index(const std::string path_index);
    void write_index(const std::string path_index);

    void get_final_centroid(const size_t group_id, float * final_centroid);

    void build_index(std::string path_learn,
    std::string path_quantizers, std::string path_edges, std::string path_info, std::string path_centroids,
    size_t VQ_layers, size_t PQ_layers, size_t LQ_layers, 
    const uint32_t * ncentroids, const size_t * M_HNSW, const size_t * efConstruction, 
    const size_t * M_PQ_layer, const size_t * nbits_PQ_layer, const size_t * num_train,
    const size_t * LQ_type, std::ofstream & record_file);

    void assign_vectors(std::string path_ids, std::string path_base, std::string path_alphas, uint32_t batch_size, size_t nbatches, std::ofstream & record_file);

    void train_pq_quantizer(const std::string path_pq,
        const size_t M_PQ, const std::string path_learn, const std::string path_OPQ, const size_t PQ_train_size, std::ofstream & record_file);

    void load_index(std::string path_index, std::string path_ids, std::string path_base,
        std::string path_base_norm, std::string path_centroid_norm, std::string path_alphas_raw,
        std::string path_alphas,
        size_t batch_size, size_t nbatches, size_t nb, std::ofstream & record_file);
    
    void index_statistic(std::string path_base, std::string path_ids,size_t nb, size_t nbatch);

    void query_test(size_t num_search_paras, size_t num_recall, size_t nq, size_t ngt, const size_t * nprobes,
        const size_t * result_k,
        std::ofstream & record_file, std::ofstream & qps_record_file, 
        std::string search_mode, std::string path_gt, std::string path_query, std::string path_base);

    void read_group_HNSW(const std::string path_group_HNSW);

    /**
     ** This is the function for dynamically get the reranking space for 
     ** For future work
     **/
    size_t get_reranking_space(size_t k, size_t group_label, size_t group_id);
    
    void read_base_alphas(std::string path_base_alpha);
    void write_base_alphas(std::string path_base_alpha);
    void read_base_alpha_norms(std::string path_base_alpha_norm);
    void write_base_alpha_norms(std::string path_base_alpha_norm);

};
}