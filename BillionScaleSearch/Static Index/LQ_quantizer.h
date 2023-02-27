#include "quantizer.h"

namespace bslib{
    struct LQ_quantizer : Base_quantizer
    {
        bool use_all_HNSW;
        size_t LQ_type;

        hnswlib::HierarchicalNSW * HNSW_all_quantizer;
        std::vector<std::vector<float>> alphas; // Initialized in read_quantizer
        std::vector<float> upper_centroids; // Initialized in constructor
        std::vector<std::vector<idx_t>> nn_centroid_ids; // Initialized in constructor
        std::vector<std::vector<float>> nn_centroid_dists; // Initialized in constructor
        std::vector<float> upper_centroid_product;

        explicit LQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group, const float * upper_centroids,
                              const idx_t * upper_centroid_ids, const float * upper_centroid_dists,  size_t LQ_type,bool use_all_HNSW = false);
        
        std::pair<float, float> LQ0_distance(const float * vector, const float * centroid, const float * nn_centroid, float nn_dist);
        std::pair<float, float> LQ0_fast_distance(float nn_dist, float v_c_dist, float v_n_dist);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_set_idxs);
        void compute_alpha(const float * centroid_vectors, const float * points, const float * centroid,
                                      const float * centroid_vector_norms_L2sqr, size_t group_size, float * alpha_list);
        void compute_final_centroid(idx_t label, float * sub_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals, const float * vector_alpha);
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x, const float * vector_alpha);
        void get_group_id(idx_t label, idx_t & group_id, idx_t & inner_group_id);
        void search_in_group(const float * query, idx_t * upper_result_labels, const float * upper_result_dists, const size_t upper_search_space, 
        const idx_t group_id, float * result_dists, idx_t * result_labels, float * vector_alpha = NULL);
        void search_all(const size_t n, const size_t k, const float * train_data, idx_t * train_data_ids);
        void compute_nn_centroids(size_t k, float * nn_centroids, float * nn_centroid_dists, idx_t * labels);
    };
}
