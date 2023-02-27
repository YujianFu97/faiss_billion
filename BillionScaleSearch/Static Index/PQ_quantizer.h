#include "quantizer.h"

namespace bslib{
    struct PQ_quantizer : Base_quantizer
    {
        size_t M;
        size_t max_nbits;
        size_t max_ksub;
        size_t dsub;

        std::vector<size_t> exact_nbits;
        std::vector<size_t> exact_ksubs;
        
        std::vector<faiss::ProductQuantizer *> PQs;
        std::vector<std::vector<float>> centroid_norms;

        explicit PQ_quantizer(size_t dimension, size_t nc_upper, size_t M, size_t max_nbits);

        idx_t new_pow(size_t k_sub, size_t pow);
        idx_t spaceID_2_label(const idx_t * spaceID, const idx_t group_id);
        idx_t label_2_spaceID(idx_t label, idx_t * spaceID);
        bool traversed(const idx_t * visited_index, const idx_t * index_check, const size_t index_num);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_set_idxs);
        void compute_final_centroid(idx_t label, float * final_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals);
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x);
        void search_in_group(const float * query, const idx_t group_id, float * result_dists, idx_t * result_labels, size_t keep_space);
        void search_all(const size_t n, const float * queries, idx_t * result_labels);
        void multi_sequence_sort(const idx_t group_id, const float * dist_sequence, size_t keep_space, float * result_dists, idx_t * result_labels);
        float get_centroid_norms(const idx_t group_id);
    };
}
