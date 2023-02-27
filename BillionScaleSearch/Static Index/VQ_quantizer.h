#include "quantizer.h"

namespace bslib{
    struct VQ_quantizer : Base_quantizer
    {

        size_t M;
        size_t efConstruction;

        bool use_HNSW;

        std::vector<hnswlib::HierarchicalNSW *> HNSW_quantizers;
        std::vector<faiss::IndexFlatL2 *> L2_quantizers; //Resized with nc_upper in read quantizer
        std::vector<size_t> exact_nc_in_groups; // If there is no enough training vectors, we shrink the number of clusters

         // The size should be train_set_size, the max size is nc_upper
        explicit VQ_quantizer(size_t dimension, size_t nc_upper, size_t max_nc_per_group, size_t M = 4, size_t efConstruction = 10, bool use_HNSW = false);
        void build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_ids);
        void compute_final_centroid(idx_t label, float * final_centroid);
        void compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals); 
        void recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x);
        void search_in_group(const float * query, const idx_t group_id, float * result_dists, idx_t * result_labels, size_t k);
        void get_group_id(const idx_t label, idx_t & group_id, idx_t & inner_group_id);
        void compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels);
        void write_HNSW(std::ofstream & output);
        void read_HNSW(std::ifstream & input);
    };
}

