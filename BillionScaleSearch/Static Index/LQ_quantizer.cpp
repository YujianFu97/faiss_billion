#include "LQ_quantizer.h"
#define Not_Found -1.0

namespace bslib{

    /**
     * This is the initialize function for LQ layer: 
     * 
     * Input: 
     * nc_upper: number of groups = number of centroids in upper layer   
     * nc_per_group: number of subcentroids per group
     * upper_centroids: data of upper centroids:                   size: nc_upper * dimension
     * upper_centroid_ids: ids of nn of upper centroids            size: nc_upper * nc_per_group
     * upper_centroid_dists: dists of nn of upper centroids        size: nc_upper * nc_per_group
     *  
     **/
    LQ_quantizer::LQ_quantizer(size_t dimension, size_t nc_upper, size_t max_nc_per_group, const float * upper_centroids,
                               const idx_t * upper_centroid_ids, const float * upper_centroid_dists, size_t LQ_type, bool use_all_HNSW):
            Base_quantizer(dimension,  nc_upper, max_nc_per_group){

            this->use_all_HNSW = use_all_HNSW;
            this->alphas.resize(nc_upper);
            this->upper_centroids.resize(dimension * nc_upper);
            this->nn_centroid_ids.resize(nc_upper);
            this->nn_centroid_dists.resize(nc_upper);
            this->LQ_type = LQ_type;

            // Need to store all upper layer centroids
            for (size_t i = 0; i < dimension * nc_upper; i++){
                this->upper_centroids[i] = upper_centroids[i];
            }

            for (size_t i = 0; i < nc_upper; i++){
                for (size_t j = 0; j < max_nc_per_group; j++){
                    this->nn_centroid_ids[i].push_back(upper_centroid_ids[i * max_nc_per_group + j]);
                    this->nn_centroid_dists[i].push_back(upper_centroid_dists[i * max_nc_per_group + j]);
                }
                assert(this->nn_centroid_ids[i].size() == max_nc_per_group && this->nn_centroid_dists[i].size() == max_nc_per_group);
            }
        }

    std::pair<float, float> LQ_quantizer::LQ0_distance(const float * vector, const float * centroid, const float * nn_centroid, float nn_dist){
            std::vector<float> centroid_vector(dimension);
            std::vector<float> point_vector(dimension);
            faiss::fvec_madd(dimension, vector, -1.0, centroid, point_vector.data()); 
            faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vector.data());

            float numerator = faiss::fvec_inner_product(centroid_vector.data(), point_vector.data(), dimension);
            const float denominator = nn_dist; 
            const float alpha = numerator / denominator; 

            std::vector<float> subcentroid(dimension); 
            faiss::fvec_madd(dimension, centroid, alpha, centroid_vector.data(), subcentroid.data()); 
            const float dist = faiss::fvec_L2sqr(vector, subcentroid.data(), dimension); 

            //if (alpha >= 0.5){
                //std::cout << "LQ0: " << numerator << " " << nn_dist << " " << alpha << " " << dist << std::endl;
            //}

            return std::make_pair(alpha, dist);
    }


    std::pair<float, float> LQ_quantizer::LQ0_fast_distance(float nn_dist, float v_c_dist, float v_n_dist){
            float alpha = (v_c_dist + nn_dist - v_n_dist) / (2 * nn_dist);

            //float cosine_sqr = (v_c_dist + nn_dist - v_n_dist)*(v_c_dist + nn_dist - v_n_dist)/(4 * v_c_dist * nn_dist);
            // The first and second type of dist
            //float dist_1 = v_c_dist * (1 - cosine_sqr);
            //float dist_2 = alpha*(alpha-1)*nn_dist + (1-alpha)*v_c_dist + alpha*v_n_dist;
            float dist_3 = v_c_dist - alpha * alpha * nn_dist;

            //if (alpha >= 0.5){
                //std::cout << "LQ0 fast: " << nn_dist << " " << v_c_dist << " " << v_n_dist << " " << alpha << " " << dist_3  << std::endl;
            //}
            return std::make_pair(alpha, dist_3); 
    }

    /**
     * This is the function for train the dataset, to compute the alpha in every group.
     * 
     * Input: 
     * centroid_vectors:                the vector between each upper centroid and its nn centroids   size: nc_per_group * dimension
     * points:                          the training points for each group                            size: group_size * dimension
     * centroid:                        one upper centroid                                            size: dimension
     * centroid_vector_norms_L2sqr:     the norm of upper centroid vectors                            size: nc_per_group
     * group_size:                      number of train vectors                                       size_t
     * 
     * Output:
     * alpha                            the alpha for one group
     * 
     **/ 
    void LQ_quantizer::compute_alpha(const float * centroid_vectors, const float * points, const float * centroid,
                                      const float * centroid_vector_norms_L2sqr, size_t group_size, float * alpha_list){
        if (group_size <= 0){
            std::cout << "No enough data point (0 point) for computing alpha in this group " << std::endl;
            exit(-1);
        }
        
        float group_numerator = 0.0;   
        float group_denominator = 0.0; 
        std::vector<float> sum_group_alpha(max_nc_per_group, 0);
        std::vector<float> sum_group_vectors(max_nc_per_group, 0);

        std::vector<float> point_vectors(group_size * dimension); 
        for (size_t i = 0; i < group_size; i++) 
            faiss::fvec_madd(dimension, points + i * dimension, -1.0, centroid, point_vectors.data() + i * dimension); 
        
        for (size_t i = 0; i < group_size; i++){ 
            const float * point_vector = point_vectors.data() + i * dimension; 
            const float * point = points + i * dimension; 

            std::priority_queue<std::pair<float, std::pair<size_t, std::pair<float, float>>>> maxheap; 

            for (size_t subc = 0; subc < max_nc_per_group; subc++){
                const float * centroid_vector = centroid_vectors + subc * dimension; 
                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, dimension); 
                numerator = (numerator > 0) ? numerator : 0.0; 

                const float denominator = centroid_vector_norms_L2sqr[subc]; 
                const float alpha = numerator / denominator; 

                std::vector<float> subcentroid(dimension); 
                faiss::fvec_madd(dimension, centroid, alpha, centroid_vector, subcentroid.data()); 

                const float dist = faiss::fvec_L2sqr(point, subcentroid.data(), dimension); 
                maxheap.emplace(-dist, std::make_pair(subc, std::make_pair(numerator, denominator))); 
            }

            sum_group_alpha[maxheap.top().second.first] +=  (maxheap.top().second.second.second > 0) ? maxheap.top().second.second.first / maxheap.top().second.second.second : 0.0;
            sum_group_vectors[maxheap.top().second.first] ++;
            group_numerator += maxheap.top().second.second.first;
            group_denominator += maxheap.top().second.second.second;
        }
        if (LQ_type == 0){
            float LQ0_alpha = (group_denominator > 0) ? group_numerator / group_denominator : 0.0;
            for (size_t i = 0; i < max_nc_per_group; i++){
                alpha_list[i] = LQ0_alpha;
            }
        }
        else{
            float sum_alpha = 0;
            for (size_t i = 0; i < max_nc_per_group; i++){sum_alpha += sum_group_alpha[i];} float avg_alpha = sum_alpha / group_size;
            for (size_t i = 0; i < max_nc_per_group; i++){
                if (sum_group_vectors[i] > 0){alpha_list[i] = sum_group_alpha[i] / sum_group_vectors[i];}
                else{alpha_list[i] = avg_alpha;}
            }
        }
    }


    /** 
     * This is the function for train the centroids and compute the alpha for every group
     * 
     * Input:
     * train_data: this is the dataset for train the alpha in every group.                      size: train_set_size * dimension
     * train_data_ids: this is the id set for dataset, which group should be put in.            size: train_set_size
     * update_ids: whether update the train set ids or not                                      bool
     * 
     * Output:
     * train_data_ids: updated train group id                          
     * 
    **/ 
    void LQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_ids){
        if (LQ_type == 2){
            std::cout << " LQ type 2, no need to compute the alpha" << std::endl;
            
            this->alphas.resize(0);

            std::cout << "Computing the centroid norm and centroid product" << std::endl;
            upper_centroid_product.resize(nc_upper * max_nc_per_group);
            for (size_t i = 0; i < nc_upper; i++){
                const float * base_centroid = upper_centroids.data() + i * dimension;
                for (size_t j = 0; j < max_nc_per_group; j++){
                    idx_t neighbor_id = nn_centroid_ids[i][j];
                    const float * neighbor_centroid = upper_centroids.data() + neighbor_id * dimension;
                    std::vector<float> centroid_vector(dimension);
                    faiss::fvec_madd(dimension, neighbor_centroid, -1.0, base_centroid, centroid_vector.data());
                    upper_centroid_product[i * max_nc_per_group + j] = faiss::fvec_inner_product(centroid_vector.data(), base_centroid, dimension);
                }
            }
            return;
        }
        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " groups " << std::endl;
        std::vector<std::vector<float>> train_set(this->nc_upper);

        for(size_t i = 0; i < train_set_size; i++){
            idx_t group_id = train_data_ids[i];
            assert(group_id <= this->nc_upper);
            for (size_t j = 0; j < this->dimension; j++){
                train_set[group_id].push_back(train_data[i * dimension + j]);
            }
        }

        size_t min_train_size = train_set[0].size() / dimension; 
        for (size_t i = 0; i < nc_upper; i++){if (min_train_size > train_set[i].size() / dimension) min_train_size = train_set[i].size() / dimension; if (i <= 1000) {std::cout << train_set[i].size() / dimension <<" ";}}
        std::cout <<  std::endl << "The min size for sub train set is: " << min_train_size << std::endl;

        std::cout << "Computing alphas for lq_quantizer with upper centroids: " << this->upper_centroids.size() / dimension << " nc_per_group: " << max_nc_per_group << std::endl;
        std::cout << "The size of upper_centroids: " << this->upper_centroids.size() / this->dimension << std::endl;
        std::cout << "The size of nn_centroid_ids: " << this->nn_centroid_ids.size() << std::endl;
        alphas.resize(nc_upper);

#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            alphas[i].resize(max_nc_per_group);
            std::vector<float> centroid_vectors(max_nc_per_group * dimension);
            const float * centroid = this->upper_centroids.data() + i * dimension;
            for (size_t j = 0; j < max_nc_per_group; j++){
                const float * nn_centroid = this->upper_centroids.data() + this->nn_centroid_ids[i][j] * dimension;
                faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vectors.data()+ j * dimension);
            }
            size_t train_group_size = train_set[i].size() / this->dimension;
            compute_alpha(centroid_vectors.data(), train_set[i].data(), centroid, this->nn_centroid_dists[i].data(), train_group_size, alphas[i].data());
        }
        std::cout << "finished computing centoids" <<std::endl;

    }

    /**
     * 
     * This is the function for updating the ids for train set data
     * 
     * Input:
     * train_data: the new train data for next layer     size: n * dimension
     * n: the size for query set
     * 
     * Output:
     * train_data_ids: the ids for train vectors        size: n * k
     * 
     **/

    void LQ_quantizer::search_all(const size_t n, const size_t k, const float * query_data, idx_t * query_data_ids){
        if (LQ_type == 2){
            for (size_t i = 0; i < n; i++){
                std::vector<float> search_dist(layer_nc);
                std::vector<idx_t> search_idxs(layer_nc);
                for (size_t group_id = 0; group_id < nc_upper; group_id++){
                    float * upper_centroid = this->upper_centroids.data() + group_id * dimension;
                    for (size_t inner_group_id = 0; inner_group_id < max_nc_per_group; inner_group_id++){
                        
                        idx_t nn_centroid_id = this->nn_centroid_ids[group_id][inner_group_id];
                        float * nn_centroid = this->upper_centroids.data() + nn_centroid_id * dimension;
                        float nn_dist = this->nn_centroid_dists[group_id][inner_group_id];
                        auto result_pair = LQ0_distance(query_data + i * dimension, upper_centroid, nn_centroid, nn_dist);
                        search_dist[group_id * max_nc_per_group + inner_group_id] = result_pair.second;
                        search_idxs[group_id * max_nc_per_group + inner_group_id] = group_id * max_nc_per_group + inner_group_id;
                    }
                }
                std::vector<float> result_dist(k);
                keep_k_min(layer_nc, k, search_dist.data(), search_idxs.data(), result_dist.data(), query_data_ids + i * k);
            }
        }

        else if (use_all_HNSW){
            for (size_t i = 0; i < n; i++){
                auto result_queue = HNSW_all_quantizer->searchKnn(query_data + i * dimension, k);
                size_t result_size = result_queue.size();
                assert(result_size == k);
                for (size_t j = 0; j < result_size; j++){
                    query_data_ids[i * k + k - j - 1] = result_queue.top().second;
                    result_queue.pop();
                }
            }
        }        
        else{
            faiss::IndexFlatL2 centroid_index(dimension * k);
            std::vector<float> one_centroid(dimension * k);

            for (idx_t label = 0; label < layer_nc; label ++){
                compute_final_centroid(label, one_centroid.data());
                centroid_index.add(1, one_centroid.data());
            }

            std::vector<float> result_dists(n);
            centroid_index.search(n, query_data, k, result_dists.data(), query_data_ids);
        }
    }

    void LQ_quantizer::get_group_id(idx_t label, idx_t & group_id, idx_t & inner_group_id){
            group_id = label / max_nc_per_group;
            inner_group_id = label - CentroidDistributionMap[group_id];
    }

    /**
     * The size for final centroid: dimension
     * The label is the idx for sub centroid in every layer: nc_upper * nc_per_group
     * 
     * Input: 
     * group_id:        the group of the target centroid   
     * inner_group_id:  the inner id of centroid in group
     * 
     * Output:
     * final_centroid:  the target centroid          size: dimension
     * 
     **/
    void LQ_quantizer::compute_final_centroid(idx_t label, float * final_centroid){
        if (LQ_type == 2){
            std::cout << " There is no centroid in LQ type2 " << std::endl;
            exit(0);
        }
        idx_t group_id, inner_group_id;
        get_group_id(label, group_id, inner_group_id);
        
        std::vector<float> centroid_vector(dimension);
        const float * nn_centroid = this->upper_centroids.data() + nn_centroid_ids[group_id][inner_group_id] * dimension;
        const float * centroid = this->upper_centroids.data() + group_id * dimension;
        faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vector.data());
        faiss::fvec_madd(dimension, centroid, alphas[group_id][inner_group_id], centroid_vector.data(), final_centroid);
    }


    /**
     * This is for computing residual between data point and the centroid
     * 
     * Input:
     * n:             number of data points.               size: umlimited
     * labels:        id of the centroid, [0, nc].         size: n
     * x:             data point.                          size: n * dimension
     * 
     * Output:
     * residuals:     residual of data point and centroid  size: n * dimension
     * 
     **/
    void LQ_quantizer::compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals, const float * vector_alpha){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            if (LQ_type == 2){
                idx_t group_id, inner_group_id;
                get_group_id(labels[i], group_id, inner_group_id);
                float * group_centroid = this->upper_centroids.data() + group_id * dimension;
                idx_t nn_id = this->nn_centroid_ids[group_id][inner_group_id];
                float * nn_centroid = this->upper_centroids.data() + nn_id * dimension;
                std::vector<float> centroid_vector(dimension);
                faiss::fvec_madd(dimension, nn_centroid, -1.0, group_centroid, centroid_vector.data());
                //for (size_t temp = 0; temp <dimension; temp ++){std::cout << group_centroid[temp] << " ";}std::cout << std::endl<< std::endl;
                //for (size_t temp = 0; temp <dimension; temp ++){std::cout << nn_centroid[temp] << " ";}std::cout << std::endl<< std::endl;
                //for (size_t temp = 0; temp <dimension; temp ++){std::cout << centroid_vector[temp] << " ";}std::cout << std::endl<< std::endl;
                std::vector<float> subcentroid(dimension); 
                //std::cout << "Vector alpha: " << vector_alpha[i] << std::endl;
                faiss::fvec_madd(dimension, group_centroid, vector_alpha[i], centroid_vector.data(), subcentroid.data()); 
                //for (size_t temp = 0; temp <dimension; temp ++){std::cout << subcentroid[temp] << " ";}std::cout << std::endl<< std::endl;
                faiss::fvec_madd(dimension, x + i * dimension, -1.0, subcentroid.data(), residuals + i * dimension);
                //for (size_t temp = 0; temp <dimension; temp ++){std::cout << residuals[i * dimension + temp] << " ";}std::cout << std::endl<< std::endl;
            }
            else{
                std::vector<float> final_centroid(dimension);
                compute_final_centroid(labels[i], final_centroid.data());
                faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
            }
        }
    }


    /**
     * This is for recover data point with residual and the centroid
     * 
     * Input:
     * n:             number of residual points.           size: umlimited
     * labels:        id of the centroid, [0, nc].         size: n
     * residuals:     precomputed residual of data point.  size: n * dimension
     * 
     * Output:
     * x:             reconstructed data point             size: n * dimension
     * 
     **/
    void LQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x, const float * vector_alpha){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            if (LQ_type == 2){
                idx_t group_id, inner_group_id;
                get_group_id(labels[i], group_id, inner_group_id);
                float * group_centroid = this->upper_centroids.data() + group_id * dimension;
                idx_t nn_id = this->nn_centroid_ids[group_id][inner_group_id];
                float * nn_centroid = this->upper_centroids.data() + nn_id * dimension;
                std::vector<float> centroid_vector(dimension);
                faiss::fvec_madd(dimension, nn_centroid, -1.0, group_centroid, centroid_vector.data());
                std::vector<float> subcentroid(dimension); 
                faiss::fvec_madd(dimension, group_centroid, vector_alpha[i], centroid_vector.data(), subcentroid.data()); 
                faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, subcentroid.data(), x + i * dimension);
            }
            else{
                std::vector<float> final_centroid(dimension);
                compute_final_centroid(labels[i], final_centroid.data());
                faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
            }
        }
    }


    /**
     * 
     * Change the search space for different queries
     * For a query sequence queries, search in their group and return the k cloest centroids
     * 
     * Search all distance in each group to compute all sub centroids
     * 
     * Input:
     * queries:              the query data points                                              size: n * dimension
     * upper_result_labels:  the upper labels for query and upper centroids                     size: n * upper_search_space
     * upper_result_dists:   the upper distance for query and upper centroids                   size: n * upper_search_space
     * upper_search_space:   the search space for each vector in upper layer
     * group_ids:            the group ids for relative query: which group should we search in  size: n
     * 
     * Output:
     * result_labels:        the search result labels                                           size: n * nc_per_group
     * result_dists:         the search result dists                                            size: n * nc_per_group
     * 
     **/
    void LQ_quantizer::search_in_group(const float * query, idx_t * upper_result_labels, const float * upper_result_dists, 
                const size_t upper_search_space, const idx_t group_id, float * result_dists, idx_t * result_labels, float * vector_alpha){        

        /*
        std::cout << "Search result: ";
        for (size_t i = 0; i < upper_search_space; i++){
            std::cout << upper_result_labels[i] << " " << upper_result_dists[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Neighbor result: ";
        for (size_t i = 0; i < max_nc_per_group; i++){
            std::cout << nn_centroid_ids[group_id][i] << " " << nn_centroid_dists[group_id][i] << " ";
        }
        std::cout << std::endl;
        */


        for (size_t inner_group_id = 0; inner_group_id < max_nc_per_group; inner_group_id++){

            result_labels[inner_group_id] = CentroidDistributionMap[group_id] + inner_group_id;
            idx_t nn_id = nn_centroid_ids[group_id][inner_group_id];
            float query_nn_dist = Not_Found;
            float query_group_dist = Not_Found;

            /*
            bool query_nn_flag = false;
            bool query_group_flag = false;
            for(size_t index = 0; index < upper_search_space; index ++){
                if  (!query_nn_flag && upper_result_labels[index] == nn_id){
                    query_nn_dist = upper_result_dists[index];
                    query_nn_flag = true;
                }

                if (!query_group_flag && upper_result_labels[index] == group_id){
                    query_group_dist = upper_result_dists[index];
                    query_group_flag = true;
                }

                if (query_nn_flag && query_group_flag){
                    break;
                }
            }
            */
            // There are many ways for finding whether the nn_id exists in the upper search result, try different ways

            idx_t * nn_id_index = std::find(upper_result_labels, upper_result_labels+ upper_search_space, nn_id);
            idx_t nn_index = nn_id_index - upper_result_labels;

            if (nn_index < upper_search_space){
                idx_t * group_id_index = std::find(upper_result_labels, upper_result_labels+ upper_search_space, group_id);
                idx_t group_index = group_id_index - upper_result_labels;
                assert(group_index < upper_search_space);
                
                //assert(query_nn_dist != Not_Found && query_group_dist != Not_Found);
                query_group_dist = upper_result_dists[group_index];
                query_nn_dist =upper_result_dists[nn_index];

                float group_nn_dist = nn_centroid_dists[group_id][inner_group_id];

                if (LQ_type == 2){
                    auto result_pair = LQ0_fast_distance(group_nn_dist, query_group_dist, query_nn_dist);
                    result_dists[inner_group_id] = result_pair.second;
                    vector_alpha[inner_group_id] = result_pair.first;
                }
                else{
                    float alpha = alphas[group_id][inner_group_id];
                    result_dists[inner_group_id] = alpha*(alpha-1)*group_nn_dist + (1-alpha)*query_group_dist + alpha*query_nn_dist;  

                    //if (!(result_dists[inner_group_id] >=0)){
                    //    std::cout << result_dists[inner_group_id] << " " << alpha << " " << group_nn_dist << " " << query_group_dist << " " << query_nn_dist << std::endl;
                    //}
                }
            }
            else{
                std::cout << nn_id << " Not Found  " << std::endl;;
                std::vector<float> query_sub_centroid_vector(dimension);
                float * nn_centroid = upper_centroids.data() + nn_centroid_ids[group_id][inner_group_id] * dimension;
                float * group_centroid = upper_centroids.data() + group_id * dimension;
                float nn_dist = nn_centroid_dists[group_id][inner_group_id];

                if (LQ_type == 2){
                    auto result_pair = LQ0_distance(query, group_centroid, nn_centroid, nn_dist);

                    result_dists[inner_group_id] = result_pair.second;
                    assert(result_pair.second >=0);
                    vector_alpha[inner_group_id] = result_pair.first;
                }
                
                //std::vector<float> sub_centroid(dimension);
                //compute_final_centroid(group_id, j, sub_centroid.data());
                //faiss::fvec_madd(dimension, sub_centroid.data(), -1.0, query, query_sub_centroid_vector.data());
                else{
                    float alpha = alphas[group_id][inner_group_id];
                    for (size_t k = 0; k < dimension; k++){
                        query_sub_centroid_vector[k] = alpha * nn_centroid[k] + (1-alpha)*group_centroid[k]-query[k];
                    }
                    result_dists[inner_group_id] = faiss::fvec_norm_L2sqr(query_sub_centroid_vector.data(), dimension);
                    assert(result_dists[inner_group_id]);
                }
            }
        }


        /*
        for (size_t i = 0; i < n; i++){
            idx_t group_id = group_ids[i];
            query_sequence_set[group_id].push_back(i);
        }

//#pragma omp parallel for
        for (size_t group_id = 0; group_id < this->nc_upper; group_id++){
            if (query_sequence_set[group_id].size() == 0)
                continue;
            else{
                std::vector<std::vector<float>> sub_centroids(this->nc_per_group, std::vector<float>(dimension));
                std::vector<bool> sub_centroids_computed(this->nc_per_group, false);
                float alpha = this->alphas[group_id];

                for (size_t train_group_id = 0; train_group_id < query_sequence_set[group_id].size(); train_group_id++){
                    idx_t sequence_id = query_sequence_set[group_id][train_group_id];
                    std::vector<float> query_sub_centroids_dists(this->nc_per_group, 0);

                    for (size_t inner_group_id = 0; inner_group_id < this->nc_per_group; inner_group_id++){
                        float query_nn_dist = Not_Found;
                        float query_group_dist = Not_Found;
                        idx_t nn_id = this->nn_centroid_ids[group_id][inner_group_id];

                        for (size_t search_id = 0; search_id < upper_search_space; search_id++){
                            if (upper_result_labels[sequence_id * upper_search_space  + search_id] == nn_id){
                                query_nn_dist = upper_result_dists[sequence_id * upper_search_space + search_id];
                            }

                            else if (upper_result_labels[sequence_id * upper_search_space + search_id] == group_id){
                                query_group_dist = upper_result_dists[sequence_id * upper_search_space + search_id];
                            }

                            if (query_nn_dist != Not_Found && query_group_dist != Not_Found)
                                break;
                        }
                        assert (query_group_dist != Not_Found);

                        if (query_nn_dist != Not_Found){
                            float group_nn_dist = this->nn_centroid_dists[group_id][inner_group_id];

                            query_sub_centroids_dists[inner_group_id] = alpha*(alpha-1)*group_nn_dist + (1-alpha)*query_group_dist + alpha*query_nn_dist;
                        }
                        else{
                        if (! sub_centroids_computed[inner_group_id]){
                            compute_final_centroid(group_id, inner_group_id, sub_centroids[inner_group_id].data());
                            sub_centroids_computed[inner_group_id] = true;
                        }
                        
                        const float * query = queries + sequence_id * dimension;
                        std::vector<float> query_sub_centroid_vector(dimension);
                        faiss::fvec_madd(dimension, sub_centroids[inner_group_id].data(), -1.0, query, query_sub_centroid_vector.data());
                        query_sub_centroids_dists[inner_group_id] = faiss::fvec_norm_L2sqr(query_sub_centroid_vector.data(), dimension);
                        }
                    }

                    for (size_t inner_group_id = 0; inner_group_id < this->nc_per_group; inner_group_id++){
                        result_dists[sequence_id * this->nc_per_group + inner_group_id] = query_sub_centroids_dists[inner_group_id];
                        result_labels[sequence_id * this->nc_per_group + inner_group_id] = CentroidDistributionMap[group_id] + inner_group_id;
                    }
                }
            }
        }
        */
    }

    /**
     * This is for building LQ layer, to compute the nearest neighbor for every centroid
     * The neighbor is global: should consider neighbors in different group
     * 
     * Input:
     * k: is the number of NN neighbors to be computed    
     * 
     * Output:
     * nn_centroids: data of the upper layer (this layer) centroids       size: nc * dimension
     * nn_dists: distance of the upper layer nearest neighbor centroids   size: nc * k
     * nn_labels: idx of the upper layer nearest neighbor centroids       size: nc * k
     * 
     **/

    void LQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * nn_ids){
        
        if (LQ_type == 2){
            std::cout << " There is no centroid in LQ type2 " << std::endl;
            exit(0);
        }
        std::vector<float> target_centroid(dimension);
        if (use_all_HNSW){
            for (idx_t label = 0; label < layer_nc;label++){
                compute_final_centroid(label, target_centroid.data());
                auto result_queue = HNSW_all_quantizer->searchKnn(target_centroid.data(), k+1);
                assert(result_queue.size() == k+1);
                // The centroid with smallest size should be itself
                // We just start from the top (As the search result is from large distance to small distance)
                // And get the first k results (from k+1 results)
                for (size_t temp = 0; temp < k; temp ++){
                    hnswlib::idx_t search_centroid_id = label;
                    assert(search_centroid_id != result_queue.top().second);
                    nn_ids[search_centroid_id * k + k - temp -1] = result_queue.top().second;
                    nn_dists[search_centroid_id * k + k - temp - 1] = result_queue.top().first;  
                    result_queue.pop();
                }
            }
        }
        else{
            faiss::IndexFlatL2 all_quantizer(dimension);

            for (idx_t label = 0; label < layer_nc; label++){
                compute_final_centroid(label, target_centroid.data());
                all_quantizer.add(1, target_centroid.data());
            }

            std::cout << "searching the idx for centroids' nearest neighbors " << std::endl;
            for (size_t i = 0; i < this->layer_nc * this->dimension; i ++){
                nn_centroids[i] = all_quantizer.xb[i];
            }

            std::vector<idx_t> search_nn_ids(this->layer_nc * (k+1));
            std::vector<float> search_nn_dis(this->layer_nc * (k+1));
            all_quantizer.search(this->layer_nc, all_quantizer.xb.data(), k+1, search_nn_dis.data(), search_nn_ids.data());
            
            for (size_t i = 0; i < this->layer_nc; i++){
                for (size_t j = 0; j < k; j++){
                    assert(i != search_nn_ids[i * (k + 1) + j + 1 ]);
                    nn_dists[i * k + j] = search_nn_dis[i * (k + 1) + j + 1];
                    nn_ids[i * k + j] = search_nn_ids[i * (k + 1) + j + 1 ];
                }
            }
        }
    }
}