#include "utils.h"

namespace bslib{

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
    float pq_L2sqr(const uint8_t *code, const float * precomputed_table, size_t code_size, size_t ksub)
    {
        float result = 0.;
        const size_t dim = code_size >> 2;
        size_t m = 0;
        for (size_t i = 0; i < dim; i++) {
            result += precomputed_table[ksub * m + code[m]]; m++;
            result += precomputed_table[ksub * m + code[m]]; m++;
            result += precomputed_table[ksub * m + code[m]]; m++;
            result += precomputed_table[ksub * m + code[m]]; m++;
        }
        return result;
    }


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
    
    void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, float * sub_dists, idx_t * sub_labels){
        if (k < m){
            faiss::maxheap_heapify(k, sub_dists, sub_labels);
            for (size_t i = 0; i < m; i++){
                if (all_dists[i] < sub_dists[0]){
                    faiss::maxheap_pop(k, sub_dists, sub_labels);
                    faiss::maxheap_push(k, sub_dists, sub_labels, all_dists[i], all_labels[i]);
                }
            }
        }
        else if (k == m){
            memcpy(sub_dists, all_dists, k * sizeof(float));
            memcpy(sub_labels, all_labels, k * sizeof(idx_t));
        }
        else{
            std::cout << "k (" << k << ") should be smaller than m (" << m << ") in keep_k_min function " << std::endl;
            exit(0);
        }
    }
    

   
    void keep_k_min_alpha(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, const float * all_alphas, 
    float * sub_dists, idx_t * sub_labels, float * sub_alphas){
        if (k < m){
            std::priority_queue<std::pair<float, std::pair<idx_t, float>>> maxheap;
            for (size_t i = 0; i < m; i++){
                maxheap.emplace(-all_dists[i], std::make_pair(all_labels[i], all_alphas[i]));
            }
            for (size_t i = 0; i < k; i++){
                sub_dists[i] = -maxheap.top().first;
                sub_labels[i] = maxheap.top().second.first;
                sub_alphas[i] = maxheap.top().second.second;
                maxheap.pop();
            }
        }
        else if (k == m){
            memcpy(sub_dists, all_dists, k * sizeof(float));
            memcpy(sub_labels, all_labels, k * sizeof(idx_t));
            memcpy(sub_alphas, all_alphas, k * sizeof(float));
        }
        else{
            std::cout << "k (" << k << ") should be smaller than m (" << m << ") in keep_k_min_alpha function " << std::endl;
            exit(0);
        }
    }
    
}