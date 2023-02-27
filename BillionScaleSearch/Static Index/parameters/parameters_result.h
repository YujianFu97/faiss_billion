#include "./parameters_model_million.h"


std::string nc_combination(const uint32_t * ncentroids, const std::string * index_type, 
const size_t layers, const size_t * M_PQ_layer, const size_t * nbits_PQ_layer){
    std::string result = "";
    size_t n_pq = 0;
    for (size_t i = 0; i < layers; i++){
        result += "_";
        if (index_type[i] == "PQ"){
            result +=  std::to_string(M_PQ_layer[n_pq]) + "_" + std::to_string(nbits_PQ_layer[n_pq]);
            n_pq ++;
        }
        else{
            result += std::to_string(ncentroids[i]);
        }
    }
    return result;
}

std::string layer_combination(const std::string * index_type, const size_t layers, const size_t * LQ_type){
    std::string result = "";
    size_t n_lq = 0;
    size_t n_vq = 0;
    for (size_t i = 0; i < layers; i++){
        result += "_"; result += index_type[i]; 
        if (index_type[i] == "VQ" && use_HNSW_VQ[n_vq]) {result += "_HNSW"; n_vq++;}
        if (index_type[i] == "LQ")                {result += std::to_string(LQ_type[n_lq]); n_lq ++;}
    }
    return result;
}


// Folder path
std::string ncentroid_conf = nc_combination(ncentroids, index_type, layers, M_PQ_layer, nbits_PQ_layer);
std::string nCentroid_with_M = ncentroid_conf + "_" + std::to_string(M_PQ);
std::string model = "models" + layer_combination(index_type, layers, LQ_type);

const std::string path_quantizers = path_folder + model + "/" + dataset + "/quantizer" + ncentroid_conf + ".qt";
const std::string path_alphas = path_folder + model + "/" + dataset + "/base_alpha" + ncentroid_conf + ".alpha";
const std::string path_alphas_raw = path_folder + model + "/" + dataset + "/base_alpha_raw_" + ncentroid_conf + ".alpha_raw";
const std::string path_ids =        path_folder + model + "/" + dataset + "/base_idxs" + ncentroid_conf + ".ivecs";
const std::string path_group_HNSW = path_folder + model + "/" + dataset + "/group_HNSWs" + ncentroid_conf + ".gHNSW";
const std::string path_info =       path_folder + model + "/" + dataset + "/" + ncentroid_conf + ".info";
const std::string path_edges =     path_folder + model + "/" + dataset + "/" + ncentroid_conf + ".edges";
const std::string path_centroids = path_folder + model + "/" + dataset + "/" + ncentroid_conf + ".centroids";

const std::string path_speed_record = path_folder + model + "/" + dataset + "/recording" + nCentroid_with_M + "_qps.txt";
const std::string path_record =     path_folder + model + "/" + dataset + "/recording" + nCentroid_with_M + ".txt";

//const std::string path_base_alpha_norm = path_folder + model + "/" + dataset + "/base_alpha_norm" + ncentroid_conf + ".norm";
const std::string path_centroid_norm = path_folder + model + "/" + dataset + "/centroid_norm" + nCentroid_with_M + ".norm";


std::string OPQ_label = use_OPQ? "OPQ_" : "";

const std::string path_OPQ =        path_folder + model + "/" + dataset + "/opq_matrix_" + nCentroid_with_M + ".opq";
const std::string path_pq =  path_folder + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + nCentroid_with_M  + "_" + std::to_string(nbits) + OPQ_label +".pq";
const std::string path_pq_norm =    path_folder + model + "/" + dataset + "/PQ_NORM" + std::to_string(M_PQ) + nCentroid_with_M  + OPQ_label + ".norm";
const std::string path_base_norm =      path_folder + model + "/" + dataset + "/base_norm" + std::to_string(M_PQ) + nCentroid_with_M + "_" + std::to_string(nbits) + OPQ_label + ".norm";
const std::string path_index =      path_folder + model + "/" + dataset + "/PQ" + std::to_string(M_PQ) + nCentroid_with_M + "_" + std::to_string(nbits) + OPQ_label + ".index";

/**
 **This is the centroids for assigining origin train vectors  size: n_group * dimension
 **/
const std::string path_groups = path_folder + model + "/" + dataset + "/selector_centroids_" + std::to_string(selector_group_size) + ".fvecs";
//This is for recording the labels based on the generated centroids

/**
 ** This is the labels for all assigned vectors, n_group * group_size 
 **/
const std::string path_labels = path_folder + model + "/" + dataset + "/selector_ids_" + std::to_string(selector_group_size);
