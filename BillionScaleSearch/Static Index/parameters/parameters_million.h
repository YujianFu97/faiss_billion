#include <cstdio>
#include <iostream>
#include <string>

//Not supported:
const bool use_reranking = false;
const bool use_norm_quantization = false;
const bool use_train_selector = false;
const bool use_all_HNSW = false;

const bool use_OPQ = false;
const bool use_HNSW_group = false;
const bool is_recording = true;
const bool saving_index = true;

const std::string dataset = "SIFT1M";
//const std::string dataset = "GIST1M";
const std::string path_folder = "/home/y/yujianfu/ivf-hnsw/";

//For train PQ
const size_t M_PQ = 16;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t dimension = 128;

const size_t train_size = 100000; //This is the size of train set
const size_t OPQ_train_size = 1000000;

const size_t selector_train_size = 100000;
const size_t selector_group_size = 2000;

const size_t PQ_train_size = 10000;
const size_t nb = 1000000;
const size_t nbatches = 10; //100
const uint32_t batch_size =  nb / nbatches;

const size_t group_HNSW_thres = 100;
const size_t group_HNSW_M = 6;
const size_t group_HNSW_efConstruction = 32;

const size_t ngt = 100;
const size_t nq = 1000;
const size_t num_search_paras = 10;
const size_t num_recall = 1;

const size_t result_k[num_recall] = {10};
const size_t max_vectors[num_search_paras] = {nb, nb, nb, nb, nb, nb, nb, nb, nb, nb};
const size_t reranking_space[num_recall] = {150};
const std::string search_mode = "non parallel";


//File paths
const std::string path_learn =     path_folder + "data/" + dataset + "/" + dataset +"_learn.fvecs";
const std::string path_base =      path_folder + "data/" + dataset + "/" + dataset +"_base.fvecs";
const std::string path_gt =        path_folder + "data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
const std::string path_query =     path_folder + "data/" + dataset + "/" + dataset +"_query.fvecs";
