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

const std::string dataset = "SIFT1B";
const std::string path_folder = "/home/y/yujianfu/ivf-hnsw/";

//For train PQ
const size_t M_PQ = 16; //Or 16
const size_t M_norm_PQ = 1;
const size_t nbits = 8;
const size_t dimension = 128;

const size_t train_size = 100000000; //This is the size of train set
const size_t OPQ_train_size = 100000;

const size_t selector_train_size = 10000000;
const size_t selector_group_size = 2000;


const size_t PQ_train_size = 1000000;
const size_t nb = 1000000000;
const size_t nbatches = 100; //100
const uint32_t batch_size =  nb / nbatches;


const size_t group_HNSW_thres = 5000;
const size_t group_HNSW_M = 64;
const size_t group_HNSW_efConstruction = 500;

const size_t ngt = 1000;
const size_t nq = 10000;
const size_t num_search_paras = 10;
const size_t num_recall = 2;

const size_t result_k[num_recall] = {1, 10};
const size_t max_vectors[num_search_paras] = {100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 5500000};
const size_t reranking_space[num_recall] = {10, 150};
const std::string search_mode = "non parallel";

//File paths
const std::string path_learn =     path_folder + "data/" + dataset + "/" + "bigann_learn.bvecs";
const std::string path_base =      path_folder + "data/" + dataset + "/" + "bigann_base.bvecs";
const std::string path_gt =        path_folder + "data/" + dataset + "/" + "gnd/idx_1000M.ivecs";
const std::string path_query =     path_folder + "data/" + dataset + "/" + "bigann_query.bvecs";

