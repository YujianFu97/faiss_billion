#include "./parameters_million.h"

/* Parameter setting: */
//Exp parameters
//For index initialization
const size_t VQ_layers = 1;
const size_t PQ_layers = 0;
const size_t LQ_layers = 0;
const size_t layers = VQ_layers + PQ_layers + LQ_layers;
const size_t LQ_type[LQ_layers] = {};

const std::string index_type[layers] = {"VQ"};
const uint32_t ncentroids[layers-PQ_layers] = {4500};


//For building index
bool use_HNSW_VQ[VQ_layers] = {false};
const size_t VQ_M_HNSW[VQ_layers] = {4};
//Set efConstruction and efSearch as the same
const size_t VQ_efConstruction [VQ_layers] = {10};

const size_t M_HNSW_all[VQ_layers] = {};
const size_t efConstruction_all [VQ_layers] = {};

const size_t M_PQ_layer[PQ_layers] = {};
const size_t nbits_PQ_layer[PQ_layers] = {};
const size_t num_train[layers] = {100000};
const size_t nprobes[num_search_paras] = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
const size_t M_HNSW = 12;
const size_t efConstruction = 36;

//Ensure that efConstruction > nc_LQ