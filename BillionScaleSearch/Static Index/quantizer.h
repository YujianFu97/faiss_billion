#ifndef _QUANTIZER_H
#define _QUANTIZER_H
#include<iostream>
#include<fstream>
#include <cstdio>
#include <queue>
#include <math.h>
#include <map>
#include <algorithm>
#include <numeric>
#include <set>

#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
//#include <faiss/IndexHNSW.h>
#include "HNSWlib/hnswalg.h"
#include "utils/utils.h"

typedef faiss::Index::idx_t idx_t;

#define MAX_DIST 1e9
#define INVALID_ID -1

namespace bslib{
struct Base_quantizer
{


    size_t dimension;
    size_t nc_upper;
    size_t max_nc_per_group;
    size_t layer_nc;

    size_t min_train_size_per_group; // Only for VQ and PQ layer
    size_t M_all_HNSW;
    size_t efConstruction_all_HNSW;
    
    std::vector<idx_t> CentroidDistributionMap;
    
    explicit Base_quantizer(size_t dimension, size_t nc_upper, size_t max_nc_per_group);
};
}
#endif