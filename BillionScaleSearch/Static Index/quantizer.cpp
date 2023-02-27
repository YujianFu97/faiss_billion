#include "quantizer.h"

namespace bslib{
    Base_quantizer::Base_quantizer(size_t dimension, size_t nc_upper, size_t max_nc_per_group):
    dimension(dimension), nc_upper(nc_upper), max_nc_per_group(max_nc_per_group){
        //Only for construction, no need to be stored on disk
        M_all_HNSW = 12;
        efConstruction_all_HNSW = 20;
        min_train_size_per_group = 10;

        //Default setting: may be changed if no enough training vectors
        layer_nc = nc_upper * max_nc_per_group;

        
        CentroidDistributionMap.resize(nc_upper);
        size_t num_centroids = 0;
        for (size_t i = 0; i < nc_upper; i++){
            this->CentroidDistributionMap[i] = num_centroids;
            num_centroids += max_nc_per_group;
        }
    }
}