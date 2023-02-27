#include "bslib_index.h"
#include "time.h"

namespace bslib{

    /**
     * The initialize function for BSLIB struct 
     * 
     * 
     * 
     **/
    Bslib_Index::Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, 
    const bool saving_index, const bool is_recording,
    bool * use_VQ_HNSW, const bool use_OPQ,
    const size_t train_size, const size_t M_pq, const size_t nbits, const size_t M_HNSW, const size_t efConstruction){

            this->dimension = dimension;
            this->layers = layers;

            this->use_VQ_HNSW = use_VQ_HNSW;
            this->use_vector_alpha = false;

            this->use_OPQ = use_OPQ;

            this->use_recording = is_recording;
            this->use_saving_index = saving_index;

            this->index_type.resize(layers);
            this->ncentroids.resize(layers);

            for (size_t i = 0; i < layers; i++){
                this->index_type[i] = index_type[i];
            }

            this->train_size = train_size;
            this->M_pq = M_pq;
            this->nbits = nbits;
            this->group_HNSW_thres = group_HNSW_thres;
            this->M_HNSW = M_HNSW;
            this->efConstruction = efConstruction;
        }
    
    /**
     * The function for adding a VQ layer in the whole structure
     * 
     * Parameters required for building a VQ layer: 
     * If use L2 quantizer: nc_upper, nc_group 
     * Use HNSW quantizer: nc_upper, nc_group, M, efConstruction
     * 
     **/
    void Bslib_Index::add_vq_quantizer(size_t nc_upper, size_t nc_per_group, bool use_HNSW, size_t M, size_t efConstruction){
        VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group, M, efConstruction, use_HNSW);
        PrintMessage("Building centroids for vq quantizer");
        vq_quantizer.build_centroids(this->train_data.data(), this->train_data.size() / dimension, this->train_data_ids.data());
        PrintMessage("Finished construct the VQ layer");
        
        this->vq_quantizer_index.push_back(vq_quantizer);
    }

    /**
     * The function for adding a LQ layer in the whole structure
     * 
     * Parameters required for building a LQ layer: 
     * nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists
     * 
     * upper_centroids: the upper centroids data   size: nc_upper * dimension
     * upper_nn_centroid_idxs: the upper centroid neighbor idxs      size: nc_upper * nc_per_group 
     * upper_nn_centroid_dists: the upper centroid neighbor dists    size: nc_upper * nc_per_group
     * 
     **/
    void Bslib_Index::add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists, size_t LQ_type){
        LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists, LQ_type);
        PrintMessage("Building centroids for lq quantizer");
        lq_quantizer.build_centroids(this->train_data.data(), this->train_data.size() / dimension, this->train_data_ids.data());
        PrintMessage("Finished construct the LQ layer");
        this->lq_quantizer_index.push_back(lq_quantizer);
    }

    /**
     * The function for adding a PQ layer in the whole structure
     * 
     * Parameter required for building a PQ layer
     * nc_upper, M, nbits
     * 
     **/
    void Bslib_Index::add_pq_quantizer(size_t nc_upper, size_t M, size_t nbits){
        PQ_quantizer pq_quantizer = PQ_quantizer(dimension, nc_upper, M, nbits);
        PrintMessage("Building centroids for pq quantizer");
        pq_quantizer.build_centroids(this->train_data.data(), this->train_data_ids.size(), this->train_data_ids.data());
        this->pq_quantizer_index.push_back(pq_quantizer);
    }

    /**
     * This is the function for encoding the origin base vectors into residuals to the centroids
     * 
     * Input:
     * encode_data: the vector to be encoded.    size: n * dimension
     * encoded_ids: the group id of the vectors  size: n
     * 
     * Output:
     * encoded_data: the encoded data            size: n *  dimension
     * 
     **/
    void Bslib_Index::encode(size_t n, const float * encode_data, const idx_t * encoded_ids, float * encoded_data, const float * alphas){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data, alphas);
        }
        else if(index_type[layers - 1] == "PQ"){
            pq_quantizer_index[pq_quantizer_index.size() -1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }

    /**
     * This is the function for decode the vector data
     * 
     * Input:
     * encoded_data: the encoded data                  size: n * dimension
     * encoded_ids:  the group id of the encoded data  size: n
     * 
     * Output:
     * decoded_data: the reconstruction data            size: n * dimension
     * 
     **/
    void Bslib_Index::decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data, const float * alphas){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else if (index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data, alphas);
        }
        else if (index_type[layers-1] == "PQ"){
            pq_quantizer_index[pq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }

    /**
     * This is the function for doing OPQ for the training and searching part.
     * Input:
     * n: this is the number of vectors to be processed
     * dataset: the vector set to be processed, the result data will be stored in the same place
     * 
     **/
    void Bslib_Index::do_OPQ(size_t n, float * dataset){
        assert(& opq_matrix != NULL);
        std::vector<float> copy_dataset(n * dimension);
        memcpy(copy_dataset.data(), dataset, n * dimension * sizeof(float));
        opq_matrix.apply_noalloc(n, copy_dataset.data(), dataset);
    }

    void Bslib_Index::reverse_OPQ(size_t n, float * dataset){
        std::vector<float> copy_dataset(n * dimension);
        memcpy(copy_dataset.data(), dataset, n * dimension * sizeof(float));
        opq_matrix.transform_transpose(n, copy_dataset.data(), dataset);
    }
    /**
     * 
     * This is the function for selecting subset of the origin set
     * 
     * Input:
     * path_learn: this is the path for learn set
     * total_size: the size of learn set size 
     * train_set_size: the size to be read 
     * 
     **/
    void Bslib_Index::read_train_set(const std::string path_learn, size_t total_size, size_t train_set_size){
        std::cout << "Reading " << train_set_size << " from " << total_size << " for training" << std::endl;
        this->train_data.resize(train_set_size * dimension, 0);
        //Train data ids is for locating the data vectors are used for which group training
        this->train_data_ids.resize(train_set_size, 0);

        if (total_size == train_set_size){
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<learn_data_type>(learn_input, this->train_data.data(), dimension, train_set_size, true, true);
        }
        else{
            std::ifstream learn_input(path_learn, std::ios::binary);
            std::vector<float> sum_train_data (total_size * dimension, 0);
            readXvecFvec<learn_data_type>(learn_input, sum_train_data.data(), dimension, total_size);
            std::cout << "Reading subset with " << train_set_size << " without selector from " << train_size << " vectors" << std::endl;
            RandomSubset<float>(sum_train_data.data(), this->train_data.data(), dimension, total_size, train_set_size);
        }
    }

    /**
     * The function for building quantizers in the whole structure
     * 
     * Input: 
     * 
     * ncentroids: number of centroids in all layers  size: layers
     * Note that the last ncentroids para for PQ layer is not used, it should be placed in PQ_paras 
     * path_quantizer: path for saving or loading quantizer
     * path_learn: path for learning dataset
     * n_train: the number of train vectors to be use in all layers     size: layers
     * 
     * HNSW_paras: the parameters for constructing HNSW graph
     * PQ_paras: the parameters for constructing PQ layer
     * 
     * 
     **/
    void Bslib_Index::build_quantizers(const uint32_t * ncentroids, const std::string path_quantizer, const std::string path_learn, const std::string path_edges,
    const std::string path_info, const std::string path_centroids, const size_t * num_train, const std::vector<HNSW_PQ_para> HNSW_paras, const std::vector<HNSW_PQ_para> PQ_paras, const size_t * LQ_type, std::ofstream & record_file){
        if (exists(path_quantizer)){
            read_quantizers(path_quantizer);

            base_HNSW = new hnswlib::HierarchicalNSW(path_info, path_centroids, path_edges);
            base_HNSW->efSearch = efConstruction;

            std::cout << "Checking the quantizers read from file " << std::endl;
            std::cout << "The number of quantizers: " << this->vq_quantizer_index.size() << " " << this->lq_quantizer_index.size() << " " << this->pq_quantizer_index.size() << std::endl;
        }
        else
        {
        PrintMessage("No preconstructed quantizers, constructing quantizers");
        //Load the train set into the index
        assert(index_type.size() == layers && index_type[0] != "LQ");
        std::cout << "adding layers to the index structure " << std::endl;
        uint32_t nc_upper = 1; 
        uint32_t nc_per_group;
        std::vector<float> upper_centroids;
        std::vector<idx_t> nn_centroids_idxs;
        std::vector<float> nn_centroids_dists;

        //Prepare train set for initialization
        read_train_set(path_learn, this->train_size, num_train[0]);
        time_recorder inner_Trecorder = time_recorder();
        
        size_t n_vq = 0;
        size_t n_lq = 0;
        size_t n_pq = 0;
        
        for (size_t i = 0; i < layers; i++){
            inner_Trecorder.reset();
            assert(n_vq + n_lq + n_pq == i);
            bool update_ids = (i == layers-1) ? false:true;
            if (i < layers-1 && index_type[i+1] == "LQ" && LQ_type[n_lq] == 2){update_ids = false;}
            nc_per_group = index_type[i] == "PQ" ? 0 : ncentroids[i];
            this->ncentroids[i] = nc_per_group;

            if (index_type[i] == "VQ"){

                std::cout << "Adding VQ quantizer with parameters: " << nc_upper << " " << nc_per_group << std::endl;

                size_t existed_VQ_layers = this->vq_quantizer_index.size();
                HNSW_PQ_para para = HNSW_paras[existed_VQ_layers];
                add_vq_quantizer(nc_upper, nc_per_group, use_VQ_HNSW[existed_VQ_layers], para.first, para.second);


                inner_Trecorder.record_time_usage(record_file, "Trained VQ layer");
                nc_upper = vq_quantizer_index[vq_quantizer_index.size()-1].layer_nc;

                //Prepare train set for the next layer
                if (update_ids){
                    read_train_set(path_learn, this->train_size, num_train[i+1]);
                    std::cout << "Updating train set for the next layer" << std::endl;
                    assign(train_data_ids.size(), train_data.data(), train_data_ids.data(), i+1);
                    inner_Trecorder.record_time_usage(record_file, "Updated train set from VQ layer for next layer with assign function");
                }

                if (!update_ids || index_type[i+1] != "VQ"){
                    // If there is no following layers or the next layer is not VQ, construct the basic HNSW for VQ centroids
                    size_t VQ_pos = vq_quantizer_index.size() - 1;
                    std::ofstream centroids_output(path_centroids, std::ios::binary);
                    this->base_HNSW = new hnswlib::HierarchicalNSW(dimension, vq_quantizer_index[VQ_pos].layer_nc, M_HNSW, 2 * M_HNSW, efConstruction);
                    for (size_t j = 0; j < vq_quantizer_index[VQ_pos].layer_nc; j++){
                        std::vector<float> VQ_centroid(dimension);
                        vq_quantizer_index[VQ_pos].compute_final_centroid(j, VQ_centroid.data());
                        base_HNSW->addPoint(VQ_centroid.data());
                        writeXvec<float>(centroids_output, VQ_centroid.data(), dimension, 1);
                    }
                    base_HNSW->SaveEdges(path_edges);
                    base_HNSW->SaveInfo(path_info);
                    centroids_output.close();
                }

                std::cout << "Trainset Sample" << std::endl;
                for (size_t temp = 0; temp <2; temp++)
                {for (size_t temp1 = 0; temp1 < dimension; temp1++)
                    {std::cout << this->train_data[temp * dimension + temp1] << " ";}
                        std::cout << train_data_ids[temp];std::cout << std::endl;}

                std::cout << i << "th VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of L2 quantizers (groups): " << vq_quantizer_index[vq_quantizer_index.size()-1].L2_quantizers.size() << 
                " the num of HNSW quantizers (groups): " <<  vq_quantizer_index[vq_quantizer_index.size()-1].HNSW_quantizers.size() << std::endl;
                std::cout << "The total number of nc in this layer: " << vq_quantizer_index[vq_quantizer_index.size()-1].layer_nc << std::endl;
                n_vq++;
            }

            else if(index_type[i] == "LQ"){
                assert (i >= 1);
                upper_centroids.resize(nc_upper * dimension);
                nn_centroids_idxs.resize(nc_upper * nc_per_group);
                nn_centroids_dists.resize(nc_upper * nc_per_group);
                

                PrintMessage("Adding LQ quantizer");
                size_t VQ_size = vq_quantizer_index.size() - 1;
                PrintMessage("VQ computing nn centroids");
#pragma omp parallel for
                for (size_t j = 0; j < nc_upper; j++){
                    std::vector<float> VQ_centroid(dimension);
                    vq_quantizer_index[VQ_size].compute_final_centroid(j, VQ_centroid.data());
                    auto result_queue = this->base_HNSW->searchKnn(VQ_centroid.data(), nc_per_group+1);
                    memcpy(upper_centroids.data() + j * dimension, VQ_centroid.data(), dimension * sizeof(float));
                    for (size_t k = 0; k < nc_per_group; k++){
                        nn_centroids_dists[j * nc_per_group + k] = result_queue.top().first;
                        nn_centroids_idxs[j * nc_per_group + k] = result_queue.top().second;
                        result_queue.pop();
                    }
                }
                std::cout << std::endl;
                add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), LQ_type[n_lq]);

                inner_Trecorder.record_time_usage(record_file, "Trained LQ layer");
                nc_upper = lq_quantizer_index[lq_quantizer_index.size()-1].layer_nc;

                std::cout << i << "th LQ quantizer added, check it " << std::endl;
                std::cout << "The LQ quantizer size is: " <<  lq_quantizer_index.size() << " the num of alphas: " << lq_quantizer_index[lq_quantizer_index.size()-1].alphas.size() << std::endl;;
                std::cout << "The number of final nc in this layer: " << lq_quantizer_index[lq_quantizer_index.size()-1].layer_nc << std::endl;
                n_lq ++;
            }
            else if (index_type[i] == "PQ"){
                //The PQ layer should be placed in the last layer
                assert(i == layers-1);
                PrintMessage("Adding PQ quantizer");
                add_pq_quantizer(nc_upper, PQ_paras[0].first, PQ_paras[0].second);

                inner_Trecorder.record_time_usage(record_file, "Trained PQ layer");
                nc_upper = pq_quantizer_index[pq_quantizer_index.size()-1].layer_nc;
                std::cout << i << "th PQ quantizer added, check it " << std::endl;
                std::cout << "The number of final nc in this layer: " << pq_quantizer_index[pq_quantizer_index.size()-1].layer_nc << std::endl;
                n_pq ++;
            }
        }
        if(use_saving_index)
            write_quantizers(path_quantizer);
        }
    }

    /**
     * 
     * This is the function for training PQ compresser 
     * 
     * Input:
     * 
     * path_pq: the path for saving and loading PQ quantizer for origin vector
     * path_norm_pq: the path for saving and loading PQ quantizer for norm value
     * path_learn: the path for learning dataset
     * 
     **/
    void Bslib_Index::train_pq(const std::string path_pq, const std::string path_learn, const std::string path_OPQ, const size_t train_set_size){

        // Load the train set fot training
        read_train_set(path_learn, this->train_size, train_set_size);

        std::cout << "Initilizing index PQ quantizer " << std::endl;
        this->pq = faiss::ProductQuantizer(this->dimension, this->M_pq, this->nbits);
        this->code_size = this->pq.code_size;
        this->k_sub = this->pq.ksub;

        std::cout << "Assigning the train dataset to compute residual" << std::endl;
        std::vector<float> residuals(train_set_size * dimension);

        std::vector<float> train_vector_alphas;
        if (use_vector_alpha){
            train_vector_alphas.resize(train_set_size);
        }
        assign(train_set_size, this->train_data.data(), train_data_ids.data(), this->layers, train_vector_alphas.data());
        

        for (size_t i = train_set_size - 100; i < train_set_size; i++){std::cout << train_data_ids[i] << " ";} std::cout << std::endl;

        std::cout << "Encoding the train dataset with " << train_set_size<< " data points " << std::endl;
        encode(train_set_size, this->train_data.data(), train_data_ids.data(), residuals.data(), train_vector_alphas.data());

        if (use_OPQ){
            PrintMessage("Training the OPQ matrix");
            this->opq_matrix = faiss::OPQMatrix(dimension, M_pq);
            this->opq_matrix.verbose = true;
            this->opq_matrix.train(train_set_size, residuals.data());
            faiss::write_VectorTransform(& this->opq_matrix, path_OPQ.c_str());
            do_OPQ(train_set_size, residuals.data());
        }

        std::cout << "Training the pq " << std::endl;
        this->pq.verbose = true;
        this->pq.train(train_set_size, residuals.data());

        if(use_saving_index){
            std::cout << "Writing PQ codebook to " << path_pq << std::endl;
            faiss::write_ProductQuantizer(& this->pq, path_pq.c_str());           
        }

    }
    
    /**
     * 
     * This is the function for computing final centroids
     * 
     * The return value of this function is the total number of groups
     * 
     **/
    void Bslib_Index::get_final_num(){
        this->final_graph_num = this->base_HNSW->maxelements_;

        if (this->index_type[layers -1] == "VQ"){
            this->final_group_num =  vq_quantizer_index[vq_quantizer_index.size() -1].layer_nc;
        }
        else if (this->index_type[layers -1] == "LQ"){
            this->final_group_num =  lq_quantizer_index[lq_quantizer_index.size() -1].layer_nc;
        }
        else if (this->index_type[layers - 1] == "PQ"){
            this->final_group_num = pq_quantizer_index[pq_quantizer_index.size() -1].layer_nc;
        }
    }



    /**
     * 
     * This is the function for adding a base batch 
     * 
     * Input:
     * n: the batch size of the batch data                                                 size: size_t
     * data: the base data                                                                 size: n * dimension
     * sequence_ids: the origin sequence id of data                                        size: n
     * group_ids: the group id of the data                                                 size: n
     * group_positions                                                                     size: n
     * 
     * For vector alpha:
     * base_norm = ||c2 + rpq||^2 - c^2 - 2 * alpha2 * c * nn
     * 
     **/
    void Bslib_Index::add_batch(size_t n, const float * data, const idx_t * sequence_ids, const idx_t * group_ids, 
    const size_t * group_positions, const bool base_norm_flag, const bool alpha_flag, const float * vector_alpha){
        time_recorder batch_recorder = time_recorder();
        bool show_batch_time = true;

        std::vector<float> residuals(n * dimension);
        //Compute residuals
        encode(n, data, group_ids, residuals.data(), vector_alpha);


        if (show_batch_time) batch_recorder.print_time_usage("compute residuals                 ");

        if (use_OPQ){
            do_OPQ(n, residuals.data());
        }

        //Compute code for residuals
        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);
        if (show_batch_time) batch_recorder.print_time_usage("PQ encode data residuals             ");

        //Add codes into index
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            idx_t group_id = group_ids[i];
            size_t group_position = group_positions[i];
            for (size_t j = 0; j < this->code_size; j++){this->base_codes[group_id][group_position * code_size + j] = batch_codes[i * this->code_size + j];}
            this->base_sequence_ids[group_id][group_position] = sequence_ids[i];

            if (use_vector_alpha && !alpha_flag){
                this->base_alphas[group_id][group_position] = vector_alpha[i];
            }
        }
        if (show_batch_time) batch_recorder.print_time_usage("add codes to index                ");

        if (!base_norm_flag){
            std::vector<float> decoded_residuals(n * dimension);
            this->pq.decode(batch_codes.data(), decoded_residuals.data(), n);

            if (use_OPQ){
                reverse_OPQ(n, decoded_residuals.data());
            }

            std::vector<float> reconstructed_x(n * dimension);
            decode(n, decoded_residuals.data(), group_ids, reconstructed_x.data(), vector_alpha);
            if (show_batch_time) batch_recorder.print_time_usage("compute reconstructed base vectors ");
            //This is the norm for reconstructed vectors
            // For vector alpha: base_norm = ||c2 + rpq||^2 - c^2 - 2 * alpha2 * c * nn 

#pragma omp parallel for
            for (size_t i = 0; i < n; i++){
                float original_base_norm = faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);

                if (use_vector_alpha){
                    size_t n_lq = lq_quantizer_index.size()-1; idx_t lq_group_id, lq_inner_group_id;
                    lq_quantizer_index[n_lq].get_group_id(group_ids[i], lq_group_id, lq_inner_group_id);
                    float centroid_norm = centroid_norms[lq_group_id];
                    size_t max_nc_per_group = lq_quantizer_index[n_lq].max_nc_per_group;
                    float centroid_product = lq_quantizer_index[n_lq].upper_centroid_product[lq_group_id * max_nc_per_group + lq_inner_group_id];
                    float alpha = vector_alpha[i];
                    base_norms[sequence_ids[i]] = original_base_norm - centroid_norm - 2 * alpha * centroid_product;
                }
                else{
                    base_norms[sequence_ids[i]] =  original_base_norm;
                }
            }
            if (show_batch_time) batch_recorder.print_time_usage("add base norms                     ");
        }

        //The size of base_norm_code or base_norm should be initialized in main function
        /*
        if (use_norm_quantization){
            std::vector<uint8_t> xnorm_codes (n * norm_code_size);
            this->norm_pq.compute_codes(vector_norms, xnorm_codes.data(), n);
            for (size_t i = 0 ; i < n; i++){
                idx_t sequence_id = sequence_ids[i];
                for (size_t j =0; j < this->norm_code_size; j++){
                    this->base_norm_codes[sequence_id * norm_code_size + j] = xnorm_codes[i * this->norm_code_size +j];
                }
            }
        }
        */

        
    }

    /**
     * 
     * This is the function for get a final centroid data
     * 
     * Input: 
     * group id: the id of the final group in the last layer
     * 
     * Output:
     * final centroid: the centroid data of the group id
     * 
     **/
    void Bslib_Index::get_final_centroid(size_t label, float * final_centroid){
        if (index_type[layers - 1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            vq_quantizer_index[n_vq - 1].compute_final_centroid(label, final_centroid);
        }
        else if(index_type[layers -1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            lq_quantizer_index[n_lq - 1].compute_final_centroid(label, final_centroid);
        }
        else{
            size_t n_pq = pq_quantizer_index.size();
            pq_quantizer_index[n_pq - 1].compute_final_centroid(label, final_centroid);
        }
    }


    /**
     * 
     * This is the function for computing norms of the VQ centroids
     * For LQ and VQ layer, we compute the norm directly, but for PQ, we compute it indirectedly
     * The number of centroids in VQ and LQ are: final_nc
     * The size of centroid norm is 0 for PQ layer
     * 
     **/
    void Bslib_Index::compute_centroid_norm(std::string path_centroid_norm){
        if (this->use_vector_alpha){
            std::cout << "No centroid norms for LQ 2 type " << std::endl;
            return;
        }

        this->centroid_norms.resize(final_graph_num);

        if (exists(path_centroid_norm)){
            std::ifstream centroid_norm_input (path_centroid_norm, std::ios::binary);
            readXvec<float> (centroid_norm_input, this->centroid_norms.data(), final_graph_num, 1, false, false);
            centroid_norm_input.close();
        }

        else{
            size_t n_centroids = base_HNSW->maxelements_;
            std::cout << "Computing centroid norm for " << base_HNSW->maxelements_ << " centroids in VQ layer" << std::endl;

#pragma omp parallel for
                for (size_t label = 0; label < n_centroids; label++){
                    this->centroid_norms[label] = faiss::fvec_norm_L2sqr(base_HNSW->getDataByInternalId(label), dimension);
                }



            std::ofstream centroid_norm_output(path_centroid_norm, std::ios::binary);
            centroid_norm_output.write((char * )& final_graph_num, sizeof(uint32_t));
            centroid_norm_output.write((char *) this->centroid_norms.data(), sizeof(float) * this->centroid_norms.size());
            centroid_norm_output.close();
        }
    }

    /**
     * 
     * This is the function for assigning the vectors into group
     * 
     * Input: 
     * assign_data: the vectors to be assigned:                 size: n * dimension
     *  
     * Output:
     * assigned_ids: the result id result for assigned vectors  size: n
     * 
     **/

    void Bslib_Index::assign(const size_t n, const float * assign_data, idx_t * assigned_ids, size_t assign_layer, float * alphas){

        /*
        if (index_type[layers - 1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            vq_quantizer_index[n_vq - 1].search_all(n, 1, assign_data, assigned_ids);
        }
        else if(index_type[layers - 1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            lq_quantizer_index[n_lq - 1].search_all(n, 1, assign_data, assigned_ids);
        }
        else if(index_type[layers - 1] == "PQ"){
            size_t n_pq = pq_quantizer_index.size();
            pq_quantizer_index[n_pq - 1].search_all(n, assign_data, assigned_ids);
        }*/

#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            size_t n_vq = 0, n_lq = 0, n_pq = 0;
            std::vector<idx_t> query_search_id(1 , 0);
            std::vector<float> query_search_alpha(1, 0);
            std::vector<idx_t> query_result_ids;
            std::vector<float> query_result_dists;

            for (size_t j = 0; j < assign_layer; j++){
                assert(n_vq+ n_lq + n_pq == j);

                if (index_type[j] == "VQ"){
                    idx_t group_id = query_search_id[0];
                    size_t group_size = vq_quantizer_index[n_vq].max_nc_per_group;
                    query_result_dists.resize(group_size, 0);
                    query_result_ids.resize(group_size, 0);

                    const float * query_data = assign_data + i * dimension;
                    vq_quantizer_index[n_vq].search_in_group(query_data, group_id, query_result_dists.data(), query_result_ids.data(), 1);
                    if (vq_quantizer_index[n_vq].use_HNSW){
                        query_search_id[j] = query_result_ids[0];
                    }
                    else{
                        std::vector<float> sub_dist(1);
                        keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data()); 
                    }
                    n_vq ++;
                }

                else if(index_type[j] == "LQ") {

                    idx_t group_id = query_search_id[0];
                    size_t group_size = lq_quantizer_index[n_lq].max_nc_per_group;
                    // Copy the upper search result for LQ layer 
                    size_t upper_group_size = query_result_ids.size();
                    
                    std::vector<idx_t> upper_result_ids(upper_group_size, 0);
                    std::vector<float> upper_result_dists(upper_group_size, 0);
                    memcpy(upper_result_ids.data(), query_result_ids.data(), upper_group_size * sizeof(idx_t));
                    memcpy(upper_result_dists.data(), query_result_dists.data(), upper_group_size * sizeof(float));
                    query_result_ids.resize(group_size, 0);
                    query_result_dists.resize(group_size, 0);

                    const float * target_data = assign_data + i * dimension;
                    std::vector<float> query_result_alphas; if(use_vector_alpha){query_result_alphas.resize(group_size);}
                    
                    lq_quantizer_index[n_lq].search_in_group(target_data, upper_result_ids.data(), upper_result_dists.data(), upper_group_size, group_id, query_result_dists.data(), query_result_ids.data(), query_result_alphas.data());
                    std::vector<float> sub_dist(1);
                    if (use_vector_alpha){
                        keep_k_min_alpha(group_size, 1, query_result_dists.data(),query_result_ids.data(), query_result_alphas.data(), sub_dist.data(), query_search_id.data(), query_search_alpha.data());
                    }
                    else{
                        keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data());
                    }
                    n_lq ++;
                }

                else if(index_type[j] == "PQ"){
                    assert(j == this->layers-1);
                    const float * target_data = assign_data + i * dimension;
                    query_result_dists.resize(1);
                    query_result_ids.resize(1);

                    pq_quantizer_index[n_pq].search_in_group(target_data, query_search_id[0], query_result_dists.data(), query_result_ids.data(), 1);
                    query_search_id[0] = query_result_ids[0];
                    n_pq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
            }
            assigned_ids[i] = query_search_id[0];
            if (use_vector_alpha && assign_layer == layers){
                alphas[i] = query_search_alpha[0];
            }
        }
    }

    float Bslib_Index::pq_L2sqr_search(const uint8_t *code)
    {
        float result = 0.;
        const size_t dim = code_size >> 2;
        size_t m = 0;
        for (size_t i = 0; i < dim; i++) {
            result += precomputed_table[k_sub * m + code[m]]; m++;
            result += precomputed_table[k_sub * m + code[m]]; m++;
            result += precomputed_table[k_sub * m + code[m]]; m++;
            result += precomputed_table[k_sub * m + code[m]]; m++;
        }
        return result;
    }

     /**
      * d = || x - y_C - y_R ||^2
      * d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
     * 
     * distance = ||query - (centroids + residual_PQ)||^2
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * (query - centroids) * residual_PQ
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * query * residual_PQ + 2 * centroids * residual_PQ
     *            ||query - centroids||^2 + - ||centroids||^2 + ||residual_PQ + centroids||^2 - 2 * query * residual_PQ 
     *             accurate                                      error                          error
     * 
     * With LQ layer
     *  d = (1 - α) * (|| x - y_C ||^2 - || y_C ||^2) + α * (|| x - y_N ||^2 - || y_N ||^2) + || y_S + y_R ||^2 - 2 * (x|y_R)
     *       -----------------------------------------   -----------------------------------   -----------------   -----------
     *                      term 1                                 term 2                        term 3          term 4
     * 
      * Term1 can be computed in assigning the queries
      * Term2 is the norm of base vectors
      * Term3 is the inner product between x and y_R
      * 
      * With OPQ:
      * Since OPQ is a Orthogonal Matrix, it does not reflect the norm value
      * Only the query * residual_PQ will be reflected
      * 
      * 
      * With vector alpha:
      * ||q - c1 + c1 - c2 - rpq||2 = ||q - c1||^2 + 2 (q - c1) * (c1 - c2 - rpq) + ||c1 - c2 - rpq||^2
      * As we have : (q - c1) * (c1 - c2) = 0 :
      * = ||q - c1||^2 - 2 rpq * (q - c1) + c1 ^ 2 - 2 * c1 (c2 + rpq) + ||c2 + rpq||^2
      * = ||q - c1||^2 - 2 * q * rpq + 2*c1 * rpq + c1^2 - 2*c1*c2 - 2*c1 * rpq + ||c2 + rpq||^2
      * = ||q - c1||2 - 2 * q * rpq + c1^2 - 2 * c1 * c2 + ||c2 + rpq||^2
      * As c1 = c + alpha1 * nn, c2 = c + alpha2 * nn
      * c1 ^2 - 2 * c1*c2 = c^2 + 2*alpha1*nn + alpha1^2*nn^2 -2 (c^2 + (alpha1 + alpha2)*c*nn + alpha1 * alpha2 * nn^2)
      * = - c^2 - 2 * alpha2 * c * nn + (alpha1^2 - 2 * alpha1 * alpha2)nn^2
      * 
      * The total would be: ||q - c1||^2 - 2 * q * rpq + ||c2 + rpq||^2 - c^2 - 2 * alpha2 * c * nn  + (alpha1 ^ 2 - 2 * alpha1 * alpha2)nn^2
      *                      q_c_dist      PQ table      |              base norm                 |    |            centroid norm            |
      * 
      * Input:
      * n: the number of query vectors
      * result_k: the kNN neighbor that we want to search
      * queries: the query vectors                                   size: n * dimension
      * keep_space: the kept centroids in each layer           
      * ground_truth: the groundtruth labels for testing             size: n * result_k
      * 
      * Output:
      * query_ids: the result origin label                           size: n
      * query_dists: the result dists to origin base vectors         size: n * result_k
      * 
      **/

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, const size_t nprobe, std::string path_base){
        // Variables for testing and validation and printing
        // Notice: they should only be activated when parallel is not used

        bool validation = false;

        std::vector<idx_t> centroid_idxs(nprobe);
        
        if (base_HNSW->efSearch < nprobe){base_HNSW->efSearch = nprobe + 1;}
        assert(nprobe < this->final_graph_num);

        std::ifstream base_input(path_base, std::ios::binary);
        for (size_t i = 0; i < n; i++){

            float * distances = query_dists + i * result_k;
            idx_t * labels = query_ids + i * result_k;

            faiss::maxheap_heapify(result_k, distances, labels);

            float * query = queries + i * dimension;
            this->pq.compute_inner_prod_table(query, this->precomputed_table.data());

            // The last layer is VQ layer
            if (VQ_search_type){
                // Search the base graph (the last VQ layer)
                auto result_queue = this->base_HNSW->searchKnn(query, nprobe);

                for (size_t j = 0; j < nprobe; j++){

                    idx_t centroid_idx = result_queue.top().second;
                    float query_centroid_dist = result_queue.top().first;
                    result_queue.pop();

                    float actual_q_c_dist,actual_c_norm;if (validation){
                        actual_q_c_dist = faiss::fvec_L2sqr(query, base_HNSW->getDataByInternalId(centroid_idx), dimension);
                        actual_c_norm = faiss::fvec_norm_L2sqr(base_HNSW->getDataByInternalId(centroid_idx), dimension);}
                    
                    const size_t group_size = base_sequence_ids[centroid_idx].size(); if (group_size == 0){continue;}
                    const uint8_t * base_code = base_codes[centroid_idx].data();
                    const float term1 = query_centroid_dist - centroid_norms[centroid_idx];


                    for (size_t k = 0; k < group_size; k++){
                        const float term3 = 2 * pq_L2sqr_search(base_code + k * code_size);
                        const float dist = term1 + base_norms[base_sequence_ids[centroid_idx][k]] - term3;

                        // The first parameter should be result_k
                        if (dist < distances[0]) {
                            faiss::maxheap_pop(result_k, distances, labels);
                            faiss::maxheap_push(result_k, distances, labels, dist, base_sequence_ids[centroid_idx][k]);
                        }

                        if (validation){
                                std::cout << "Validation " << std::endl;
                                uint32_t dim = 0;
                                std::vector<float> base_vector(dimension);
                                base_input.seekg(base_sequence_ids[centroid_idx][k] * dimension * sizeof(base_data_type) + base_sequence_ids[centroid_idx][k] * sizeof(uint32_t), std::ios::beg);
                                base_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == this->dimension);
                                base_input.read((char *) base_vector.data(), sizeof(base_data_type) * dimension);
                                float actual_base_norm = faiss::fvec_norm_L2sqr(base_vector.data(), dimension);
                                std::cout << "Read base vector" << std::endl;
                                float actual_dist = faiss::fvec_L2sqr(query, base_vector.data(), dimension);

                                std::vector<float> actual_residual_vector(dimension);
                                std::vector<idx_t> id(1); id[0] = centroid_idx;
                                std::cout << "Compute residual " << std::endl;
                                vq_quantizer_index[vq_quantizer_index.size() - 1].compute_residual_group_id(1, id.data(), base_vector.data(), actual_residual_vector.data());
                                float actual_prod_result = 0;
                                std::cout << "Compute prod " << std::endl;
                                for (size_t a = 0; a < dimension; a ++){actual_prod_result += 2 * query[a] * actual_residual_vector[a];}

                                std::cout << "Compute actual dist " << std::endl;
                                float computed_actual_dist = actual_q_c_dist - actual_c_norm + actual_base_norm - actual_prod_result;

                                std::cout << " cmd " << computed_actual_dist << " actd " << actual_dist << " resd " << dist << " aqcd " << actual_q_c_dist << " qcd  " << query_centroid_dist  << std::endl;
                                std::cout << " acn " << actual_c_norm << " cn " << centroid_norms[centroid_idx] << 
                                 " abn " <<  actual_base_norm  <<" bn " << base_norms[base_sequence_ids[centroid_idx][k]] << " fprodr " << 
                                term3 << " prodr " <<  " actr " <<  actual_prod_result << std::endl;
                        }

                    }
                }
            }
            // The last layer is LQ layer
            else{
                auto result_queue = this->base_HNSW->searchBaseLayer(query, this->base_HNSW->efSearch);
                std::vector<idx_t> used_centroid_idxs;
                while (result_queue.size() > nprobe){
                    idx_t centroid_idx = result_queue.top().second;
                    used_centroid_idxs.push_back(centroid_idx);
                    query_centroid_dists[centroid_idx] = result_queue.top().first;
                    result_queue.pop();
                }

                for (size_t j = 0; j < nprobe; j++){
                    idx_t centroid_idx = result_queue.top().second;
                    centroid_idxs[j] = centroid_idx;
                    if (result_queue.top().first > 10e10){std::cout <<  "wrong search results" << result_queue.top().first << " " << centroid_idx << std::endl;exit(0);}
                    assert(centroid_idx < final_graph_num);
                    query_centroid_dists[centroid_idx] = result_queue.top().first;
                    used_centroid_idxs.push_back(centroid_idx);
                    result_queue.pop();
                }

                for (size_t j = 0; j < nprobe; j++){
                    idx_t centroid_idx = centroid_idxs[j];

                    float actual_q_c_dist,actual_c_norm;if (validation){
                        actual_q_c_dist = faiss::fvec_L2sqr(query, base_HNSW->getDataByInternalId(centroid_idx), dimension);
                        actual_c_norm = faiss::fvec_norm_L2sqr(base_HNSW->getDataByInternalId(centroid_idx), dimension);}

                    for (size_t k = 0; k < lq_quantizer_index[0].max_nc_per_group; k++){
                        float alpha = lq_quantizer_index[0].alphas[centroid_idx][k];
                        float term1 = (1 - alpha) * (query_centroid_dists[centroid_idx] - centroid_norms[centroid_idx]);

                        size_t final_label = lq_quantizer_index[0].CentroidDistributionMap[centroid_idx] + k;

                        size_t group_size = base_sequence_ids[final_label].size(); if (group_size == 0){continue;}
                        const uint8_t * base_code = base_codes[final_label].data();
                        assert(base_codes[final_label].size() == group_size * code_size);

                        idx_t nn_idx = lq_quantizer_index[0].nn_centroid_ids[centroid_idx][k];

                        if (query_centroid_dists[nn_idx] < EPS){
                            float * nn_centroid = lq_quantizer_index[0].upper_centroids.data() + nn_idx * dimension;
                            assert(nn_idx < final_graph_num);
                            // Not visited in HNSW graph, compute the distance
                            query_centroid_dists[nn_idx] = faiss::fvec_L2sqr(query, nn_centroid, dimension);
                            used_centroid_idxs.push_back(nn_idx);
                        }

                        if (query_centroid_dists[nn_idx] > 10e10){std::cout << "Wrong dist " << i << " " << query_centroid_dists[nn_idx] << " " << nn_idx << std::endl;exit(0);}

                        const float term2 = alpha * (query_centroid_dists[nn_idx] - centroid_norms[nn_idx]);

                        float actual_q_n_dist, actual_n_norm;if (validation){actual_q_n_dist = faiss::fvec_L2sqr(query, base_HNSW->getDataByInternalId(nn_idx), dimension);actual_n_norm = faiss::fvec_norm_L2sqr(base_HNSW->getDataByInternalId(nn_idx), dimension);}
                        
                        for (size_t temp = 0; temp < group_size; temp++){
                            const float term4 = 2 * pq_L2sqr_search(base_code + temp * code_size);
                            const float dist = term1 + term2 + base_norms[base_sequence_ids[final_label][temp]] - term4; //term3 = norms[j]

                            if (dist < distances[0]){
                                faiss::maxheap_pop(result_k, distances, labels);
                                faiss::maxheap_push(result_k, distances, labels, dist, base_sequence_ids[final_label][temp]);
                            }

                            if (validation){
                                uint32_t dim = 0;
                                std::vector<float> base_vector(dimension);
                                base_input.seekg(base_sequence_ids[final_label][temp] * dimension * sizeof(base_data_type) + base_sequence_ids[final_label][temp] * sizeof(uint32_t), std::ios::beg);
                                base_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == this->dimension);
                                base_input.read((char *) base_vector.data(), sizeof(base_data_type) * dimension);
                                float actual_base_norm = faiss::fvec_norm_L2sqr(base_vector.data(), dimension);
                                std::vector<float> decoded_vector(dimension);
                                std::vector<float> reconstructed_vector(dimension);

                                pq.decode(base_code + temp * code_size, decoded_vector.data());
                                std::vector<idx_t> id(1); id[0] = final_label; std::vector<float> alphas;
                                decode(1, decoded_vector.data(), id.data(), reconstructed_vector.data(), alphas.data());
                                float prod_result = 0;
                                for (size_t a = 0; a < dimension; a ++){prod_result += 2 * query[a] * decoded_vector[a];}
                                std::vector<float> actual_residual_vector(dimension);
                                lq_quantizer_index[0].compute_residual_group_id(1, id.data(), base_vector.data(), actual_residual_vector.data(), alphas.data());
                                float actual_prod_result = 0;
                                for (size_t a = 0; a < dimension; a ++){actual_prod_result += 2 * query[a] * actual_residual_vector[a];}

                                float actual_decoded_norm = faiss::fvec_norm_L2sqr(reconstructed_vector.data(), dimension);
                                float actual_dist = faiss::fvec_L2sqr(query, base_vector.data(), dimension);
                                float computed_actual_dist = (1-alpha) * (actual_q_c_dist - actual_c_norm) + alpha * (actual_q_n_dist - actual_n_norm) + actual_base_norm - actual_prod_result;
                                float * centroid_c =  base_HNSW->getDataByInternalId(centroid_idx);float * centroid_n = base_HNSW->getDataByInternalId(nn_idx);
                                std::vector <float> centroid_LQ(dimension); lq_quantizer_index[0].compute_final_centroid(final_label, centroid_LQ.data());

                                std::cout << "Centroid: ";for (size_t a = 0; a < dimension; a++){std::cout << centroid_c[a] << " ";}std::cout << std::endl;

                                std::cout << "Centroid in LQ: ";for (size_t a = 0; a < dimension; a++){std::cout << lq_quantizer_index[0].upper_centroids[centroid_idx * dimension + a] << " ";}std::cout << std::endl;

                                std::cout << "Nentroid: ";for (size_t a = 0; a< dimension; a++){std::cout << centroid_n[a] << " ";}std::cout << std::endl;

                                std::cout << "Nentroid in LQ: ";for (size_t a = 0; a < dimension; a++){std::cout << lq_quantizer_index[0].upper_centroids[lq_quantizer_index[0].nn_centroid_ids[centroid_idx][k] * dimension + a] << " ";}std::cout << std::endl;

                                std::cout << "Fentroid: ";for (size_t a = 0; a < dimension; a++){std::cout << centroid_LQ[a] << " ";}std::cout << std::endl;

                                std::cout << "alpha " << alpha << " cmd " << computed_actual_dist << " actd " << actual_dist << " resd " << dist << " aqcd " << actual_q_c_dist << " qcd  " << query_centroid_dists[centroid_idx]  << std::endl;
                                std::cout << " acn " << actual_c_norm << " scn " << centroid_norms[centroid_idx] << " aqnd " 
                                << actual_q_n_dist << " qnd " << query_centroid_dists[nn_idx] << " ann " <<
                                actual_n_norm << " nn " << centroid_norms[nn_idx] << " abn " <<
                                actual_base_norm << " adn "<<  actual_decoded_norm <<" bn " << base_norms[base_sequence_ids[final_label][temp]] << " fprodr " << 
                                term4 << " prodr " << prod_result << " actr " <<  actual_prod_result << std::endl;
                            }
                        }
                    }
                }
                // Zero computed dists for later queries
                for (idx_t used_centroid_idx : used_centroid_idxs){query_centroid_dists[used_centroid_idx] = 0;}
            }
        }
        std::cout << "Search finished " << std::endl;
    }


    void Bslib_Index::write_quantizers(const std::string path_quantizer){
        PrintMessage("Writing quantizers");
        std::ofstream quantizers_output(path_quantizer, std::ios::binary);
        size_t n_vq = 0;
        size_t n_lq = 0;
        size_t n_pq = 0;
        quantizers_output.write((char *) this->ncentroids.data(), this->layers * sizeof(size_t));

        for (size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                PrintMessage("Writing VQ quantizer layer");
                const size_t layer_nc = vq_quantizer_index[n_vq].layer_nc;
                const size_t nc_upper = vq_quantizer_index[n_vq].nc_upper;
                const size_t max_nc_per_group = vq_quantizer_index[n_vq].max_nc_per_group;

                quantizers_output.write((char *) & layer_nc, sizeof(size_t));
                quantizers_output.write((char *) & nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & max_nc_per_group, sizeof(size_t));
                quantizers_output.write((char *) vq_quantizer_index[n_vq].exact_nc_in_groups.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char *) vq_quantizer_index[n_vq].CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));
                size_t HNSW_flag = vq_quantizer_index[n_vq].use_HNSW ? 1 : 0;
                quantizers_output.write((char *) & HNSW_flag, sizeof(size_t));


                std::cout << " nc in this layer: " << layer_nc << " nc in upper layer: " << nc_upper << std::endl;

                if (vq_quantizer_index[n_vq].use_HNSW){
                    std::cout << "Writing HNSW indexes " << std::endl;
                    size_t M = vq_quantizer_index[n_vq].M;
                    size_t efConstruction = vq_quantizer_index[n_vq].efConstruction;

                    quantizers_output.write((char * ) & M, sizeof(size_t));
                    quantizers_output.write((char * ) & efConstruction, sizeof(size_t));

                    vq_quantizer_index[n_vq].write_HNSW(quantizers_output);
                }
                else{
                    std::cout << "Writing L2 centroids " << std::endl;
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        size_t group_quantizer_data_size = vq_quantizer_index[n_vq].exact_nc_in_groups[group_id] * this->dimension;
                        assert(vq_quantizer_index[n_vq].L2_quantizers[group_id]->xb.size() == group_quantizer_data_size);
                        quantizers_output.write((char * ) vq_quantizer_index[n_vq].L2_quantizers[group_id]->xb.data(), group_quantizer_data_size * sizeof(float));
                    }
                }
                assert(n_vq + n_lq + n_pq == i);
                n_vq ++;
            }
            else if (index_type[i] == "LQ"){
                PrintMessage("Writing LQ quantizer layer");
                const size_t nc_upper = lq_quantizer_index[n_lq].nc_upper;
                const size_t max_nc_per_group = lq_quantizer_index[n_lq].max_nc_per_group;
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].layer_nc, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].max_nc_per_group, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].LQ_type, sizeof(size_t));

                assert(lq_quantizer_index[n_lq].upper_centroids.size() == nc_upper * dimension);
                quantizers_output.write((char *) lq_quantizer_index[n_lq].upper_centroids.data(), nc_upper * this->dimension*sizeof(float));

                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_ids[j].size() == max_nc_per_group);
                    quantizers_output.write((char *)lq_quantizer_index[n_lq].nn_centroid_ids[j].data(), max_nc_per_group * sizeof(idx_t));
                }
                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_dists[j].size() == max_nc_per_group);
                    quantizers_output.write((char * )lq_quantizer_index[n_lq].nn_centroid_dists[j].data(), max_nc_per_group * sizeof(float));
                }

                if (lq_quantizer_index[n_lq].LQ_type != 2){
                    assert(lq_quantizer_index[n_lq].alphas.size() == nc_upper);
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        quantizers_output.write((char *) lq_quantizer_index[n_lq].alphas[group_id].data(), max_nc_per_group * sizeof(float));
                    }
                }
                else{
                    assert(lq_quantizer_index[n_lq].upper_centroid_product.size() == nc_upper * max_nc_per_group);
                    quantizers_output.write((char *) lq_quantizer_index[n_lq].upper_centroid_product.data(), max_nc_per_group * sizeof (float)); 
                }

                assert(n_vq + n_lq + n_pq == i);
                n_lq ++;
            }
            else if (index_type[i] == "PQ"){
                PrintMessage("Writing PQ quantizer layer");
                const size_t layer_nc = pq_quantizer_index[n_pq].layer_nc;
                const size_t nc_upper = pq_quantizer_index[n_pq].nc_upper;
    
                const size_t M = pq_quantizer_index[n_pq].M;
                const size_t max_nbits = pq_quantizer_index[n_pq].max_nbits;
                const size_t max_ksub = pq_quantizer_index[n_pq].max_ksub;

                quantizers_output.write((char *) & layer_nc, sizeof(size_t));
                quantizers_output.write((char * ) & nc_upper, sizeof(size_t));
                quantizers_output.write((char * ) & M, sizeof(size_t));
                quantizers_output.write((char * ) & max_nbits, sizeof(size_t));
                quantizers_output.write((char * ) & max_ksub, sizeof(size_t));

                quantizers_output.write((char * ) pq_quantizer_index[n_pq].exact_nbits.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char * ) pq_quantizer_index[n_pq].exact_ksubs.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char * ) pq_quantizer_index[n_pq].CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));

                assert(pq_quantizer_index[n_pq].PQs.size() == nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    size_t centroid_size = pq_quantizer_index[n_pq].PQs[j]->centroids.size();
                    assert(centroid_size == dimension * pq_quantizer_index[n_pq].PQs[j]->ksub);
                    quantizers_output.write((char * )pq_quantizer_index[n_pq].PQs[j]->centroids.data(), centroid_size * sizeof(float));
                }

                for (size_t j = 0; j < nc_upper; j++){
                    quantizers_output.write((char *) pq_quantizer_index[n_pq].centroid_norms[j].data(), M* pq_quantizer_index[n_pq].exact_ksubs[j] *sizeof(float));
                }

                assert(n_vq + n_lq + n_pq == i);
                n_pq ++;
            }
            else{
                std::cout << "Index type error: " << index_type[i] << std::endl;
                exit(0);
            }
        }
        PrintMessage("Write quantizer finished");
        quantizers_output.close();
    }


    void Bslib_Index::read_quantizers(const std::string path_quantizer){
        PrintMessage("Reading quantizers ");
        std::ifstream quantizer_input(path_quantizer, std::ios::binary);

        //For each layer, there is nc, nc_upper and nc_per_group
        size_t layer_nc;
        size_t nc_upper;
        size_t max_nc_per_group;
        quantizer_input.read((char *) this->ncentroids.data(), this->layers * sizeof(size_t));

        for(size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                std::cout << "Reading VQ layer" << std::endl;
                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & max_nc_per_group, sizeof(size_t));
                std::vector<size_t> exact_nc_in_group(nc_upper);
                std::vector<size_t> CentroidDistributionMap(nc_upper);
                quantizer_input.read((char *) exact_nc_in_group.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *)CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));
                size_t VQ_HNSW_flag;
                quantizer_input.read((char *) & VQ_HNSW_flag, sizeof(size_t));

                std::cout << " nc in this layer: " << layer_nc << " nc in upper layer: " << nc_upper << std::endl;

                assert(max_nc_per_group * nc_upper >= layer_nc);

                if (VQ_HNSW_flag == 1){
                    size_t M;
                    size_t efConstruction;
                    quantizer_input.read((char *) & M, sizeof(size_t));
                    quantizer_input.read((char *) & efConstruction, sizeof(size_t));
                    
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, max_nc_per_group, M, efConstruction, true);
                    vq_quantizer.layer_nc = layer_nc;
                    for (size_t j = 0; j < nc_upper; j++){
                        vq_quantizer.exact_nc_in_groups[j] = exact_nc_in_group[j];
                        vq_quantizer.CentroidDistributionMap[j] = CentroidDistributionMap[j];
                    }

                    vq_quantizer.read_HNSW(quantizer_input);
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
                else{
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, max_nc_per_group);
                    vq_quantizer.layer_nc = layer_nc;
                    for (size_t j = 0; j < nc_upper; j++){
                        vq_quantizer.exact_nc_in_groups[j] = exact_nc_in_group[j];
                        vq_quantizer.CentroidDistributionMap[j] = CentroidDistributionMap[j];
                    }
                    
                    for (size_t j = 0; j < nc_upper; j++){
                        std::vector<float> centroids(exact_nc_in_group[j] * this->dimension);
                        quantizer_input.read((char *) centroids.data(), exact_nc_in_group[j] * dimension * sizeof(float));
                        faiss::IndexFlatL2 * centroid_quantizer = new faiss::IndexFlatL2(dimension);
                        centroid_quantizer->add(exact_nc_in_group[j], centroids.data());
                        vq_quantizer.L2_quantizers[j] = centroid_quantizer;
                    }
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
            }

            else if (index_type[i] == "LQ"){
                std::cout << "Reading LQ layer " << std::endl;
                size_t LQ_type;
                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & max_nc_per_group, sizeof(size_t));
                quantizer_input.read((char *) & LQ_type, sizeof(size_t));

                assert(max_nc_per_group * nc_upper == layer_nc);
                std::cout << " nc in this layer: " << layer_nc << " nc in upper layer: " << nc_upper << " max nc in group: " << max_nc_per_group << " " << std::endl;
                std::vector<float> alphas(max_nc_per_group);
                std::vector<float> upper_centroids(nc_upper * dimension);
                std::vector<idx_t> nn_centroid_ids(nc_upper * max_nc_per_group);
                std::vector<float> nn_centroid_dists(nc_upper * max_nc_per_group);

                quantizer_input.read((char *) upper_centroids.data(), nc_upper * this->dimension * sizeof(float));
                quantizer_input.read((char *) nn_centroid_ids.data(), nc_upper * max_nc_per_group * sizeof(idx_t));
                quantizer_input.read((char *) nn_centroid_dists.data(), nc_upper * max_nc_per_group * sizeof(float));

                LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, max_nc_per_group, upper_centroids.data(), nn_centroid_ids.data(), nn_centroid_dists.data(), LQ_type);
                
                if (LQ_type != 2){
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        quantizer_input.read((char *) alphas.data(), max_nc_per_group * sizeof(float));
                        lq_quantizer.alphas[group_id].resize(max_nc_per_group);
                        for (size_t inner_group_id = 0; inner_group_id < max_nc_per_group; inner_group_id++){
                            lq_quantizer.alphas[group_id][inner_group_id] = alphas[inner_group_id];
                        }
                    }
                }
                else{
                    lq_quantizer.upper_centroid_product.resize(nc_upper * max_nc_per_group);
                    quantizer_input.read((char *) lq_quantizer.upper_centroid_product.data(), nc_upper * max_nc_per_group * sizeof(float));
                }
                
                this->lq_quantizer_index.push_back(lq_quantizer);
            }

            else if (index_type[i] == "PQ"){
                std::cout << "Reading PQ layer " << std::endl;
                size_t layer_nc;
                size_t M;
                size_t max_nbits;
                size_t max_ksub;

                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & M, sizeof(size_t));
                quantizer_input.read((char *) & max_nbits, sizeof(size_t));
                quantizer_input.read((char *) & max_ksub, sizeof(size_t));
                std::cout << " M in PQ layer: " << M << " max nbits in PQ layer: " << max_nbits << std::endl;
                
                
                PQ_quantizer pq_quantizer = PQ_quantizer(dimension, nc_upper, M, max_nbits);
                pq_quantizer.layer_nc = layer_nc;
                quantizer_input.read((char *) pq_quantizer.exact_nbits.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *) pq_quantizer.exact_ksubs.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *) pq_quantizer.CentroidDistributionMap.data(), nc_upper * sizeof(size_t));

                pq_quantizer.PQs.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    faiss::ProductQuantizer * product_quantizer = new faiss::ProductQuantizer(dimension, M, nbits);
                    size_t centroid_size = dimension * product_quantizer->ksub;
                    quantizer_input.read((char *) product_quantizer->centroids.data(), centroid_size * sizeof(float));
                    pq_quantizer.PQs[j] = product_quantizer;
                }

                pq_quantizer.centroid_norms.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    pq_quantizer.centroid_norms[j].resize(M * pq_quantizer.exact_ksubs[j]);
                    quantizer_input.read((char *) pq_quantizer.centroid_norms[j].data(), M * pq_quantizer.exact_ksubs[j] * sizeof(float));
                }
                this->pq_quantizer_index.push_back(pq_quantizer);
            }
        }
        PrintMessage("Read quantizers finished");
        quantizer_input.close();
    }

    void Bslib_Index::write_index(const std::string path_index){
        std::ofstream output(path_index, std::ios::binary);
        output.write((char *) & this->final_group_num, sizeof(size_t));


        output.write((char *) & this->final_group_num, sizeof(size_t));
        output.write((char *) & this->final_graph_num, sizeof(size_t));
        assert(base_codes.size() == final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            assert(base_codes[i].size() / code_size == base_sequence_ids[i].size());
            size_t group_size = base_codes[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_codes[i].data(), group_size * sizeof(uint8_t));
        }

        output.write((char *) & this->final_group_num, sizeof(size_t));
        assert(base_sequence_ids.size() == final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            size_t group_size = base_sequence_ids[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_sequence_ids[i].data(), group_size * sizeof(idx_t));
        }

        if(!use_vector_alpha){
            assert(centroid_norms.size() == final_graph_num);
            output.write((char *) & this->final_graph_num, sizeof(size_t));
            assert(centroid_norms.size() == this->final_graph_num);
            output.write((char *) centroid_norms.data(), this->final_graph_num * sizeof(float));
        }
        output.close();
    }

    /*
        Read the index file to get the whole constructed index
        Input: 
            path_index:    str, the path to get the index
        
        Output:
            None (The index is updated)

    */
    void Bslib_Index::read_index(const std::string path_index){
        std::ifstream input(path_index, std::ios::binary);
        size_t final_nc_input;
        size_t final_ng_input;
        size_t group_size_input;

        this->base_codes.resize(this->final_group_num);
        
        this->base_sequence_ids.resize(this->final_group_num);
        input.read((char *) & final_nc_input, sizeof(size_t));


        input.read((char *) & final_nc_input, sizeof(size_t));
        input.read((char *) & final_ng_input, sizeof(size_t));

        std::cout << final_nc_input << " " << final_group_num << std::endl;
        assert(final_nc_input == this->final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            base_codes[i].resize(group_size_input);
            input.read((char *) base_codes[i].data(), group_size_input * sizeof(uint8_t));
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            base_sequence_ids[i].resize(group_size_input);
            input.read((char *) base_sequence_ids[i].data(), group_size_input * sizeof(idx_t));
        }

        if (!use_vector_alpha){
            input.read((char *) & final_ng_input, sizeof(size_t));
            assert(final_ng_input == this->final_graph_num);
            this->centroid_norms.resize(this->final_graph_num);
            input.read((char *) centroid_norms.data(), this->final_graph_num * sizeof(float));
        }

        input.close();
    }

    /*
        Construct the index with the train set
        Input:
            path_learn:    str, the path to read the dataset

        Output: 
            None  (The index is updated)
    */
    void Bslib_Index::build_index(std::string path_learn,
    std::string path_quantizers, std::string path_edges, std::string path_info, std::string path_centroids,
    size_t VQ_layers, size_t PQ_layers, size_t LQ_layers, 
    const uint32_t * ncentroids, const size_t * M_HNSW, const size_t * efConstruction, 
    const size_t * M_PQ_layer, const size_t * nbits_PQ_layer, const size_t * num_train,
    const size_t * LQ_type, std::ofstream & record_file){

        PrintMessage("Initializing the index");
        Trecorder.reset();

        // Prepare the parameters for VQ index and HNSW index
        std::vector<HNSW_PQ_para> HNSW_paras;
        for (size_t i = 0; i < VQ_layers; i++){
            HNSW_PQ_para new_para; new_para.first = M_HNSW[i]; new_para.second = efConstruction[i];
            HNSW_paras.push_back(new_para);
        }

        std::vector<HNSW_PQ_para> PQ_paras;
        for (size_t i = 0; i < PQ_layers; i++){
            HNSW_PQ_para new_para; new_para.first = M_PQ_layer[i]; new_para.second = nbits_PQ_layer[i];
            PQ_paras.push_back(new_para);
        }

        // LQ layer type: 0: original, 1: different alpha in sub-group 2: different alpha for vectors
        if (index_type[layers-1] == "LQ" && LQ_type[LQ_layers -1] == 2){
            use_vector_alpha = true;
        }

        PrintMessage("Constructing the quantizers");
        this->build_quantizers(ncentroids, path_quantizers, path_learn, path_edges, path_info, path_centroids, num_train, HNSW_paras, PQ_paras, LQ_type, record_file);
        this->get_final_num();
        

        if (use_recording){
            std::string message = "Constructed the index, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    /*
        Assign the whole dataset to the index
        Input: 
            path:            str, the path to read the dataset
            batch_size:      int, number of vectors in each batch
            nbatches:        int, number of batches to be partitioned

        Output:
            None    (The index is updated)
    */
    void Bslib_Index::assign_vectors(std::string path_ids, std::string path_base, std::string path_alphas_raw,
        uint32_t batch_size, size_t nbatches,  std::ofstream & record_file){

        PrintMessage("Assigning the points");
        Trecorder.reset();
        if (!exists(path_ids)){
            std::ifstream base_input (path_base, std::ios::binary);
            std::ofstream base_output (path_ids, std::ios::binary);
            std::ofstream alphas_output;

            std::vector <float> batch(batch_size * dimension);
            std::vector<idx_t> assigned_ids(batch_size);
            std::vector<float> vector_alphas;
            if (use_vector_alpha){
                vector_alphas.resize(batch_size); 
                alphas_output = std::ofstream(path_alphas_raw, std::ios::binary);
            }

            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size, false, false);
                this->assign(batch_size, batch.data(), assigned_ids.data(), this->layers, vector_alphas.data());
                base_output.write((char * ) & batch_size, sizeof(uint32_t));
                base_output.write((char *) assigned_ids.data(), batch_size * sizeof(idx_t));
                alphas_output.write((char * ) & batch_size, sizeof(uint32_t));
                alphas_output.write((char * ) vector_alphas.data(), batch_size * sizeof(float));

                if (i % 10 == 0){
                    std::cout << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                    record_file << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.record_time_usage(record_file, " ");
                }
            }
            base_input.close();
            base_output.close();
            if (use_recording){
                std::string message = "Assigned the base vectors in sequential mode";
                Mrecorder.print_memory_usage(message);
                Mrecorder.record_memory_usage(record_file,  message);
                Trecorder.print_time_usage(message);
                Trecorder.record_time_usage(record_file, message);
            }
        }
    }

    /*
        Train the PQ quantizer
        Input:
        Output:
    */
    void Bslib_Index::train_pq_quantizer(const std::string path_pq,
        const size_t M_pq, const std::string path_learn, const std::string path_OPQ, const size_t PQ_train_size,  std::ofstream & record_file){
        
        PrintMessage("Constructing the PQ compressor");
        Trecorder.reset();
        if (exists(path_pq)){
            
            this->pq = * faiss::read_ProductQuantizer(path_pq.c_str());
            this->code_size = this->pq.code_size;
            this->k_sub = this->pq.ksub;
            std::cout << "Loading PQ codebook from " << path_pq << std::endl;

            if (use_OPQ){
                this->opq_matrix = * dynamic_cast<faiss::LinearTransform *>((faiss::read_VectorTransform(path_OPQ.c_str())));
                std::cout << "Loading OPQ matrix from " << path_OPQ << std::endl;
            }
        }
        else
        {
            this->M_pq = M_pq;

            std::cout << "Training PQ codebook" << std::endl;
            this->train_pq(path_pq, path_learn, path_OPQ, PQ_train_size);
        }
        std::cout << "Checking the PQ with its code size:" << this->pq.code_size << std::endl;
        this->precomputed_table.resize(pq.ksub * pq.M);

        if (use_recording){
            std::string message = "Trained the PQ, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }
    

    void Bslib_Index::load_index(std::string path_index, std::string path_ids, std::string path_base,
        std::string path_base_norm, std::string path_centroid_norm, std::string path_alphas_raw,
        std::string path_base_alphas, size_t batch_size, size_t nbatches, size_t nb, std::ofstream & record_file){
        Trecorder.reset();
        if (exists(path_index)){
            PrintMessage("Loading pre-constructed index");

            if (use_vector_alpha){
                read_base_alphas(path_base_alphas);
            }

            this->base_norms.resize(nb);
            std::ifstream base_norms_input(path_base_norm, std::ios::binary);
            readXvec<float>(base_norms_input, base_norms.data(), nb, 1, false, false);

            read_index(path_index);

            if (use_recording){
                std::string message = "Loaded pre-constructed index ";
                Mrecorder.print_memory_usage(message);
                Mrecorder.record_memory_usage(record_file,  message);
                Trecorder.print_time_usage(message);
                Trecorder.record_time_usage(record_file, message);
            }
        }
        else{
            PrintMessage("Loading the index");
            std::vector<idx_t> ids(nb);

            std::ifstream ids_input(path_ids, std::ios::binary);
            readXvec<idx_t> (ids_input, ids.data(), batch_size, nbatches); 
            //std::vector<idx_t> pre_hash_ids; if (use_hash) {pre_hash_ids.resize(nb, 0); memcpy(pre_hash_ids.data(), ids.data(), nb * sizeof(idx_t)); HashMapping(nb, pre_hash_ids.data(), ids.data(), index.final_group_num);}
            std::vector<size_t> groups_size(this->final_group_num, 0); std::vector<size_t> group_position(nb, 0);
            for (size_t i = 0; i < nb; i++){group_position[i] = groups_size[ids[i]]; groups_size[ids[i]] ++;}

            this->base_codes.resize(this->final_group_num);
            this->base_sequence_ids.resize(this->final_group_num);

            this->base_norms.resize(nb);
            for (size_t i = 0; i < this->final_group_num; i++){
                this->base_codes[i].resize(groups_size[i] * this->code_size);
                this->base_sequence_ids[i].resize(groups_size[i]);
            }

            std::vector<float> vector_alphas(nb);
            std::ifstream base_alphas_raw_input = std::ifstream(path_alphas_raw, std::ios::binary);
            readXvec<float> (base_alphas_raw_input, vector_alphas.data(), batch_size, nbatches);
            base_alphas_raw_input.close();

            // Whether the alpha vector is loaded in index
            bool alpha_flag = false;

            if (use_vector_alpha){
                if (exists(path_base_alphas)){
                    read_base_alphas(path_base_alphas);
                    alpha_flag = true;
                }
                else{
                    //Load the alpha in nb size
                    size_t n_lq = lq_quantizer_index.size() - 1;
                    assert(index_type[layers-1] == "LQ" && lq_quantizer_index[n_lq].LQ_type == 2);

                    base_alphas.resize(this->final_group_num);

                    for (size_t i = 0; i < final_group_num; i++){
                        base_alphas[i].resize(groups_size[i]);
                    }
                }
            }

            std::ifstream base_input(path_base, std::ios::binary);
            std::vector<float> base_batch(batch_size * dimension);
            std::vector<idx_t> batch_sequence_ids(batch_size);

            bool base_norm_flag = false;
            if (exists(path_base_norm)){
                base_norm_flag = true;
                std::cout << "Loading pre-computed base norm " << std::endl;
                std::ifstream base_norms_input(path_base_norm, std::ios::binary);
                readXvec<float>(base_norms_input, base_norms.data(), nb, 1, false, false);
                base_norms_input.close();
            }

            std::cout << "Start adding batches " << std::endl;
            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, base_batch.data(), dimension, batch_size);
                for (size_t j = 0; j < batch_size; j++){batch_sequence_ids[j] = batch_size * i + j;}

                if (alpha_flag){
                    this->add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, 
                        group_position.data()+i*batch_size, base_norm_flag, alpha_flag, vector_alphas.data());
                }
                else{
                    this->add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, 
                    group_position.data()+i*batch_size, base_norm_flag, alpha_flag,
                    vector_alphas.data() + i * batch_size);
                }

                if (i % 10 == 0){
                    std::cout << " adding batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                    record_file << " adding batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.record_time_usage(record_file, " ");
                }
            }

            if (!base_norm_flag){
                std::ofstream base_norm_output(path_base_norm, std::ios::binary);
                base_norm_output.write((char * )& nb, sizeof(uint32_t));
                base_norm_output.write((char *) base_norms.data(), base_norms.size() * sizeof(float));
                base_norm_output.close();
            }

            if (use_vector_alpha && !alpha_flag){
                write_base_alphas(path_base_alphas);
            }

            this->compute_centroid_norm(path_centroid_norm);

            //In order to save disk usage
            //Annotate the write_index function
            if (this->use_saving_index){
                this->write_index(path_index);
            }
            std::string message = "Constructed and wrote the index ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    void Bslib_Index::index_statistic(std::string path_base, std::string path_ids, 
                                        size_t nb, size_t nbatch){
        // Average distance between the base vector and centroid

        std::ifstream vector_ids(path_ids, std::ios::binary);
        std::ifstream base_vectors(path_base, std::ios::binary);

        size_t test_size = nb;

        std::vector<float> vectors(dimension * test_size);
        std::vector<idx_t> ids(test_size);
    
        readXvec<idx_t>(vector_ids, ids.data(), nb / nbatch, nbatch);

        readXvecFvec<base_data_type> (base_vectors, vectors.data(), dimension, test_size);

        std::vector<float> alphas_raw(test_size / nbatch);

        std::vector<float> vector_residuals(dimension * test_size);

        encode(test_size, vectors.data(), ids.data(), vector_residuals.data(), alphas_raw.data());

        float avg_dist = 0;
        for (size_t i = 0; i < test_size; i++){
            float dist = faiss::fvec_norm_L2sqr(vector_residuals.data() + i * dimension, dimension);
            avg_dist += dist;
        }
        std::cout << std::endl << "Ag dist: " << avg_dist / test_size<< std::endl;
        std::cout << "The base graph size: " << this->final_graph_num << std::endl;
        std::cout << "The final group size: " << this-> final_group_num << std::endl;

        float min_group_size, max_group_size, mean_group_size, std_group_size = 0;
        mean_group_size = nb / final_group_num;
        min_group_size = nb;
        max_group_size = 0;

        for (size_t i = 0; i < final_group_num; i++){
            float group_size = base_sequence_ids[i].size();
            std_group_size += (group_size - mean_group_size) * (group_size - mean_group_size);
            if (group_size > max_group_size){max_group_size = group_size;}
            if (group_size < min_group_size){min_group_size = group_size;}
        }
        std_group_size = this->final_group_num > 0 ? sqrt(std_group_size / this->final_group_num) / mean_group_size : 0;
        std::cout << "The max, min, std of group size: " << max_group_size << " " << min_group_size << " " << std_group_size << std::endl;
    }


    void Bslib_Index::query_test(size_t num_search_paras, size_t num_recall, size_t nq, size_t ngt, const size_t * nprobes,
         const size_t * result_k,
        std::ofstream & record_file, std::ofstream & qps_record_file, 
        std::string search_mode, std::string path_gt, std::string path_query, std::string path_base){

        PrintMessage("Loading groundtruth");
        std::vector<uint32_t> groundtruth(nq * ngt);
        {
            std::ifstream gt_input(path_gt, std::ios::binary);
            readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, false);
        }

        PrintMessage("Loading queries");
        std::vector<float> queries(nq * dimension);
        {
            std::ifstream query_input(path_query, std::ios::binary);
            readXvecFvec<base_data_type>(query_input, queries.data(), dimension, nq, false, false);
        }

        PrintMessage("Start Search");
        this->VQ_search_type = this->index_type[layers -1] == "VQ" ? true : false;
        if (!this->VQ_search_type){query_centroid_dists.resize(final_graph_num);}

        for (size_t i = 0; i < num_search_paras; i++){

        for (size_t j = 0; j < num_recall; j++){
            size_t recall_k = result_k[j];
            std::vector<float> query_distances(nq * recall_k);
            std::vector<idx_t> query_labels(nq * recall_k);
            size_t correct = 0;

            Trecorder.reset();
            search(nq, recall_k, queries.data(), query_distances.data(), query_labels.data(), nprobes[i], path_base);
            std::cout << "The qps for searching is: " << Trecorder.getTimeConsumption() / nq << " us " << std::endl;
            std::string message = "Finish Search ";
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
            Trecorder.record_time_usage(qps_record_file, message);

            if (use_OPQ){
                reverse_OPQ(nq, queries.data());
            }

            for (size_t i = 0; i < nq; i++){
                std::unordered_set<idx_t> gt;

                //for (size_t j = 0; j < recall_k; j++){
                for (size_t j = 0; j < 1; j++){
                    gt.insert(groundtruth[ngt * i + j]);
                }

                //assert (gt.size() == recall_k);
                for (size_t j = 0; j < recall_k; j++){
                    if (gt.count(query_labels[i * recall_k + j]) != 0){
                        correct ++;
                    }
                }
            }
            //float recall = float(correct) / (recall_k * nq);
            float recall = float(correct) / (nq);
            Rrecorder.print_recall_performance(nq, recall, recall_k, search_mode, nprobes[i]);
            Rrecorder.record_recall_performance(record_file, nq, recall, recall_k, search_mode, nprobes[i]);
            Rrecorder.record_recall_performance(qps_record_file, nq, recall, recall_k, search_mode, nprobes[i]);
            std::cout << std::endl;
        }
        }
    }


    void Bslib_Index::write_base_alphas(std::string path_base_alpha){
        assert(use_vector_alpha);
        assert(base_alphas.size() == this->final_group_num);
        std::ofstream base_alphas_output(path_base_alpha, std::ios::binary);
        base_alphas_output.write((char *) & final_group_num, sizeof(size_t));
        for (size_t i = 0; i < final_group_num; i++){
            size_t group_size = this->base_alphas[i].size();
            base_alphas_output.write((char * ) & group_size, sizeof(size_t));
            base_alphas_output.write((char * ) this->base_alphas[i].data(), group_size * sizeof(float));
        }
        base_alphas_output.close();
    }

    void Bslib_Index::read_base_alphas(std::string path_base_alpha){
        std::cout << "Loading base alphas " << std::endl;
        assert(use_vector_alpha);
        assert(this->base_alphas.size() == 0);
        std::ifstream base_alphas_input (path_base_alpha, std::ios::binary);
        size_t group_num;
        base_alphas_input.read((char *) & group_num, sizeof(size_t));
        
        assert(group_num == this->final_group_num);
        this->base_alphas.resize(group_num);
        size_t group_size;
        for (size_t i = 0; i < group_num; i++){
            base_alphas_input.read((char *) & group_size, sizeof(size_t));
            this-> base_alphas[i].resize(group_size);
            base_alphas_input.read((char *) this->base_alphas[i].data(), group_size * sizeof(float));
        }
    }
}

