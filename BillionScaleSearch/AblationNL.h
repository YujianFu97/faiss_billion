#include "Index/BIndex.h"
#include "parameters/BillionScale/ParameterResults.h"
#include "utils/utils.h"

#include "FastPFor/headers/codecfactory.h"

struct cmp{

    bool operator ()( std::pair<uint32_t, float> &a, std::pair<uint32_t, float> &b)
    {
        return a.second < b.second;
    }
};

struct  nlcmp
{
    bool operator ()( std::pair<std::pair<uint32_t, uint32_t>, float> &a, std::pair<std::pair<uint32_t, uint32_t>, float> &b)
    {
        return a.second > b.second;
    }
};

// ||q - (CN + alpha * (CT - CN))|| = ||q - (1 - alpha)CN - alpha * CT||
// = (1 - Alpha) * ||q - CenN|| + Alpha * ||q - CenT|| + Alpha * (1 - Alpha) * ||CenN - CenT||
// Wrong: = (2 * alpha - alpha * alpha) * ||q - CenT|| + (1 - alpha * alpha) * ||q - CenN|| + 2 * alpha * (alpha - 1) * ||CenT - CenN||
inline float FetchQueryNLDist(float QCenTarDist, float QCenNNDist, float CenNNCenTarDist, float alpha){
    //float alphasqaure = alpha * alpha;
    //return (2 * alpha - alphasqaure) * QCenTarDist + (1 - alphasqaure) * QCenNNDist + 2 * (alphasqaure - alpha) * CenNNCenTarDist;
    return (1 - alpha) * QCenNNDist + alpha * QCenTarDist - alpha * (1 - alpha) * CenNNCenTarDist;
}

template<typename T>
void SampleSet(float * ResultData, std::string PathLearn, size_t LoadSize, size_t TotalSize, size_t Dimension){

    if (LoadSize == TotalSize){
        std::ifstream TrainInput(PathLearn, std::ios::binary);
        readXvecFvec<T>(TrainInput, ResultData, Dimension, LoadSize, true, true);
    }
    else{
        long RandomSeed = 1234;
        std::vector<int> RandomId(TotalSize); faiss::RandomGenerator RNG(RandomSeed);

        faiss::rand_perm(RandomId.data(), TotalSize, RandomSeed+1);
#pragma omp parallel for
        for (size_t i = 0; i < LoadSize; i++){
            std::ifstream TrainInput(PathLearn, std::ios::binary);
            std::vector<T> OriginData(Dimension);
            TrainInput.seekg(RandomId[i] * (Dimension * sizeof(T) + sizeof(uint32_t)) + sizeof(uint32_t), std::ios::beg);
            TrainInput.read((char *) OriginData.data(), Dimension * sizeof(T));
            for (size_t j = 0; j < Dimension; j++){
                ResultData[i * Dimension + j] = 1.0 * OriginData[j];
            }
            TrainInput.close();
        }
        std::cout << "Check the sampled vectors: \n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << ResultData[i] << " ";
        }
        std::cout << "\n";
        for (size_t i = 0; i < Dimension; i++){
            std::cout << ResultData[(LoadSize-1) * Dimension + i] << " ";
        }
        std::cout << "\n";
    }
    return;
}

bool Overlap(uint32_t * Vec1, int64_t * Vec2, uint32_t size1, uint32_t size2)
{

    for (size_t i = 0; i < size1; i++){
        for (size_t j = 0; j < size2; j++){
            if (Vec1[i] == Vec2[j]){
                return true;
            }
        }
    }
    return false;
}

bool Intersect(uint32_t * first1, uint32_t * last1, int64_t * first2, int64_t * last2)
{
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2) {
            ++first1;
            continue;
        }
        if (*first2 < *first1) {
            ++first2;
            continue;
        } 
        return true;
    }
    return false;
}

void PrintNL(std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> & NeighborList, std::vector<std::unordered_map<uint32_t, float>> & NeighborListAlpha){
    assert(NeighborList.size() == nc && NeighborListAlpha.size() == nc);

    std::cout << "The structure of NL: \n";
    for (size_t i = 0; i < nc; i++){
        std::cout << "Neighbor List Alpha of NNCluster: " << i << "\n";
        for (auto it = NeighborListAlpha[i].begin(); it != NeighborListAlpha[i].end(); it++){
            std::cout << it->first << " " << it->second << " | ";
        }

        std::cout << "\nVector IDs and their target clusters: ";

        for (auto it = NeighborList[i].begin(); it != NeighborList[i].end(); it++){
            std::cout << it->first << ": ";
            for (size_t j = 0; j < it->second.size(); j++){
                std::cout << it->second[j] << " ";
            }
            std::cout << "| ";
        }
        std::cout << "\n\n";
    } 
    exit(0);
    return;
}

uint32_t SearchMultiFull(time_recorder & Trecorder, size_t nq, size_t RecallK, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem, size_t EfNeighborList, bool UseList,
hnswlib::HierarchicalNSW * CenGraph, float * QCDist, uint32_t * QCID, std::vector<std::vector<uint32_t>> & BaseCompIds, float * CentroidsNorm, uint32_t * BaseIdCompIndice,
float * NeighborListAlpha, float * NeighborListAlphaNorm, uint32_t * NeighborListAlphaTar, uint32_t * NeighborListAlphaIndice, uint32_t * NeighborListAlphaCompIndice, uint32_t * NeighborListVec, uint32_t * NeighborListVecIndice,
uint32_t * NeighborListNumTarIndice, uint32_t * NeighborListNumTar, uint32_t * NeighborListTarIDIndice, uint32_t * NeighborListTarID,
int64_t * NL2SearchID, float * NL2SearchDist, DataType * BaseVectors, uint32_t * DeCompBaseIDs, uint32_t * DeCompAlphaTarID, uint32_t * DeCompVecID, uint32_t * DeCompNumTar, uint32_t * DeCompTarID, int64_t * ClusterIDSet, uint32_t * ClusterIDSetIndice,
FastPForLib::IntegerCODEC & codec, uint32_t MaxAlphaListSize, uint32_t MaxVecListSize, uint32_t MaxTarListSize
){
    size_t CompStartIndice, CompIndice, CompSize, CompSize1, CompSize2, CompSize3, StartIndice, EndIndice, SumVisitedItem = 0;

    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto ResultQueue = CenGraph->searchBaseLayer(Query + QueryIdx * Dimension, EfSearch);
        for (size_t i = 0; i < EfSearch; i++){
            QCDist[QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().first;
            QCID[QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().second;
            ResultQueue.pop();
        }
        faiss::maxheap_heapify(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK);
    }


    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        size_t VisitedItem = 0;
        uint32_t CenClusterID = QCID[QueryIdx * EfSearch];
        // Search the vectors in the base target cluster

        codec.decodeArray(BaseCompIds[CenClusterID].data(), BaseCompIds[CenClusterID].size(), DeCompBaseIDs, CompSize);
        for (size_t j = 0; j < CompSize; j++){
            float Dist = faiss::fvec_L2sqr(BaseVectors + DeCompBaseIDs[j] * Dimension, Query + QueryIdx * Dimension, Dimension);

            if (Dist < QueryDists[QueryIdx * RecallK]){
                faiss::maxheap_pop(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK);
                faiss::maxheap_push(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK, Dist, DeCompBaseIDs[j]);
            }
        }

        VisitedItem += CompSize;

        if (VisitedItem > MaxItem){
            SumVisitedItem += VisitedItem;
            continue;
        }
        
        if (UseList){
            //Trecorder.reset();
            // Todo: remove this assertion
            assert(EfNeighborList <= EfSearch * (EfSearch - 1));

            //int64_t NL2SearchID[EfNeighborList];
            //float NL2SearchDist[EfNeighborList];
            faiss::maxheap_heapify(EfNeighborList, NL2SearchDist, NL2SearchID);

            // Compute the distance between the vector and neighbor lists
            for (size_t i = 1; i < EfSearch; i++){
                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i];

                // Decompress the alpha target ID with selected NNCluster
                CompStartIndice = NNClusterID == 0 ? 0 : NeighborListAlphaCompIndice[NNClusterID - 1];
                CompIndice = NeighborListAlphaCompIndice[NNClusterID] - CompStartIndice;
                if (CompIndice == 0){continue;}
                CompSize1 = MaxAlphaListSize;

                codec.decodeArray(NeighborListAlphaTar + CompStartIndice, CompIndice, DeCompAlphaTarID, CompSize1);

                StartIndice = NNClusterID == 0 ? 0 : NeighborListAlphaIndice[NNClusterID - 1];
                EndIndice = NeighborListAlphaIndice[NNClusterID];
                assert(CompSize1 == (EndIndice - StartIndice));

                for (uint32_t j = 0; j < EfSearch; j++){
                    if (i == j) continue;
                    uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + j];

                    for (size_t temp = 0; temp < CompSize1; temp++){
                        if (TargetClusterID == DeCompAlphaTarID[temp]){
                            float Alpha = NeighborListAlpha[StartIndice + temp];
                            float AlphaNorm = NeighborListAlphaNorm[StartIndice + temp];
                            float QueryNLDist = (1 - Alpha) * (QCDist[QueryIdx * EfSearch + i] - CentroidsNorm[NNClusterID]) + Alpha * (QCDist[QueryIdx * EfSearch + j] - CentroidsNorm[TargetClusterID]) + AlphaNorm;

                            int64_t NLIndice = i * EfSearch + j;
                            if (QueryNLDist < NL2SearchDist[0]){
                                //std::cout << NLIndice << " Inserted\n";
                                faiss::maxheap_pop(EfNeighborList, NL2SearchDist, NL2SearchID);
                                faiss::maxheap_push(EfNeighborList, NL2SearchDist, NL2SearchID, QueryNLDist, NLIndice);
                            }
                            break;
                        }
                    }
                }
            }

            //int64_t ClusterIDSet[(EfSearch - 1) * EfSearch]; 
            //uint32_t ClusterIDSetIndice[EfSearch - 1];
            //Trecorder.recordTimeConsumption1();

            //Trecorder.reset();

            for (size_t i = 0; i < EfSearch -1; i++){
                for (size_t j = 0; j < EfSearch; j++){
                    ClusterIDSet[i * EfSearch + j] = -1;
                    //ClusterIDSet[i][j] = -1;
                }
                ClusterIDSetIndice[i] = 0;
            }
            
            for (size_t i = 0; i < EfNeighborList; i++){
                if (NL2SearchID[i] < 0){continue;}
                uint32_t NNClusterIndice = NL2SearchID[i] / EfSearch;
                uint32_t TargetClusterIndice = NL2SearchID[i] % EfSearch;
                uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + TargetClusterIndice];

                ClusterIDSet[(NNClusterIndice - 1) * EfSearch + ClusterIDSetIndice[NNClusterIndice -1]] = TargetClusterID;
                //ClusterIDSet[NNClusterIndice -1][ClusterIDSetIndice[NNClusterIndice -1]] = TargetClusterID;
                ClusterIDSetIndice[NNClusterIndice -1] ++;
            }
            //Trecorder.recordTimeConsumption2();

            //Trecorder.reset();

/*
            for (size_t i = 0; i < EfSearch - 1; i++){
                std::sort(ClusterIDSet + i * EfSearch, ClusterIDSet + i * EfSearch + ClusterIDSetIndice[i]);
            }
*/
            for (size_t i = 0; i < EfSearch - 1; i++){
                if (ClusterIDSetIndice[i] == 0){continue;}

                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i + 1];

                CompStartIndice = NNClusterID == 0 ? 0 : NeighborListVecIndice[NNClusterID - 1];
                CompIndice = NeighborListVecIndice[NNClusterID] - CompStartIndice;
                CompSize1 = MaxVecListSize;
                codec.decodeArray(NeighborListVec + CompStartIndice, CompIndice, DeCompVecID, CompSize1);

                CompStartIndice = NNClusterID == 0 ? 0 : NeighborListNumTarIndice[NNClusterID - 1];
                CompIndice = NeighborListNumTarIndice[NNClusterID] - CompStartIndice;
                CompSize2 = MaxVecListSize;
                codec.decodeArray(NeighborListNumTar + CompStartIndice, CompIndice, DeCompNumTar, CompSize2);

                CompStartIndice = NNClusterID == 0 ? 0 : NeighborListTarIDIndice[NNClusterID - 1];
                CompIndice = NeighborListTarIDIndice[NNClusterID] - CompStartIndice;
                CompSize3 = MaxTarListSize;
                codec.decodeArray(NeighborListTarID + CompStartIndice, CompIndice, DeCompTarID, CompSize3);

                /*
                for (size_t temp = 0; temp < CompSize1; temp++){
                    std::cout << DeCompVecID[temp] << " ";
                }
                std::cout << "\n";
                for (size_t temp = 0; temp < CompSize2; temp++){
                    std::cout << DeCompNumTar[temp] << " ";
                }
                std::cout << "\n";
                for (size_t temp = 0; temp < CompSize3; temp++){
                    std::cout << DeCompTarID[temp] << " ";
                }
                std::cout << "\n\n\n";
                */

                assert(CompSize1 == CompSize2);
                uint32_t Indice = 0;
                for (size_t temp = 0; temp < CompSize1; temp++){
                    //bool CheckFlag = Intersect(DeCompTarID + Indice, DeCompTarID + Indice + DeCompNumTar[temp],  ClusterIDSet[i], ClusterIDSet[i] + ClusterIDSetIndice[i]);
                    //bool CheckFlag = Intersect(DeCompTarID + Indice, DeCompTarID + Indice + DeCompNumTar[temp],  ClusterIDSet + i * EfSearch, ClusterIDSet + i * EfSearch + ClusterIDSetIndice[i]);
                    bool CheckFlag = Overlap(DeCompTarID + Indice, ClusterIDSet + i * EfSearch, DeCompNumTar[temp], ClusterIDSetIndice[i]);

/*
                    for (size_t temp0 = 0; temp0 < DeCompNumTar[temp]; temp0++){
                        std::cout << DeCompTarID[Indice + temp0] << " ";
                    }
                    std::cout << "/";
                    for (size_t temp0 = 0; temp0 < ClusterIDSetIndice[i]; temp0 ++){
                        std::cout << ClusterIDSet[i * EfSearch + temp0] << " ";
                    }
                    std::cout << CheckFlag << " " << CheckFlag_ << " |\n";
*/
                    //assert(CheckFlag == CheckFlag_);

                    if (CheckFlag){
                        VisitedItem ++;
                        uint32_t VectorID = DeCompVecID[temp];
                        float Dist = faiss::fvec_L2sqr(BaseVectors + VectorID * Dimension, Query + QueryIdx * Dimension, Dimension);
                        if (Dist < QueryDists[QueryIdx * RecallK]){
                            faiss::maxheap_pop(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK);
                            faiss::maxheap_push(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK, Dist, VectorID);
                        }
                    }
                    Indice += DeCompNumTar[temp];
                }
                assert(Indice == CompSize3);
            }
            //Trecorder.recordTimeConsumption3();
        }

        if (!UseList){
            for (uint32_t i = 1; i < EfSearch; i++){
                if (VisitedItem > MaxItem){break;}

                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i];

                codec.decodeArray(BaseCompIds[NNClusterID].data(), BaseCompIds[NNClusterID].size(), DeCompBaseIDs, CompSize);

                // The NNCluster is partly searched in the NL, search the remaining part
                for (size_t j = 0; j < CompSize; j++){
                    uint32_t VectorID = DeCompBaseIDs[j];
                    VisitedItem ++;
                    float Dist = faiss::fvec_L2sqr(BaseVectors + VectorID * Dimension, Query + QueryIdx * Dimension, Dimension);
                    if (Dist < QueryDists[QueryIdx * RecallK]){
                        faiss::maxheap_pop(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK);
                        faiss::maxheap_push(RecallK, QueryDists + QueryIdx * RecallK, QueryIds + QueryIdx * RecallK, Dist, VectorID);
                    }
                }

            }
        }
        //Trecorder.recordTimeConsumption3();
        SumVisitedItem += VisitedItem;
    }
    return SumVisitedItem;
}

uint32_t SearchMulti(time_recorder & Trecorder, size_t nq, size_t K, size_t QuantBatchSize, float * Query, int64_t * QueryIds, float * QueryDists, size_t EfSearch, size_t MaxItem, size_t EfNeighborList, bool UseList, bool UseQuantize, bool NeighborListOnly,
faiss::ProductQuantizer * PQ, faiss::LinearTransform * OPQ, hnswlib::HierarchicalNSW * CenGraph, float * CentroidsDist, float * QCDist, uint32_t * QCID, float * PQTable,
float * CNorms, float * BaseNorms, uint8_t * BaseCodes, std::vector<std::vector<uint32_t>> & BaseIds, std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::pair<std::vector<uint32_t>, std::vector<float>>>> & NewNeighborList, 
float * NeighborListAlpha, uint32_t * NeighborListAlphaTar, uint32_t * NeighborListAlphaIndice, uint32_t * NeighborListVector, uint32_t * NeighborListVectorIndice
){
/*
    std::vector<float> BaseVectors(nb * Dimension);
    std::ifstream BaseInput(PathBase, std::ios::binary);
    readXvecFvec<float>(BaseInput, BaseVectors.data(), Dimension, nb);
*/

    uint32_t SumVisitedItem = 0;
    //Trecorder.reset();

    if (UseOPQ){
        OPQ->apply(nq, Query);
    }
    for(size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto ResultQueue = CenGraph->searchBaseLayer(Query + QueryIdx * Dimension, EfSearch);
        for (size_t i = 0; i < EfSearch; i++){
            //std::cout << ResultQueue.top().second << " ";
            QCDist[QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().first;
            QCID  [QueryIdx * EfSearch + EfSearch - i - 1] = ResultQueue.top().second;
            ResultQueue.pop();
        }
        //std::cout << "\n";
    }
    //Trecorder.recordTimeConsumption1();

    //Trecorder.reset();
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        PQ->compute_inner_prod_table(Query + QueryIdx * Dimension, PQTable + QueryIdx * PQ->ksub * PQ->M);
        faiss::maxheap_heapify(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
    }
    //Trecorder.recordTimeConsumption2();


    //Trecorder.reset();
    //std::cout << nq << "\n";
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){

        //Trecorder.reset();
        //std::cout << " Search the neighbor list 0\n"; 
        size_t VisitedItem = 0;
        uint32_t BaseClusterID = QCID[QueryIdx * EfSearch];
        // Search the vectors in the base target cluster
        for (size_t j = 0; j < BaseIds[BaseClusterID].size(); j++){
            uint32_t VectorID = BaseIds[BaseClusterID][j]; 
            float VNorm = BaseNorms[VectorID]; 
            float ProdDist = 0;
            for (size_t k = 0; k < PQ->code_size; k++){
                ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodes[VectorID * PQ->code_size + k]];
            }
            float Dist = QCDist[QueryIdx * EfSearch] - CNorms[BaseClusterID] + VNorm - 2 * ProdDist;
            //Dist = faiss::fvec_L2sqr(BaseVectors.data() + VectorID * Dimension, Query + QueryIdx * Dimension, Dimension);
            
            if (Dist < QueryDists[QueryIdx * K]){
                faiss::maxheap_pop(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
                faiss::maxheap_push(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K, Dist, VectorID);
            }
        }
        VisitedItem += BaseIds[BaseClusterID].size();
        //std::cout << " Visited Items: " << VisitedItem << "\n"; 
/*
        if (VisitedItem > MaxItem){
            SumVisitedItem += VisitedItem;
            continue;
        }
*/
        //Trecorder.recordTimeConsumption1();

        //std::cout << " Search the neighbor list 1\n"; 
        std::vector<std::unordered_set<uint32_t>> VisitedVecSets(EfSearch);
        if (UseList){

            //Trecorder.reset();
            assert(EfNeighborList <= EfSearch * (EfSearch - 1));

            int64_t NL2SearchID[EfNeighborList];
            float NL2SearchDist[EfNeighborList];

            faiss::maxheap_heapify(EfNeighborList, NL2SearchDist, NL2SearchID);

            // Compute the distance between the vector and neighbor lists
            for (size_t i = 1; i < EfSearch; i++){
                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i];

                for (uint32_t j = 0; j < EfSearch; j++){
                    if (i == j) continue;
                    uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + j];

                    uint32_t EndIndice = NeighborListAlphaIndice[NNClusterID];
                    uint32_t StartIndice = NNClusterID == 0 ? 0 : NeighborListAlphaIndice[NNClusterID - 1];

                    for (size_t temp = StartIndice; temp < EndIndice; temp++){
                        if (NeighborListAlphaTar[temp] == TargetClusterID){
                            float Alpha = NeighborListAlpha[temp];
                            float QueryNLDist = FetchQueryNLDist(QCDist[QueryIdx * EfSearch + j], QCDist[QueryIdx * EfSearch + i], CentroidsDist[TargetClusterID * nc + NNClusterID], Alpha);
                            int64_t NLIndice = i * EfSearch + j;
                            if (QueryNLDist < NL2SearchDist[0]){
                                faiss::maxheap_pop(EfNeighborList, NL2SearchDist, NL2SearchID);
                                faiss::maxheap_push(EfNeighborList,NL2SearchDist, NL2SearchID, QueryNLDist, NLIndice);
                            }
                            break;
                        }
                    }
                }
            }

/*
            for (size_t i = 0; i < EfNeighborList; i++){
                std::cout << NL2SearchID[i] << " ";
            }
            std::cout << "\n";
*/

            int64_t ClusterIDSet[EfSearch - 1][EfSearch]; 
            for (size_t i = 0; i < EfSearch -1; i++){
                for (size_t j = 0; j < EfSearch; j++){
                    ClusterIDSet[i][j] = -1;
                }
            }

            for (size_t i = 0; i < EfNeighborList; i++){
                if (NL2SearchID[i] < 0){continue;}
                uint32_t NNClusterIndice = NL2SearchID[i] / EfSearch;
                uint32_t TargetClusterIndice = NL2SearchID[i] % EfSearch;
                uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + TargetClusterIndice];

                for (size_t j = 0; j < EfSearch; j++){
                    if (ClusterIDSet[NNClusterIndice - 1][j] == -1){
                        ClusterIDSet[NNClusterIndice - 1][j] = TargetClusterID;
                        break;
                    }
                }
            }

            //Trecorder.recordTimeConsumption2();
            // Search the vectors in these NNClusters
            //Trecorder.reset();
            for (size_t i = 0; i < EfSearch - 1; i++){

                uint32_t TarSize = 0;
                for (; TarSize < EfSearch; TarSize++){
                    if (ClusterIDSet[i][TarSize] == -1){
                        break;
                    }
                }
                if (TarSize == 0){continue;}

                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i + 1];
                uint32_t EndIndice = NeighborListVectorIndice[NNClusterID];
                uint32_t VecStartIndice = NNClusterID == 0? 0 : NeighborListVectorIndice[NNClusterID - 1];
                float CDist = QCDist[QueryIdx * EfSearch + i + 1] - CNorms[NNClusterID];

                //Trecorder.reset();

                for (; VecStartIndice < EndIndice; VecStartIndice = VecStartIndice + 1 + 1 + NeighborListVector[VecStartIndice + 1])
                {
                    
                    bool CheckFlag = false;
                    for (size_t temp0 = 0; temp0 < NeighborListVector[VecStartIndice + 1]; temp0++){
                        for (size_t temp1 = 0; temp1 < TarSize; temp1++){
                            if (NeighborListVector[VecStartIndice + 1 + 1 + temp0] == ClusterIDSet[i][temp1]){
                                CheckFlag = true;
                                break;
                            }
                        }
                        if (CheckFlag){break;}
                    }

                    if (CheckFlag){
                        //std::cout << NeighborListVector[VecStartIndice] << " ";
                        VisitedItem ++;
                        uint32_t VectorID = NeighborListVector[VecStartIndice];
                        float VNorm = BaseNorms[VectorID];
                        float ProdDist = 0;
                        for (size_t k = 0; k < PQ->code_size; k++){
                            ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodes[VectorID * PQ->code_size + k]];
                        }
                        float Dist = CDist + VNorm - 2 * ProdDist;
                        //Dist = faiss::fvec_L2sqr(BaseVectors.data() + VectorID * Dimension, Query + QueryIdx * Dimension, Dimension);
                        if (Dist < QueryDists[QueryIdx * K]){
                            faiss::maxheap_pop(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
                            faiss::maxheap_push(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K, Dist, VectorID);
                        }
                    }
                }
                

                //std::cout << "\n\n";
            }
            Trecorder.recordTimeConsumption3();
        }
        
        //std::cout << "\n";

        if (!UseList || !NeighborListOnly){
            for (uint32_t i = 1; i < EfSearch; i++){
                if (VisitedItem > MaxItem){break;}

                uint32_t NNClusterID = QCID[QueryIdx * EfSearch + i];

                size_t ClusterSize = BaseIds[NNClusterID].size();
                float CDist = QCDist[QueryIdx * EfSearch + i] - CNorms[NNClusterID];

                // The NNCluster is partly searched in the NL, search the remaining part
                for (size_t j = 0; j < ClusterSize; j++){
                    //if ((VisitedVecSets[i].count(BaseIds[NNClusterID][j]) == 0)){
                        //{VisitedVecSets[i].insert(BaseIds[NNClusterID][j]);}

                    if (!UseList || (VisitedVecSets[i].count(BaseIds[NNClusterID][j]) == 0)){
                        if (UseList){VisitedVecSets[i].insert(BaseIds[NNClusterID][j]);}

                        uint32_t VectorID = BaseIds[NNClusterID][j];
                        float VNorm = BaseNorms[VectorID];
                        VisitedItem ++;

                        float ProdDist = 0;
                        for (size_t k = 0; k < PQ->code_size; k++){
                            ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodes[VectorID * PQ->code_size + k]];
                        }
                        float Dist = CDist + VNorm - 2 * ProdDist;
                        //float Dist = faiss::fvec_L2sqr(BaseVectors.data() + VectorID * Dimension, Query + QueryIdx * Dimension, Dimension);
                        if (Dist < QueryDists[QueryIdx * K]){
                            faiss::maxheap_pop(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
                            faiss::maxheap_push(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K, Dist, VectorID);
                        }
                    }
                }
            }
        }
        SumVisitedItem += VisitedItem;
        //Trecorder.recordTimeConsumption3();
    }

    
    //std::cout << "The number of NLConflict is: " << float(SumNLConflict) / nq << "\n";

    return SumVisitedItem;
}


void TrainQuantizer(time_recorder & Trecorder, size_t PQTrainSize, std::string PathLearn, std::string PathPQ, std::string PathOPQ,
hnswlib::HierarchicalNSW * CenGraph){
    Trecorder.reset();

    if (!Retrain && exists(PathPQ) && ((UseOPQ && exists(PathOPQ)) || (!UseOPQ)) ){
        std::cout << "PQ and OPQ Quantizer Exist, Load PQ from " << PathPQ <<  std::endl;
        return;
    }

    std::cout << "Training PQ and OPQ quantizer" << std::endl;
    std::vector<float> PQTrainSet(PQTrainSize * Dimension);
    std::cout << "Sampling " << PQTrainSize << " from " << nt << " for PQ training" << std::endl;
    SampleSet<DataType>(PQTrainSet.data(), PathLearn, PQTrainSize, nt, Dimension);

    std::vector<uint32_t> DataID(PQTrainSize);

#pragma omp parallel for
    for (size_t i = 0; i < PQTrainSize; i++){
        auto Result = CenGraph->searchKnn(PQTrainSet.data() + i * Dimension, 1);
        DataID[i] = Result.top().second;
    }
    std::vector<float> Residual(PQTrainSize * Dimension);

#pragma omp parallel for
    for (size_t i = 0; i < PQTrainSize; i++){
        faiss::fvec_madd(Dimension, PQTrainSet.data() + i * Dimension, -1.0, CenGraph->getDataByInternalId(DataID[i]), Residual.data() + i * Dimension);
    }
    if (UseOPQ){
        faiss::ProductQuantizer * PQTrain  = new faiss::ProductQuantizer(Dimension, M_PQ, CodeBits);
        faiss::OPQMatrix * OPQTrain = new faiss::OPQMatrix(Dimension, M_PQ);
        OPQTrain->verbose = false;
        OPQTrain->niter = 50;
        OPQTrain->pq = PQTrain;
        OPQTrain->train(PQTrainSize, Residual.data());

        std::vector<float> CopyDataset(PQTrainSize * Dimension);
        memcpy(CopyDataset.data(), Residual.data(), PQTrainSize * Dimension * sizeof(float));
        OPQTrain->apply_noalloc(PQTrainSize, CopyDataset.data(), Residual.data());


        faiss::write_VectorTransform(OPQTrain, PathOPQ.c_str());

        
        faiss::write_ProductQuantizer(PQTrain, PathPQ.c_str());
        Trecorder.print_time_usage("OPQ Trainining Completed");
        free(OPQTrain); free(OPQTrain);
    }
    else{
        faiss::ProductQuantizer * PQTrain  = new faiss::ProductQuantizer(Dimension, M_PQ, CodeBits);

        PQTrain->verbose = true;
        PQTrain->train(PQTrainSize, Residual.data());
        faiss::write_ProductQuantizer(PQTrain, PathPQ.c_str());
        free(PQTrain);
        std::cout << "Save PQ quantizer to: " << PathPQ << std::endl;
        Trecorder.print_time_usage("Train PQ Quantizer");
    }
    return;
}

void QuantizeVector(size_t N, hnswlib::HierarchicalNSW * CenGraph, faiss::ProductQuantizer * PQ, faiss::LinearTransform * OPQ, float * BaseData, uint32_t * BaseID, uint8_t * BaseCode, float * BaseNorm, float * OPQCentroids){
    std::vector<float> Residual(N * Dimension);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        faiss::fvec_madd(Dimension, BaseData + i * Dimension, -1.0, CenGraph->getDataByInternalId(BaseID[i]), Residual.data() + i * Dimension);
    }

    if (UseOPQ){
        std::vector<float> CopyDataset(N * Dimension);
        memcpy(CopyDataset.data(), Residual.data(), N * Dimension * sizeof(float));
        OPQ->apply_noalloc(N, CopyDataset.data(), Residual.data());
    }

    std::cout << "Computing the vector codes\n";
    PQ->compute_codes(Residual.data(), BaseCode, N);

    for (size_t i = 0; i < 100; i++){
        std::cout << int(BaseCode[i]) << " ";
    }

    std::cout << "Decoding the vector codes\n";
    std::vector<float> RecoveredResidual(N * Dimension);
    PQ->decode(BaseCode, RecoveredResidual.data(), N);
    
    std::cout << "The quanization error infomation: \n";
    std::vector<float> RecoveredBase(N * Dimension);

    // The loss in PQ quantization
    uint32_t ErrorSample = 1000;
    std::cout << "\nThe base vector residual norm: " << faiss::fvec_norm_L2sqr(Residual.data(), Dimension * ErrorSample) / ErrorSample << std::endl;
    std::cout << "The PQ quantization error of residual: " << faiss::fvec_L2sqr(RecoveredResidual.data(), Residual.data(), Dimension * ErrorSample) / ErrorSample << "\n\n";


    if (UseOPQ){
#pragma omp parallel for
        for (size_t i = 0; i < N; i++){
            faiss::fvec_add(Dimension, RecoveredResidual.data() + i * Dimension, OPQCentroids + BaseID[i] * Dimension, RecoveredBase.data() + i * Dimension);
        }
    }
    else{
#pragma omp parallel for
        for (size_t i = 0; i < N; i++){
            faiss::fvec_add(Dimension, RecoveredResidual.data() + i * Dimension, CenGraph->getDataByInternalId(BaseID[i]), RecoveredBase.data() + i * Dimension);
        }
    }
    faiss::fvec_norms_L2sqr(BaseNorm, RecoveredBase.data(), Dimension, N);
}


void QuantizeBaseset(size_t NumBatch, size_t BatchSize, std::string PathBase, std::string PathBaseIDSeq, std::string PathBaseCode, std::string PathBaseNorm, std::string PathOPQCentroids,
std::vector<uint8_t> & BaseCodes, std::vector<float> & BaseNorms,
hnswlib::HierarchicalNSW * CenGraph, faiss::ProductQuantizer * PQ, faiss::LinearTransform * OPQ
){
    if (!Retrain && exists(PathBaseCode) && exists(PathBaseNorm) && ((UseOPQ && exists(PathOPQCentroids)) || (!UseOPQ)) ){
        std::cout << "Loading Base Vector PQ Codes" << std::endl;
        std::ifstream BaseCodeInput(PathBaseCode, std::ios::binary);
        std::ifstream BaseNormInput(PathBaseNorm, std::ios::binary);

        std::cout << "Loading the code \n";
        BaseCodes.resize(nb * PQ->code_size);
        BaseCodeInput.read((char * ) BaseCodes.data(), nb * PQ->code_size * sizeof(uint8_t));
        BaseCodeInput.close();

        std::cout << "Loading the base norms \n";
        BaseNorms.resize(nb);
        BaseNormInput.read((char *) BaseNorms.data(), nb * sizeof(float));
        BaseNormInput.close();

        if (UseOPQ){
            std::ifstream OPQCentroidsInput(PathOPQCentroids, std::ios::binary);
            uint32_t Dim;
            for (size_t i = 0; i < nc; i++){
                OPQCentroidsInput.read((char * ) & Dim, sizeof(uint32_t)); assert(Dim == Dimension);
                OPQCentroidsInput.read((char *) CenGraph->getDataByInternalId(i), Dimension * sizeof(float));
            }
            OPQCentroidsInput.close();
        }
        std::cout << "Load Vector Codes Completed" << std::endl;
        return;
    }

    time_recorder Trecorder = time_recorder();
    BaseCodes.resize(nb * PQ->code_size);
    BaseNorms.resize(nb);

    std::vector<float> OPQCentroids;
    if (UseOPQ){
        OPQCentroids.resize(nc * Dimension);
        std::cout << "Appling OPQ Rotation" << std::endl;
        for (size_t i = 0; i < nc; i++){
            OPQ->apply_noalloc(1, CenGraph->getDataByInternalId(i), OPQCentroids.data() + i * Dimension);
        }
    }

    std::ifstream BaseInput(PathBase, std::ios::binary);
    std::vector<float> BaseSet(Dimension *BatchSize);
    std::ifstream BaseIDSeqInput(PathBaseIDSeq, std::ios::binary);
    std::vector<uint32_t> BaseSetID(BatchSize);
    std::cout << "Quantizing the base vectors " << std::endl;

    Trecorder.reset();
    for (size_t i = 0; i <  NumBatch; i++){
        Trecorder.print_time_usage("Quantize BaseSet Completed " + std::to_string(i + 1) + " batch in " + std::to_string(NumBatch) + " Batches");
        readXvecFvec<DataType>(BaseInput, BaseSet.data(), Dimension, BatchSize, true, true);
        BaseIDSeqInput.read((char *)BaseSetID.data(), BatchSize * sizeof(uint32_t));
        Trecorder.print_time_usage("Quantizing the vectors");
        QuantizeVector(BatchSize, CenGraph, PQ, OPQ, BaseSet.data(), BaseSetID.data(), BaseCodes.data() + i * BatchSize * PQ->code_size, BaseNorms.data() + i * BatchSize, OPQCentroids.data());
    }

    std::ofstream BaseCodeOutput(PathBaseCode, std::ios::binary);
    BaseCodeOutput.write((char * )BaseCodes.data(), nb * PQ->code_size * sizeof(uint8_t));
    BaseCodeOutput.close();
    std::ofstream BaseNormOutput(PathBaseNorm, std::ios::binary);
    BaseNormOutput.write((char *)BaseNorms.data(), nb * sizeof(float));
    BaseNormOutput.close();

    if (UseOPQ){
        std::ofstream OPQCentroidOutput(PathOPQCentroids, std::ios::binary);
        for (size_t i = 0; i < nc; i++){
            OPQCentroidOutput.write((char *) & Dimension, sizeof(uint32_t));
            OPQCentroidOutput.write((char *) (OPQCentroids.data() + i * Dimension), Dimension * sizeof(float));
            memcpy(CenGraph->getDataByInternalId(i), OPQCentroids.data() + i * Dimension, Dimension * sizeof(float));
        }
        OPQCentroidOutput.close();
    }
    return;
}

void ComputeNN(
size_t Graph_num_batch, size_t SearchK, size_t DatasetSize,

time_recorder & TRecorder, memory_recorder & MRecorder, 

std::string NNDatasetName, std::string PathDataset, std::string PathDatasetNN, std::string PathSubGraphFolder
 ){
    // Compute the nearest neighbors of vectors in a given set (include the vectors themselves) 

    if (!exists(PathDatasetNN)){

        // Prepare the subgraph for DatasetNN search
        PrepareFolder(PathSubGraphFolder.c_str());

        std::cout << "Build the subgraph for searching NN of target dataset vectors with num_batch: " << Graph_num_batch << "\n";
        // Build multiple graph for search NN 
        std::ifstream DatasetInput(PathDataset, std::ios::binary);

        size_t Graph_batch_size = DatasetSize / Graph_num_batch;

        for (size_t i = 0; i < Graph_num_batch; i++){
            std::string PathSubGraphInfo = PathSubGraphFolder + NNDatasetName + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + ".info";
            std::string PathSubGraphEdge = PathSubGraphFolder + NNDatasetName + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(i) + ".edge";

            if (!exists(PathSubGraphInfo)){
                std::vector<float> DatasetBatch(Graph_batch_size * Dimension);
                DatasetInput.seekg(i * Graph_batch_size * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
                readXvecFvec<DataType>(DatasetInput, DatasetBatch.data(), Dimension, Graph_batch_size, true, true);

                hnswlib::HierarchicalNSW * SubGraph = new hnswlib::HierarchicalNSW(Dimension, Graph_batch_size);
                for (size_t j = 0; j < Graph_batch_size; j++){
                    SubGraph->addPoint(DatasetBatch.data() + j * Dimension);
                }

                SubGraph->SaveInfo(PathSubGraphInfo);
                SubGraph->SaveEdges(PathSubGraphEdge);
                delete(SubGraph);
            }
        }

        TRecorder.print_time_usage("Build subgraph for dataset");

        std::cout << "Search the NNs of target dataset vectors\n";
        for (size_t i = 0; i < Graph_num_batch; i++){
            std::string PathDatasetNNSub = PathDatasetNN + "_Sub_" + std::to_string(i) + "_" + std::to_string(Graph_num_batch);

            std::vector<float> DatasetBatch(Graph_batch_size * Dimension);
            DatasetInput.seekg(i * Graph_batch_size * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
            readXvecFvec<DataType> (DatasetInput, DatasetBatch.data(), Dimension, Graph_batch_size, true, true);

            // Search the dataset NN for vectors
            if (!exists(PathDatasetNNSub)){

                std::vector<std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, cmp>> NNQueueList;
                NNQueueList.resize(Graph_batch_size);
                for (size_t j = 0; j < Graph_num_batch; j++){
                    TRecorder.print_time_usage("Search the " + std::to_string(i + 1) + " th batch " + std::to_string(j + 1) + " th graph in " + std::to_string(Graph_num_batch) + " batches");
                    MRecorder.print_memory_usage("");
                    std::string PathSubGraphInfo = PathSubGraphFolder + NNDatasetName + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(j) + ".info";
                    std::string PathSubGraphEdge = PathSubGraphFolder + NNDatasetName + "SubGraph_" + std::to_string(Graph_num_batch) + "_" + std::to_string(j) + ".edge";

                    hnswlib::HierarchicalNSW * SubGraph;
                    DatasetInput.seekg(j * Graph_batch_size * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
                    SubGraph = new hnswlib::HierarchicalNSW(PathSubGraphInfo, PathSubGraphEdge, DatasetInput);

#pragma omp parallel for
                    for (size_t k = 0; k < Graph_batch_size; k++){
                        auto result = SubGraph->searchKnn(DatasetBatch.data() + k * Dimension, SearchK);

                        for (size_t temp = 0; temp < SearchK; temp++){
                            if (NNQueueList[k].size() < SearchK){
                                NNQueueList[k].emplace(std::make_pair(j * Graph_batch_size + result.top().second, result.top().first));
                            }
                            else if (result.top().first < NNQueueList[k].top().second){
                                NNQueueList[k].pop();
                                NNQueueList[k].emplace(std::make_pair(j * Graph_batch_size + result.top().second, result.top().first));
                            }
                            result.pop();
                        }
                    }
                    std::cout << "Completed the retrieval \n";
                    delete(SubGraph);
                }
                std::ofstream DatasetNNoutputSub(PathDatasetNNSub, std::ios::binary);
                for (size_t j = 0; j < Graph_batch_size; j++){
                    std::vector<uint32_t> VectorNN(SearchK);
                    for (size_t k = 0; k < SearchK; k++){
                        VectorNN[SearchK - k -1] = NNQueueList[j].top().first;
                        NNQueueList[j].pop();
                    }
                    DatasetNNoutputSub.write((char *) VectorNN.data(), SearchK * sizeof(uint32_t));
                }
                DatasetNNoutputSub.close();

                MRecorder.print_memory_usage("");
                TRecorder.print_time_usage("Completed NN search on a batch");
            }
        }

        MRecorder.print_memory_usage("");
        TRecorder.print_time_usage("Completed NN search and Assignment");

        if (!exists(PathDatasetNN)){
            std::ofstream DatasetNNOutput(PathDatasetNN, std::ios::binary);
            for (size_t i = 0 ; i < Graph_num_batch; i++){
                std::string PathDatasetNNSub = PathDatasetNN + "_Sub_" + std::to_string(i) + "_" + std::to_string(Graph_num_batch);
                assert(exists(PathDatasetNNSub));
                std::ifstream DatasetNNInputSub(PathDatasetNNSub, std::ios::binary);
                std::vector<uint32_t> DatasetNNBatch(Graph_batch_size * Dimension);
                DatasetNNInputSub.read((char *) DatasetNNBatch.data(), Graph_batch_size * SearchK * sizeof(uint32_t));
                DatasetNNOutput.write((char *) DatasetNNBatch.data(),  Graph_batch_size * SearchK * sizeof(uint32_t));
                DatasetNNInputSub.close();
            }
            DatasetNNOutput.close();
        }
        DatasetInput.close();
    }
}


void BuildNeighborList(
    size_t SearchK, size_t Num_batch, size_t DatasetSize, size_t NeighborNum, size_t NLTargetK, bool BaseConflict, bool DistPrune, bool UseQuantize, size_t QuantBatchSize,
    hnswlib::HierarchicalNSW * CenGraph, time_recorder & TRecorder, 
    std::string PathDatasetNN, std::string PathNeighborList, std::string PathNeighborListInfo, std::string PathNNDataset, std::string PathDatasetNeighborID, 
    std::string PathDatasetNeighborDist, std::string PathBaseNeighborID, std::string PathBaseNeighborDist
){
    // Check the boundary conflict and genrate the neighbor list for search
    // When to search the points not in the neighbor list? (After searching all vectors in the neighbor list (except for those with lager dist than threshold))

    // Index with distance to the boundary

    if (!exists(PathNeighborList)){
        std::cout << "Constructing the neighbor list index and save it to: " << PathNeighborList << "\n";
        size_t Batch_size = DatasetSize / Num_batch;

        // Get the neighbor ID and the dist of training vectors
        if (!exists(PathDatasetNeighborID) || !exists(PathDatasetNeighborDist)){
            std::ofstream DatasetNeighborIDOutput(PathDatasetNeighborID, std::ios::binary);
            std::ofstream DatasetNeighborDistOutput(PathDatasetNeighborDist, std::ios::binary);
            std::ifstream DatasetInput(PathNNDataset, std::ios::binary);

            std::vector<float> DatasetBatch(Batch_size * Dimension);
            std::vector<uint32_t> DatasetIDBatch(Batch_size * NeighborNum);
            std::vector<float> DatasetDistBatch(Batch_size * NeighborNum);

            for (size_t i = 0; i < Num_batch; i++){

                readXvecFvec<DataType>(DatasetInput, DatasetBatch.data(), Dimension, Batch_size, true, false);
#pragma omp parallel for
                for (size_t j = 0; j < Batch_size; j++){

                    auto result =  CenGraph->searchKnn(DatasetBatch.data() + j * Dimension, NeighborNum);
                    for (size_t temp = 0; temp < NeighborNum; temp++){
                        DatasetIDBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().second;
                        DatasetDistBatch[j * NeighborNum + NeighborNum - temp - 1] = result.top().first;
                        result.pop();
                    }
                }

                DatasetNeighborIDOutput.write((char *) DatasetIDBatch.data(), Batch_size * NeighborNum * sizeof(uint32_t));
                DatasetNeighborDistOutput.write((char *) DatasetDistBatch.data(), Batch_size * NeighborNum * sizeof(float));
                std::cout << "Completed batch assignment: " << i << " / " << Num_batch << "\n";
            }
        }

        assert(exists(PathDatasetNN));
        std::ifstream DatasetNNInput(PathDatasetNN, std::ios::binary);
        std::vector<uint32_t> DatasetNNSeq(DatasetSize * SearchK);

        DatasetNNInput.read((char *) DatasetNNSeq.data(), DatasetSize * SearchK * sizeof(uint32_t));
        DatasetNNInput.close();

        std::cout << "The example base NN of vectors\n";
        for (size_t i = 0; i < 10; i++){
            for (size_t j = 0; j < SearchK; j++){
                std::cout << DatasetNNSeq[i * SearchK + j] << " ";
            }
            std::cout << "\n";
        }

        // We use the neighborhood structure for neighbor list, try other methods: i.e. distance boundary from small trainset / distance ratio between two clusters
        // Struct: NNClusterID, TargetClusterID, VectorID
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::set<uint32_t>>> BoundaryConflictMap;

        // Load the assignment info of test dataset
        assert(exists(PathDatasetNeighborID));
        std::ifstream NeighborIDInput(PathDatasetNeighborID, std::ios::binary);
        std::vector<uint32_t> VectorAssignment(DatasetSize);
        std::vector<uint32_t> NeighborIDBatch(Batch_size * NeighborNum);

        for (size_t i = 0; i < Num_batch; i++){
            NeighborIDInput.read((char *) NeighborIDBatch.data(), Batch_size * NeighborNum * sizeof(uint32_t));
            for (size_t j = 0; j < Batch_size; j++){
                VectorAssignment[i * Batch_size + j] = NeighborIDBatch[j * NeighborNum];
            }
        }
        NeighborIDInput.close();
        for (size_t i = 0; i < 100; i++){
            std::cout << VectorAssignment[i] << " ";
        }
        std::cout << "\n";

        size_t SumConflicts = 0;

        for (uint32_t i = 0; i < Num_batch; i++){
            for (uint32_t j = 0; j < Batch_size; j++){

                uint32_t TargetVectorID = i * Batch_size + j;
                uint32_t TargetClusterID = VectorAssignment[TargetVectorID];

                for (size_t k = 0; k < NLTargetK; k++){
                    uint32_t NNID = DatasetNNSeq[(i * Batch_size + j) * SearchK + k];
                    uint32_t NNClusterID = VectorAssignment[NNID];

                    if (TargetClusterID != NNClusterID){
                        SumConflicts ++;
                        // Boundary conflict, include this pair to neighbor conflict queue

                        if (BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end()){
                            // There is already boundary conflict recorded on this boundary

                            if (BoundaryConflictMap[NNClusterID].find(TargetClusterID) != BoundaryConflictMap[NNClusterID].end()){
                                BoundaryConflictMap[NNClusterID][TargetClusterID].insert(NNID);
                            }
                            else{
                                BoundaryConflictMap[NNClusterID][TargetClusterID] = std::set<uint32_t>({NNID});
                            }
                        }
                        else{
                            BoundaryConflictMap[NNClusterID] = std::unordered_map<uint32_t, std::set<uint32_t>> ({{TargetClusterID, std::set<uint32_t>({NNID})}});
                        }

                        // Todo: should we use the reverse indice, i.e. the NN structure in the neighbor list?
/*

                        if (BoundaryConflictMap.find(TargetClusterID) != BoundaryConflictMap.end()){
                            if (BoundaryConflictMap[TargetClusterID].find(NNClusterID) != BoundaryConflictMap[TargetClusterID].end()){
                                BoundaryConflictMap[TargetClusterID][NNClusterID].insert(TargetVectorID);
                            }
                            else{
                                BoundaryConflictMap[TargetClusterID][NNClusterID] = std::set<uint32_t>({TargetVectorID});
                            }
                        }
                        else{
                            BoundaryConflictMap[TargetClusterID] = std::unordered_map<uint32_t, std::set<uint32_t>> ({{NNClusterID, std::set<uint32_t>({TargetVectorID})}});
                        }

*/
                    }
                }
            }
        }

        std::cout << "Total number of conflicts " << SumConflicts << " in " << DatasetSize << " vectors and " << nc << " clusters, SearchK: " << SearchK << "\n";
        TRecorder.print_time_usage("Identify all boundary conflicts");

        // Scan the base set and get the distance boundary (NN vector to the boundary) of the neighbor list
        // Two choices: Use the average boundary distance / Use the average boundary distance and the average centroid distance

        uint32_t NLVectorSize = 0;
        // Calcultae the distance that is related to the cluster and boundary
        // Struct: targetclusterID, NNClusterID, SumBoundDist, SumTarCenDist, SumNNCenDist. SumDistRatio
        std::map<std::pair<uint32_t, uint32_t>, std::vector<float>> BoundaryConflictCentroidMap;

        std::map<std::pair<uint32_t, uint32_t>, std::pair<std::vector<uint32_t>, std::vector<float>>> BoundaryConflictAlphaMap;

        std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::tuple<float, float, float, float>>> BoundaryConflictDistMap;

        std::ifstream DatasetInput(PathNNDataset, std::ios::binary);
        std::ifstream DatasetNeighborIDInput(PathDatasetNeighborID, std::ios::binary);
        std::ifstream DatasetNeighborDistInput(PathDatasetNeighborDist, std::ios::binary);

        std::vector<float> DatasetBatch(Batch_size * Dimension);
        std::vector<uint32_t> DatasetNeighborIDBatch(Batch_size * NeighborNum);
        std::vector<float> DatasetNeighborDistBatch(Batch_size * NeighborNum);

        for (uint32_t i = 0; i < Num_batch; i++){
            readXvecFvec<DataType>(DatasetInput, DatasetBatch.data(), Dimension, Batch_size, true,false);
            DatasetNeighborIDInput.read((char *) DatasetNeighborIDBatch.data(), Batch_size * NeighborNum * sizeof(uint32_t));
            DatasetNeighborDistInput.read((char *) DatasetNeighborDistBatch.data(), Batch_size * NeighborNum * sizeof(float));

            for (uint32_t j = 0; j < Batch_size; j++){
                uint32_t VectorID = i * Batch_size + j;
                uint32_t VectorClusterID = VectorAssignment[VectorID];

                if(BoundaryConflictMap.find(VectorClusterID) != BoundaryConflictMap.end()){
                    for (auto TargetIDIt = BoundaryConflictMap[VectorClusterID].begin(); TargetIDIt != BoundaryConflictMap[VectorClusterID].end(); TargetIDIt++){
                        if (TargetIDIt->second.count(VectorID) != 0){
                            uint32_t NNClusterID = VectorClusterID;
                            uint32_t TargetClusterID = TargetIDIt->first;

                            if (NNClusterID != VectorClusterID || NNClusterID != DatasetNeighborIDBatch[j * NeighborNum]){
                                std::cout << "Error in vector assignmentID and NN ClusterID: " << NNClusterID << " " << VectorClusterID << " " << DatasetNeighborIDBatch[j * NeighborNum]<< "\n";
                                exit(0);
                            }

                            auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);

                            if (BaseConflict && !UseQuantize){
                                if (BoundaryConflictCentroidMap.find(ClusterIndice) == BoundaryConflictCentroidMap.end()){
                                    BoundaryConflictCentroidMap.insert(std::make_pair(ClusterIndice, std::vector<float>(Dimension, 0)));
                                }
                                faiss::fvec_add(Dimension, BoundaryConflictCentroidMap[ClusterIndice].data(), DatasetBatch.data() + j * Dimension, BoundaryConflictCentroidMap[ClusterIndice].data());
                            }

                            float NNClusterDist = DatasetNeighborDistBatch[j * NeighborNum];
                            float TargetClusterDist = -1;

                            for (size_t temp0 = 1; temp0 < NeighborNum; temp0++){
                                if (DatasetNeighborIDBatch[j * NeighborNum + temp0] == TargetClusterID){
                                    TargetClusterDist = DatasetNeighborDistBatch[j * NeighborNum + temp0];
                                    break;
                                }
                            }
                            if (TargetClusterDist < 0){
                                TargetClusterDist = faiss::fvec_L2sqr(DatasetBatch.data() + j * Dimension, CenGraph->getDataByInternalId(TargetClusterID), Dimension);
                            }
                            if (NNClusterDist > TargetClusterDist){ std::cout << NNClusterDist << " " << TargetClusterDist << "\n";}
                            assert(NNClusterDist <= TargetClusterDist);

                            float CentroidDist = faiss::fvec_L2sqr(CenGraph->getDataByInternalId(NNClusterID), CenGraph->getDataByInternalId(TargetClusterID), Dimension);

                            // Compute the distance between the vector and the boundary
                            float CosNLNNTarget = (NNClusterDist + CentroidDist - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidDist));
                            // Note this is sqrt distance, Todo: we use the boundary or the centroid distance?
                            float DistNLBoundary = sqrt(CentroidDist) / 2 - sqrt(NNClusterDist) * CosNLNNTarget;

                            assert(BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end() && BoundaryConflictMap[NNClusterID].find(TargetClusterID) != BoundaryConflictMap[NNClusterID].end());

                            if (BaseConflict && UseQuantize){
                                float VectorAlpha = NNClusterDist * CosNLNNTarget * CosNLNNTarget / (TargetClusterDist);
                                if (BoundaryConflictAlphaMap.find(ClusterIndice) == BoundaryConflictAlphaMap.end()){
                                    // Project this vector to the line between target and NN centroid, record the alpha
                                    BoundaryConflictAlphaMap.insert(std::make_pair(ClusterIndice, std::make_pair(std::vector<uint32_t>(), std::vector<float>())));
                                }
                                BoundaryConflictAlphaMap[ClusterIndice].first.emplace_back(VectorID);
                                BoundaryConflictAlphaMap[ClusterIndice].second.emplace_back(VectorAlpha);
                            }

                            if (DistPrune){
                                if (BoundaryConflictDistMap.find(ClusterIndice) == BoundaryConflictDistMap.end()){
                                    BoundaryConflictDistMap[ClusterIndice] = std::make_pair(1, std::make_tuple(DistNLBoundary, TargetClusterDist, NNClusterDist, (NNClusterDist / TargetClusterDist)));
                                }
                                else{
                                    BoundaryConflictDistMap[ClusterIndice].first ++;
                                    std::get<0>(BoundaryConflictDistMap[ClusterIndice].second) += DistNLBoundary;
                                    std::get<1>(BoundaryConflictDistMap[ClusterIndice].second) += TargetClusterDist;
                                    std::get<2>(BoundaryConflictDistMap[ClusterIndice].second) += NNClusterDist;
                                    std::get<3>(BoundaryConflictDistMap[ClusterIndice].second) += (NNClusterDist / TargetClusterDist);
                                }
                            }
                            NLVectorSize ++;
                        }
                    }
                }
            }
        }

        if (DistPrune){
            for (auto it = BoundaryConflictDistMap.begin(); it != BoundaryConflictDistMap.end(); it++){
                uint32_t TargetClusterID = it->first.first; uint32_t NNClusterID = it->first.second;
                uint32_t ListSize = BoundaryConflictMap[NNClusterID][TargetClusterID].size();
                assert(ListSize == it->second.first);
                std::get<0>(it->second.second) /= ListSize;
                std::get<1>(it->second.second) /= ListSize;
                std::get<2>(it->second.second) /= ListSize;
                std::get<3>(it->second.second) /= ListSize;
                //std::cout << TargetClusterID << " " << NNClusterID << " " << std::get<0>(it->second.second) << " " << std::get<1>(it->second.second) << " " << std::get<2>(it->second.second) << " " << std::get<3>(it->second.second) << " " << ListSize << "\n";
            }
        }

        TRecorder.print_time_usage("Compute the boundary threshold distance and vector alpha");

        // We are using a smaller subset for construction, delete the original conflict map for the train set
        if (!BaseConflict){
            assert(DistPrune);
            for (auto it = BoundaryConflictMap.begin(); it != BoundaryConflictMap.end(); it++){
                uint32_t NNClusterID = it->first;
                for (auto setit = it->second.begin(); setit != it->second.end(); setit++){
                    setit->second.clear();
                    uint32_t TargetClusterID = setit->first;

                    auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);

                    if (UseQuantize){
                        assert (BoundaryConflictAlphaMap.find(ClusterIndice) == BoundaryConflictAlphaMap.end());
                        BoundaryConflictAlphaMap.insert(std::make_pair(ClusterIndice, std::make_pair(std::vector<uint32_t>(), std::vector<float>())));
                        assert (BoundaryConflictAlphaMap.find(ClusterIndice) != BoundaryConflictAlphaMap.end());
                    }
                    else{
                        assert(BoundaryConflictCentroidMap.find(ClusterIndice) == BoundaryConflictCentroidMap.end());
                        BoundaryConflictCentroidMap.insert(std::make_pair(ClusterIndice, std::vector<float>(Dimension, 0)));
                        assert(BoundaryConflictCentroidMap.find(ClusterIndice) != BoundaryConflictCentroidMap.end() && BoundaryConflictCentroidMap[ClusterIndice].size() == Dimension);
                    }
                }
            }
        }

        // Scan the vectors in baseset to include them in neighbor list
        if (DistPrune){
            uint32_t InsertedSize = 0;
            std::ifstream BaseNeighborIDInput(PathBaseNeighborID, std::ios::binary);
            std::ifstream BaseNeighborDistInput(PathBaseNeighborDist, std::ios::binary);
            std::ifstream BaseInput (PathBase, std::ios::binary);
 
            size_t Base_batch_size = nb / Num_batch;
            std::vector<float> BaseBatch(Base_batch_size * Dimension);
            std::vector<uint32_t> BaseNeighborIDBatch(Base_batch_size * NeighborNum);
            std::vector<float> BaseNeighborDistBatch(Base_batch_size * NeighborNum);
            std::vector<uint32_t> BaseAssignment(nb);

            for (size_t i = 0; i < Num_batch; i++){
                BaseNeighborIDInput.read((char *) BaseNeighborIDBatch.data(), Base_batch_size * NeighborNum * sizeof(uint32_t));
                for (size_t j = 0; j < Base_batch_size; j++){
                    BaseAssignment[i * Base_batch_size + j] = BaseNeighborIDBatch[j * NeighborNum];
                }
            }
            BaseNeighborIDInput.seekg(0, std::ios::beg);

            for (uint32_t i = 0; i < Num_batch; i++){
                readXvecFvec<DataType>(BaseInput, BaseBatch.data(), Dimension, Base_batch_size, true,false);

                BaseNeighborIDInput.read((char *)BaseNeighborIDBatch.data(), Base_batch_size * NeighborNum * sizeof(uint32_t));
                BaseNeighborDistInput.read((char *) BaseNeighborDistBatch.data(), Base_batch_size * NeighborNum * sizeof(float));
                std::vector<std::vector<std::pair<uint32_t, float>>> VectorInsertion(Base_batch_size, std::vector<std::pair<uint32_t, float>>());

#pragma omp parallel for
                for (uint32_t j = 0; j < Base_batch_size; j++){
                    uint32_t VectorID = i * Base_batch_size + j;
                    uint32_t NNClusterID = BaseAssignment[VectorID];
                    assert(NNClusterID == BaseNeighborIDBatch[j * NeighborNum]);

                    if (BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end()){
                        for (auto it = BoundaryConflictMap[NNClusterID].begin(); it != BoundaryConflictMap[NNClusterID].end(); it++){

                            uint32_t TargetClusterID = it->first;
                            if (it->second.count(VectorID) != 0){
                                // Vector already in the neighbor list
                                continue;
                            }
                            else{
                                float NNClusterDist = BaseNeighborDistBatch[j * NeighborNum];
                                float TargetClusterDist = -1;

                                for (size_t temp = 1; temp < NeighborNum; temp++){
                                    if (BaseNeighborIDBatch[j * NeighborNum + temp] == TargetClusterID){
                                        TargetClusterDist = BaseNeighborDistBatch[j * NeighborNum + temp];
                                        break;
                                    }
                                }

                                if (TargetClusterDist < 0){
                                    TargetClusterDist = faiss::fvec_L2sqr(BaseBatch.data() + j * Dimension, CenGraph->getDataByInternalId(TargetClusterID), Dimension);
                                }
                                if (NNClusterDist > TargetClusterDist){
                                    // It denotes we made mistake in base vector assignment: re-assign the base vectors and the insertion
                                    std::cout << VectorID << " " << NNClusterID << " " << NNClusterDist << " " << TargetClusterID << " " << TargetClusterDist << "\n";
                                    exit(0);
                                }

                                float CentroidDist = faiss::fvec_L2sqr(CenGraph->getDataByInternalId(NNClusterID), CenGraph->getDataByInternalId(TargetClusterID), Dimension);

                                float CosVecNNTarget = (NNClusterDist + CentroidDist - TargetClusterDist) / (2 * sqrt(NNClusterDist * CentroidDist));
                                float DistNLBoundary = sqrt(CentroidDist) / 2 - sqrt(NNClusterDist) * CosVecNNTarget;
                                float VectorAlpha = sqrt(NNClusterDist) * CosVecNNTarget / sqrt(CentroidDist);

                                //std::cout << NNClusterDist << " " << TargetClusterDist << " " << CentroidsDist[TargetClusterID * nc + NNClusterID] << " " << CosVecNNTarget << " " << VectorAlpha << "\n";

                                auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);
                                assert(BoundaryConflictDistMap.find(ClusterIndice) != BoundaryConflictDistMap.end());
/*
                                if ((NNClusterDist / TargetClusterDist) < std::get<3>(BoundaryConflictDistMap[ClusterIndice].second)){
                                    VectorInsertion[j].emplace_back(TargetClusterID, NNClusterID);
                                }
*/

                                // The distance between the query and the boundary is smaller than 
                                // the threshold between the distance between the vector and the boundary 

                                //std::cout << NNClusterDist << " " << TargetClusterDist << " " << CentroidsDist[TargetClusterID * nc + NNClusterID] << " " << CosVecNNTarget << " " << DistNLBoundary << " " << std::get<0>(BoundaryConflictDistMap[ClusterIndice].second) << "\n";

                                if (DistNLBoundary < std::get<0>(BoundaryConflictDistMap[ClusterIndice].second)){
                                    VectorInsertion[j].emplace_back(TargetClusterID, VectorAlpha);
                                }
/*
                                if (TargetClusterDist < std::get<1>(BoundaryConflictDistMap[ClusterIndice].second)){
                                    BoundaryConflictMap[NNClusterID][TargetClusterID].insert(VectorID);
                                }
*/
                            }
                        }
                    }
                }
                std::cout << "Distance computation completed\n";

                for (uint32_t j = 0; j < Base_batch_size; j++){

                    if (j % (Base_batch_size / (1000)) == 0){
                        std::cout << j <<" / " << Base_batch_size << "\r";
                    }

                    size_t InsertionSize = VectorInsertion[j].size();
                    InsertedSize += InsertionSize;
                    uint32_t VectorID = i * Base_batch_size + j;
                    //std::cout << "1\n";
                    uint32_t NNClusterID = BaseAssignment[VectorID];

                    for (size_t temp = 0; temp < InsertionSize; temp++){
                        uint32_t TargetClusterID = VectorInsertion[j][temp].first;
                        float VectorAlpha = VectorInsertion[j][temp].second;

                        //std::cout << "2 " << TargetClusterID << " " << NNClusterID << " " << temp << "\n";
                        assert(BoundaryConflictMap.find(NNClusterID) != BoundaryConflictMap.end() && BoundaryConflictMap[NNClusterID].find(TargetClusterID) != BoundaryConflictMap[NNClusterID].end());
                        BoundaryConflictMap[NNClusterID][TargetClusterID].insert(VectorID);

                        auto ClusterIndice = std::make_pair(TargetClusterID, NNClusterID);

                        if (UseQuantize){
                            //std::cout << "3\n";
                            assert(BoundaryConflictAlphaMap.find(ClusterIndice) != BoundaryConflictAlphaMap.end());
                            BoundaryConflictAlphaMap[ClusterIndice].first.emplace_back(VectorID);
                            BoundaryConflictAlphaMap[ClusterIndice].second.emplace_back(VectorAlpha);
                        }
                        else{
                            assert(BoundaryConflictCentroidMap.find(ClusterIndice) != BoundaryConflictCentroidMap.end());
                            assert(BoundaryConflictCentroidMap[ClusterIndice].size() == Dimension);
                            faiss::fvec_add(Dimension, BoundaryConflictCentroidMap[ClusterIndice].data(),BaseBatch.data() + j * Dimension, BoundaryConflictCentroidMap[ClusterIndice].data());
                        }
                    }
                }
                std::cout << "Insert the vectors, " << InsertedSize << " items inserted\n";
                TRecorder.print_time_usage("Completed scan on batch " + std::to_string(i + 1) + " / " + std::to_string(Num_batch));
            }

            std::cout << "Inserted number of vectors: " << InsertedSize << "\n";
            TRecorder.print_time_usage("Scan all vector candidates for neighbor list");
        }

        // Erase the neighbor list with no vectors
        if (!BaseConflict){
            for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end();){
                uint32_t NNClusterID = NNIt->first;

                for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end();){
                    
                    if (TargetIt->second.size() == 0){
                        uint32_t TargetClusterID = TargetIt->first;
                        NNIt->second.erase(TargetIt ++);
                        if (UseQuantize){BoundaryConflictAlphaMap.erase(std::make_pair(TargetClusterID, NNClusterID));}
                        else{BoundaryConflictCentroidMap.erase(std::make_pair(TargetClusterID, NNClusterID));}
                    }
                    else{
                        ++ TargetIt;
                    }
                }

                if (NNIt->second.size() == 0){
                    BoundaryConflictMap.erase( NNIt++);
                }
                else{
                    ++ NNIt;
                }
            }
        }

        std::map<std::pair<uint32_t, uint32_t>, float> NeighborListAlpha;
        uint32_t NLIndice = 0;

        // Project the center of the neighborlist to the line between the target centroid and the NN centroid
        if (!UseQuantize){
            for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end(); NNIt ++){
                uint32_t NNClusterID = NNIt->first;

                for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end(); TargetIt++){

                    uint32_t TargetClusterID = TargetIt->first;

                    size_t NeighborListSize = TargetIt->second.size();

                    assert(BoundaryConflictCentroidMap[std::make_pair(TargetClusterID, NNClusterID)].size() == Dimension);

                    for (size_t i = 0; i < Dimension; i++){
                        BoundaryConflictCentroidMap[std::make_pair(TargetClusterID, NNClusterID)][i] /= NeighborListSize;
                    }

                    // Project the neighborlist centroid to the line between two centroids of two clusters: TargetCluster and NNCLuster
                    // Let the project point is: p0, the target cluster centroid: p1, the NN cluster centroid: p2, the neighborlist centroid: p3
                    // As the neighborlist centroid are in the NN cluster centroid, we compute the alpha: alpha = ||p3, p2|| / ||p1, p2|| and p0 = p2 + alpha * (p1 - p2)
                    // cos(A) = (a^2 + c^2 - b^2) / (2 * a * c)
                    float DistCenNNCen = faiss::fvec_L2sqr(BoundaryConflictCentroidMap[std::make_pair(TargetClusterID, NNClusterID)].data(), CenGraph->getDataByInternalId(NNClusterID), Dimension);
                    float DistCenTarCen = faiss::fvec_L2sqr(BoundaryConflictCentroidMap[std::make_pair(TargetClusterID, NNClusterID)].data(), CenGraph->getDataByInternalId(TargetClusterID), Dimension);

                    float DistTarCenNNCen = faiss::fvec_L2sqr(CenGraph->getDataByInternalId(NNClusterID), CenGraph->getDataByInternalId(TargetClusterID), Dimension);

                    float CosNLNNTarget = (DistCenNNCen + DistTarCenNNCen - DistCenTarCen) / (2 * sqrt(DistCenNNCen * DistTarCenNNCen));
                    float Alpha = sqrt(DistCenNNCen) * CosNLNNTarget / sqrt(DistTarCenNNCen);
                    NeighborListAlpha[std::make_pair(NNClusterID, TargetClusterID)] =  Alpha;

                    // Record the proportion of distance between NN centroid and the projected point
                    if (Alpha > 0.5){
                        std::cout << " LimBoundist: " << std::get<0>(BoundaryConflictDistMap[std::make_pair(TargetClusterID, NNClusterID)].second) << " CenBoundist: " << sqrt(DistTarCenNNCen) / 2 - sqrt(DistCenNNCen) * CosNLNNTarget << "\n";
                        std::cout << " NN: " << TargetClusterID << " Target: " << NNClusterID << " Indice: " << NLIndice << " Size: " << NeighborListSize << " " << DistCenNNCen << " " << DistCenTarCen << " " << DistTarCenNNCen << " cos: " << CosNLNNTarget << " alpha: " << Alpha<< "\n";
                        for (auto iter = TargetIt->second.begin(); iter !=  TargetIt->second.end(); iter++){
                            std::cout << *iter << " ";
                        }
                        std::cout << "\n";
                        auto result = CenGraph->searchKnn(BoundaryConflictCentroidMap[std::make_pair(TargetClusterID, NNClusterID)].data(), nc);

                        for (size_t temp = 0; temp < nc; temp++){
                            std::cout << result.top().second << " " << result.top().first << " | ";
                            result.pop();
                        }
                        std::cout << "\n";
                        exit(0);
                    }
                    NLIndice ++;
                }
            }
            TRecorder.print_time_usage("Compute the alpha for all neighbor list centroids");
        }

        // Run clustering to decide the quantization of each neighbor list, the number of quantization is decided by the neighbor list size
        if (UseQuantize){
#pragma omp parallel for
            for (size_t Indice = 0; Indice < BoundaryConflictAlphaMap.size(); Indice++){
                auto ClusterIt = BoundaryConflictAlphaMap.begin();
                std::advance(ClusterIt, Indice);

                size_t NLSize = ClusterIt->second.first.size();
                size_t NumQuantization = std::ceil(float(NLSize) / QuantBatchSize);
                std::vector<uint32_t> VectorIdx(NLSize);
                std::iota(VectorIdx.begin(), VectorIdx.end(), 0);
                std::stable_sort(VectorIdx.begin(), VectorIdx.end(), [&ClusterIt](size_t i1, size_t i2){ return ClusterIt->second.second[i1] < ClusterIt->second.second[i2];});

                std::vector<float> VectorAlphaCopy(NLSize); std::vector<uint32_t> VectorIDCopy(NLSize);
                memcpy(VectorAlphaCopy.data(), ClusterIt->second.second.data(), NLSize * sizeof(float));
                memcpy(VectorIDCopy.data(), ClusterIt->second.first.data(), NLSize * sizeof(uint32_t));

                ClusterIt->second.first.resize(0); ClusterIt->second.second.resize(NumQuantization);
                ClusterIt->second.first.emplace_back(NLSize); ClusterIt->second.first.emplace_back(NumQuantization); 

                ClusterIt->second.second.resize(NumQuantization);

                uint32_t VectorIndice = 0;
                for (size_t i = 0; i < NumQuantization; i++){
                    ClusterIt->second.second[i] = VectorAlphaCopy[VectorIdx[VectorIndice]];

                    for (size_t j = 0; j < QuantBatchSize; j++){
                        if (VectorIndice >= NLSize){break;}

                        ClusterIt->second.first.emplace_back(VectorIDCopy[VectorIdx[VectorIndice]]);
                        VectorIndice ++;
                    }
                }
                assert(ClusterIt->second.first.size() == (1 + 1 + NLSize));

            }
            TRecorder.print_time_usage("Cluster the vector alpha for neighbor list");
        }

        std::ofstream NeighborListOutput(PathNeighborList, std::ios::binary);
        std::ofstream NeighborListInfoOutput(PathNeighborListInfo, std::ios::binary);

        // All information for Neighbor List is acquired, save the neighbor list index to disk
        if (UseQuantize){
            for (auto ClusterIt = BoundaryConflictAlphaMap.begin(); ClusterIt != BoundaryConflictAlphaMap.end(); ClusterIt++){
                assert(ClusterIt->second.first[0] > 0);
                uint32_t TargetClusterID = ClusterIt->first.first;
                uint32_t NNClusterID = ClusterIt->first.second;

                NeighborListOutput.write((char *) & TargetClusterID, sizeof(uint32_t));
                NeighborListOutput.write((char *) & NNClusterID, sizeof(uint32_t));
                size_t IDSize = ClusterIt->second.first.size();
                NeighborListOutput.write((char *) & IDSize, sizeof(uint32_t));
                size_t AlphaSize = ClusterIt->second.second.size();
                NeighborListOutput.write((char *) & AlphaSize, sizeof(uint32_t));

                NeighborListOutput.write((char *) ClusterIt->second.first.data(), IDSize * sizeof(uint32_t));
                NeighborListOutput.write((char *) ClusterIt->second.second.data(), AlphaSize * sizeof(float));
            }
        }
        else{

            std::cout << "Reorgnize and save the neighor list to disk\n";
            uint32_t NumAlphaList = 0, NumTarList = 0, NumVecID = 0, 
                     MaxVecListSize = 0, MaxTarListSize = 0, MaxAlphaListSize = 0,
                     MinVecListSize = std::numeric_limits<uint32_t>::max(), MinTarListSize = std::numeric_limits<uint32_t>::max(), MinAlphaListSize = std::numeric_limits<uint32_t>::max();

            // Iteate over the clusters and construct the neighbor list in sequence
            std::vector<std::vector<uint32_t>> NeighborListClusters(nc, std::vector<uint32_t>{});
            std::vector<std::vector<uint32_t>> NeighborListVecID(nc, std::vector<uint32_t>{});
            std::vector<std::vector<uint32_t>> NeighborListNumCluster(nc, std::vector<uint32_t>{});

            for (auto NNIt = BoundaryConflictMap.begin(); NNIt != BoundaryConflictMap.end(); NNIt++){

                uint32_t NNClusterID = NNIt->first;
                NumAlphaList += NNIt->second.size();
                if (NNIt->second.size() > MaxAlphaListSize){MaxAlphaListSize = NNIt->second.size();}
                if (NNIt->second.size() < MinAlphaListSize){MinAlphaListSize = NNIt->second.size();}

                // Include all vectors in this cluster, index as the vector ID <-> cluster IDs
                std::unordered_map<uint32_t, std::vector<uint32_t>> VecInNls;
                for (auto TargetIt = NNIt->second.begin(); TargetIt != NNIt->second.end();TargetIt++){

                    uint32_t TargetClusterID = TargetIt->first;

                    for (auto VecIDIt = TargetIt->second.begin(); VecIDIt != TargetIt->second.end(); VecIDIt++){
                        if (VecInNls.find(*VecIDIt) != VecInNls.end()){
                            VecInNls[*VecIDIt].emplace_back(TargetClusterID);
                        }
                        else{
                            VecInNls.insert(std::make_pair(*VecIDIt, std::vector<uint32_t>{TargetClusterID}));
                        }
                    }
                }

                for (auto VecIt = VecInNls.begin(); VecIt != VecInNls.end(); VecIt++){
                    NeighborListVecID[NNClusterID].emplace_back(VecIt->first);
                    NeighborListNumCluster[NNClusterID].emplace_back(VecIt->second.size());
                    std::stable_sort(VecIt->second.begin(), VecIt->second.end());
                    NeighborListClusters[NNClusterID].insert(NeighborListClusters[NNClusterID].end(), VecIt->second.begin(), VecIt->second.end());
                }
                NumVecID += NeighborListVecID[NNClusterID].size();
                NumTarList += NeighborListClusters[NNClusterID].size();

                if (NeighborListVecID[NNClusterID].size() > MaxVecListSize){MaxVecListSize = NeighborListVecID[NNClusterID].size();}
                if (NeighborListClusters[NNClusterID].size() > MaxTarListSize){MaxTarListSize = NeighborListClusters[NNClusterID].size();}
                if (NeighborListVecID[NNClusterID].size() < MinVecListSize){MinVecListSize = NeighborListVecID[NNClusterID].size();}
                if (NeighborListClusters[NNClusterID].size() < MinTarListSize){MinTarListSize  = NeighborListClusters[NNClusterID].size();}
            }

            uint32_t Indice = 0;
            uint32_t PrintIndice = nc - 1;
            std::cout << "Print the vector ID and target cluster ID in cluster: " << PrintIndice << "\n";
            for (size_t i = 0; i < NeighborListVecID[PrintIndice].size() / 10; i++){
                std::cout << NeighborListVecID[PrintIndice][i] << " ";
                std::cout << NeighborListNumCluster[PrintIndice][i] << " ";
                for (size_t j = Indice; j < Indice + NeighborListNumCluster[PrintIndice][i]; j++){
                    std::cout << NeighborListClusters[PrintIndice][j] << " ";
                }
                Indice += NeighborListNumCluster[PrintIndice][i];
                std::cout << " | ";
            }
            std::cout << "\n";

            FastPForLib::CODECFactory factory;
            FastPForLib::IntegerCODEC &codec = * factory.getFromName("simdfastpfor128");
            std::vector<uint32_t> Compressedoutput;
            std::vector<uint32_t> Compressedsizelist(nc, 0);

            NeighborListInfoOutput.write((char *) & NumAlphaList, sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) & NumVecID, sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) & NumTarList, sizeof(uint32_t));

            NeighborListInfoOutput.write((char *) & MaxVecListSize, sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) & MaxTarListSize, sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) & MaxAlphaListSize, sizeof(uint32_t));
            std::cout << "The NumAlphaList: " << NumAlphaList << " The NumVectorList: " << NumVecID << " The NumTarList: " << NumTarList <<
            " The MaxVecListSize: " << MaxVecListSize << " The MaxTarListSize: " << MaxTarListSize << " The MaxAlphaListSize: " << MaxAlphaListSize << 
            " The MinVecListSize: " << MinVecListSize << " The MinTarListSize: " << MinTarListSize << " The MinAlphaListSize: " << MinAlphaListSize;

            // Write the alpha list of all neighbor lists
            std::vector<float> AlphaCentroidNorm;

            for (uint32_t i = 0; i < nc; i++){
                size_t Compressedsize = 0;
                
                if (BoundaryConflictMap.find(i) != BoundaryConflictMap.end()){
                    for (auto it = BoundaryConflictMap[i].begin(); it != BoundaryConflictMap[i].end(); it++){
                        float Alpha = NeighborListAlpha[std::make_pair(i, it->first)];
                        NeighborListOutput.write((char *) & Alpha, sizeof(float));
                        
                        std::vector<float> AlphaCentroidTemp(Dimension, 0);
                        faiss::fvec_madd(Dimension, AlphaCentroidTemp.data(), (1 - Alpha), CenGraph->getDataByInternalId(i), AlphaCentroidTemp.data());
                        faiss::fvec_madd(Dimension, AlphaCentroidTemp.data(), Alpha, CenGraph->getDataByInternalId(it->first), AlphaCentroidTemp.data());
                        AlphaCentroidNorm.emplace_back(faiss::fvec_norm_L2sqr(AlphaCentroidTemp.data(), Dimension));
                    }
                    Compressedsize = BoundaryConflictMap[i].size();
                }
                Compressedsizelist[i] = i == 0? Compressedsize : Compressedsizelist[i - 1] + Compressedsize;
            }
            assert(Compressedsizelist[nc - 1] == NumAlphaList);
            assert(NumAlphaList == AlphaCentroidNorm.size());
            NeighborListOutput.write((char *) AlphaCentroidNorm.data(), AlphaCentroidNorm.size() * sizeof(float));
            NeighborListOutput.write((char *) Compressedsizelist.data(), nc * sizeof(uint32_t));


            // Write the targetID of neighbor list of all neighbor lists 
            for (uint32_t i = 0; i < nc; i++){

                std::vector<uint32_t> AlphaList;
                if (BoundaryConflictMap.find(i) != BoundaryConflictMap.end()){
                    for (auto it = BoundaryConflictMap[i].begin(); it != BoundaryConflictMap[i].end(); it++){
                        AlphaList.emplace_back(it->first);
                    }
                }

                size_t Compressedsize = 0;
                if (AlphaList.size() > 0){
                    Compressedoutput.resize(AlphaList.size() + 1024);
                    Compressedsize = Compressedoutput.size();

                    codec.encodeArray(AlphaList.data(), AlphaList.size(), Compressedoutput.data(), Compressedsize);
                    //memcpy(Compressedoutput.data(), AlphaList.data(), AlphaList.size() * sizeof(uint32_t));
                    //Compressedsize = AlphaList.size();

                    NeighborListOutput.write((char *) Compressedoutput.data(), Compressedsize * sizeof(uint32_t));
                    if (Compressedsize > AlphaList.size()){
                        std::cout << "Compressedsize: " << Compressedsize << " ALphalist Size: " << AlphaList.size() << "\n";
                        assert(Compressedsize <= AlphaList.size());
                    }
                }

                Compressedsizelist[i] = i == 0? Compressedsize : Compressedsizelist[i - 1] + Compressedsize;
            }

            NeighborListOutput.write((char *) Compressedsizelist.data(), nc * sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) (Compressedsizelist.data() + nc - 1), sizeof(uint32_t));

            std::cout << " The CompNumAlpha: " << Compressedsizelist[nc - 1];

        /*-------------------------------------------------------------------*/
            for (uint32_t i = 0; i < nc; i++){
                size_t Compressedsize = 0;
                if (NeighborListVecID[i].size() > 0){
                    Compressedoutput.resize(NeighborListVecID[i].size() + 1024);
                    Compressedsize = Compressedoutput.size();

                    codec.encodeArray(NeighborListVecID[i].data(), NeighborListVecID[i].size(), Compressedoutput.data(), Compressedsize);
                    //memcpy(Compressedoutput.data(), NeighborListVecID[i].data(), NeighborListVecID[i].size() * sizeof(uint32_t));
                    //Compressedsize = NeighborListVecID[i].size();

                    if (Compressedsize > NeighborListVecID[i].size()){
                        std::cout << "\nCompressedsize: " << Compressedsize << " NeighborListVecID Size: " << NeighborListVecID[i].size() << "\n";
                        for (size_t temp = 0; temp < Compressedsize; temp++){
                            std::cout << Compressedoutput[temp] << " ";
                        }
                        std::cout << "\n";
                        assert(Compressedsize <= NeighborListVecID[i].size());
                    }

                    NeighborListOutput.write((char *) Compressedoutput.data(), Compressedsize * sizeof(uint32_t));
                }
                Compressedsizelist[i] = i == 0? Compressedsize : Compressedsizelist[i - 1] + Compressedsize;
            }
            NeighborListOutput.write((char *) Compressedsizelist.data(), nc * sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) (Compressedsizelist.data() + nc - 1), sizeof(uint32_t));
            std::cout << " CompNumVecID: " << Compressedsizelist[nc - 1];

            for (uint32_t i = 0; i < nc; i++){
                size_t Compressedsize = 0;
                if (NeighborListNumCluster[i].size() > 0){
                    Compressedoutput.resize(NeighborListNumCluster[i].size() + 1024);
                    Compressedsize = Compressedoutput.size();
                    
                    codec.encodeArray(NeighborListNumCluster[i].data(), NeighborListNumCluster[i].size(), Compressedoutput.data(), Compressedsize);
                    //memcpy(Compressedoutput.data(), NeighborListNumCluster[i].data(), NeighborListNumCluster[i].size() * sizeof(uint32_t));
                    //Compressedsize = NeighborListNumCluster[i].size();

                    if (Compressedsize > NeighborListNumCluster[i].size()){
                        std::cout << "Compressedsize: " << Compressedsize << " NeighborListNumCluster Size: " << NeighborListNumCluster[i].size() << "\n";
                        for (size_t temp = 0; temp < Compressedsize; temp++){
                            std::cout << Compressedoutput[temp] << " ";
                        }
                        std::cout << "\n";
                        assert(Compressedsize <= NeighborListNumCluster[i].size());
                    }

                    NeighborListOutput.write((char *) Compressedoutput.data(), Compressedsize * sizeof(uint32_t));
                }
                
                Compressedsizelist[i] = i == 0? Compressedsize : Compressedsizelist[i - 1] + Compressedsize;
            }
            NeighborListOutput.write((char *) Compressedsizelist.data(), nc * sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) (Compressedsizelist.data() + nc - 1), sizeof(uint32_t));
            std::cout << " CompNumTar: " << Compressedsizelist[nc - 1];

            for (uint32_t i = 0; i < nc; i++){
                size_t Compressedsize = 0;
                if (NeighborListClusters[i].size() > 0){
                    Compressedoutput.resize(NeighborListClusters[i].size() + 1024);
                    Compressedsize = Compressedoutput.size();
                    
                    codec.encodeArray(NeighborListClusters[i].data(), NeighborListClusters[i].size(), Compressedoutput.data(), Compressedsize);
                    //memcpy(Compressedoutput.data(), NeighborListClusters[i].data(), NeighborListClusters[i].size() * sizeof(uint32_t));
                    //Compressedsize = NeighborListClusters[i].size();

                    if (Compressedsize > NeighborListClusters[i].size()){
                        std::cout << "Compressedsize: " << Compressedsize << " NeighborListClusters Size: " << NeighborListClusters[i].size() << "\n";
                        for (size_t temp = 0; temp < Compressedsize; temp++){
                            std::cout << Compressedoutput[temp] << " ";
                        }
                        std::cout << "\n";
                        assert(Compressedsize <= NeighborListClusters[i].size());
                    }

                    NeighborListOutput.write((char *) Compressedoutput.data(), Compressedsize * sizeof(uint32_t));
                }

                Compressedsizelist[i] = i == 0 ? Compressedsize : Compressedsizelist[i - 1] + Compressedsize;
            }

            NeighborListOutput.write((char *) Compressedsizelist.data(), nc * sizeof(uint32_t));
            NeighborListInfoOutput.write((char *) (Compressedsizelist.data() + nc - 1), sizeof(uint32_t));
            std::cout << " CompTarID: " << Compressedsizelist[nc - 1];

            NeighborListOutput.close();
            NeighborListInfoOutput.close();
        }
        exit(0);
        return;
    }
}


template<typename T>
uint32_t LoadClusterInfo(size_t MaxNumInMemoryClusters, size_t ClusterSize, uint32_t ClusterID, size_t DataD, size_t LoadD, std::ifstream & BaseInvInput,
std::unordered_map<uint32_t, uint32_t> & InMemoryClusterMap, std::vector<std::vector<T>> & LoadedClusterData, std::vector<size_t> & AccumulateSize, 
std::unordered_set<uint32_t> & RequiredCluster
){
    auto result = InMemoryClusterMap.find(ClusterID);

    if (result != InMemoryClusterMap.end()){
        return result->second;
    }
    size_t NumInMemoryClusters = InMemoryClusterMap.size();
    std::vector<T> FullData(DataD);

    if (NumInMemoryClusters < MaxNumInMemoryClusters){
        // Check if there is still avaliable memory, if yes, expand the list, otherwise, delete the cluster not in use
        LoadedClusterData.resize(NumInMemoryClusters + 1);

        LoadedClusterData[NumInMemoryClusters].resize(ClusterSize * LoadD);
        BaseInvInput.seekg(AccumulateSize[ClusterID] * DataD * sizeof(T), std::ios::beg);

        for (size_t i = 0; i < ClusterSize; i++){
            BaseInvInput.read((char *) FullData.data(),  DataD * sizeof(T));
            memcpy(LoadedClusterData[NumInMemoryClusters].data() + i * LoadD, FullData.data(), LoadD * sizeof(T));
        }
        
        InMemoryClusterMap.insert({ClusterID, NumInMemoryClusters});
        return NumInMemoryClusters;
    }
    else{
        // Choose a un-required and overwrite on the data 
        for (auto it = InMemoryClusterMap.begin();  it != InMemoryClusterMap.end();){

            if (RequiredCluster.count(it->first) == 0){
                uint32_t ClusterIndice = it->second;
                LoadedClusterData[ClusterIndice].resize(ClusterSize * LoadD);
                BaseInvInput.seekg(AccumulateSize[ClusterID] * DataD * sizeof(T), std::ios::beg);

                for (size_t i = 0; i < ClusterSize; i++){
                    BaseInvInput.read((char *) FullData.data(), DataD * sizeof(T));
                    memcpy(LoadedClusterData[ClusterID].data() + i * LoadD, FullData.data(), LoadD * sizeof(T));
                }
                
                InMemoryClusterMap.erase(it);
                InMemoryClusterMap.insert({ClusterID, ClusterIndice});
                return ClusterIndice;
            }
            else{
                it ++;
            }
        }
    }
    std::cout << "Error: no space to load all required cluster information\n"; exit(0);
    return 0;
}

// Get the minimum search cost that can visit all of the neighbor groundtruth 
inline std::pair<size_t, size_t> FetchSearchCost(size_t NeighborNum, std::vector<size_t>  & AllClusterSize, uint32_t NNClusterID, uint32_t * NeighborClusterID){
    size_t SearchCost = 0;
    size_t VisitedGt = 0;
    for (size_t i = 0; i < NeighborNum; i++){
        SearchCost += AllClusterSize[NeighborClusterID[i]];
        if (NNClusterID == NeighborClusterID[i]){
            VisitedGt ++;
            break;
        }
    }
    return std::make_pair(SearchCost, VisitedGt);
}

// We want the vector is in the same cluster with their BeNNs
size_t NeiOptimize(uint32_t CentralClusterID, size_t TargetK, size_t TargetNeighborNum, bool Visualize,
std::vector<uint32_t> BaseClusterID, std::vector<size_t> & AllClusterSize, std::vector<std::vector<uint32_t>> & BaseClusterNN, std::unordered_map<uint32_t, uint32_t> & InMemoryClusterNNMap,
std::vector<uint32_t> & OriginAssignmentID, std::vector<std::unordered_map<uint32_t, uint32_t>> BaseInvIndices,
std::vector<uint32_t> & AssignmentID, std::vector<std::vector<uint32_t>> & BaseNeighborClusterID, std::unordered_map<uint32_t, uint32_t> & InMemoryClusterNeiIDMap,
std::vector<std::vector<uint32_t>> & BaseBeNN
){

    size_t NumShift = 0;
    assert(BaseNeighborClusterID[InMemoryClusterNeiIDMap[CentralClusterID]].size() == (BaseClusterID.size() * TargetNeighborNum));
    assert(BaseClusterNN[InMemoryClusterNNMap[CentralClusterID]].size() == (BaseClusterID.size() * TargetK));

    uint32_t CenNNClusterIndice = InMemoryClusterNNMap[CentralClusterID];
    uint32_t CenNeiClusterIndice = InMemoryClusterNNMap[CentralClusterID];

    for (size_t i = 0; i < BaseClusterID.size(); i++){
        uint32_t CenVectorID = BaseClusterID[i];
        assert(OriginAssignmentID[CenVectorID] == CentralClusterID);

        for (size_t j = 0; j < TargetK; j++){

            if(Visualize) std::cout << "\nChecking " << CentralClusterID << " th cluster "  << i << " th vector, the " << j << " th neighbor, total " << BaseClusterNN[CenNNClusterIndice].size() << " to be checked\n";

            uint32_t NN = BaseClusterNN[CenNNClusterIndice][i * TargetK + j];
            if (AssignmentID[NN] == OriginAssignmentID[CenVectorID]){ // If the NN id is the same with original id, no need to do the shift
                continue;
            }

            if(Visualize) std::cout << "gt: " << NN << " Central vector assign cluster: " << AssignmentID[CenVectorID] << " Origin Cluster: " << OriginAssignmentID[CenVectorID] << " NN Assign Cluster: " << AssignmentID[NN] << " ToBeNNNum: " << BaseBeNN[NN].size() << "\n";

            // Check the origin cost
            size_t NNRNNNum = BaseBeNN[NN].size(); // This is the number of vectors that take NN as its K nearest neighbors
            size_t OriginalSearchCost = 0;  // This is the search cost that is related to the NN vector
            float OriginalSearchK = 0;

            if (Visualize) std::cout << "Checking the original search cost: " << NNRNNNum << "\n";

            std::vector<uint32_t > NNClusterIndiceRecord(NNRNNNum);
            std::vector<uint32_t> NeiClusterIndiceRecord(NNRNNNum);
            std::vector<uint32_t> InnerClusterIndiceRecord(NNRNNNum);


            for (size_t temp0 = 0; temp0 < NNRNNNum; temp0++){
                uint32_t NNRNNID = BaseBeNN[NN][temp0];
                uint32_t NNRNNOriginClusterID = OriginAssignmentID[NNRNNID];
                NNClusterIndiceRecord[temp0] = InMemoryClusterNNMap[NNRNNOriginClusterID];
                NeiClusterIndiceRecord[temp0] = InMemoryClusterNeiIDMap[NNRNNOriginClusterID];
                InnerClusterIndiceRecord[temp0] = BaseInvIndices[NNRNNOriginClusterID][NNRNNID];
            }

            for (size_t temp0 = 0; temp0 < NNRNNNum; temp0++){

                    uint32_t NNClusterIndice = NNClusterIndiceRecord[temp0];
                    uint32_t InnerClusterIndice = InnerClusterIndiceRecord[temp0];
                    uint32_t NeiClusterIndice = NeiClusterIndiceRecord[temp0];
                    

                for (size_t temp1 = 0; temp1 < TargetK; temp1++){

                    uint32_t NNRNNNNID = BaseClusterNN[NNClusterIndice][InnerClusterIndice * TargetK + temp1];
                    auto result = FetchSearchCost(TargetNeighborNum, AllClusterSize, AssignmentID[NNRNNNNID],  BaseNeighborClusterID[NeiClusterIndice].data() + InnerClusterIndice * TargetNeighborNum);
                    if(Visualize) std::cout << temp0 << " " << temp1 << " " << result.first << " " << result.second << "\n";
                    OriginalSearchCost += result.first;
                    OriginalSearchK += result.second;
                }
            }

            //std::cout << "Check the origin cost\n";

            size_t MinShiftCost = OriginalSearchCost;
            float MaxShiftK = OriginalSearchK;
            int OptIndice = -1;
            uint32_t OriginID = AssignmentID[NN];

            for (size_t temp = 0; temp < TargetNeighborNum; temp++){

                uint32_t NeiSearchID = BaseNeighborClusterID[CenNeiClusterIndice][j * TargetNeighborNum + temp];
                if (NeiSearchID == AssignmentID[NN]){
                    break;
                }

                AssignmentID[NN] = NeiSearchID;
                AllClusterSize[OriginID]--;
                AllClusterSize[NeiSearchID]++;

                //std::cout << "Get the search cost of shift result\n";
                // Check the shift cost, can the shift reduce the search cost?

                size_t ShiftSearchCost = 0;
                float ShiftSearchK =0;
                if(Visualize) std::cout << "Checking the shift search cost: " << NNRNNNum << "\n";
                for (size_t temp0 = 0; temp0 < NNRNNNum; temp0++){
                        uint32_t NNClusterIndice = NNClusterIndiceRecord[temp0];
                        uint32_t InnerClusterIndice = InnerClusterIndiceRecord[temp0];
                        uint32_t NeiClusterIndice = NeiClusterIndiceRecord[temp0];

    //#pragma omp parallel for
                    for (size_t temp1 = 0; temp1 < TargetK; temp1++){
                        uint32_t NNRNNNNID = BaseClusterNN[NNClusterIndice][InnerClusterIndice * TargetK + temp1];

                        auto result = FetchSearchCost(TargetNeighborNum, AllClusterSize, AssignmentID[NNRNNNNID], BaseNeighborClusterID[NeiClusterIndice].data() + InnerClusterIndice * TargetNeighborNum);
                        ShiftSearchCost += result.first;
                        ShiftSearchK += result.second;
                        if(Visualize) std::cout << temp0 << " " << temp1 << " " << result.first << " " << result.second << "\n";
                    }
                }
                if(Visualize) std::cout << "The original search cost: " << OriginalSearchCost << " Shift search cost: " << ShiftSearchCost << "\n";

                if (ShiftSearchK >= MaxShiftK && ShiftSearchCost <= MinShiftCost){
                    MaxShiftK = ShiftSearchK; MinShiftCost = ShiftSearchCost;
                    OptIndice = temp;
                }

                AssignmentID[NN] = OriginID;
                AllClusterSize[OriginID]++;
                AllClusterSize[NeiSearchID]--;
            }
            // Shift the NN id to the best cluster (or any cluster on the search path)

            if (OptIndice >= 0){
                NumShift ++;
                uint32_t NeiSearchID = BaseNeighborClusterID[CenNeiClusterIndice][j * TargetNeighborNum + OptIndice];
                AssignmentID[NN] = NeiSearchID;
                AllClusterSize[OriginID]--;
                AllClusterSize[NeiSearchID]++;
                if(Visualize) std::cout << "Shift the vector from " << OriginID << " to " << NeiSearchID << " th cluster\n";
            }
        }
    }

    if (Visualize) std::cout << "The number of shift in this optimization process: " << NumShift << "\n";
    return NumShift;
/*
            bool ShiftFlag = true;
            if (ShiftClusterCost > OriginalClusterCost){
                ShiftFlag = false;
            }
            else{
                // Need to define the loss function, if the loss decrease, then do the shift, else we recover the assignment
                // We use a parameter *prop* to decide the relative importance between distance and the search cost, the larger prop means we consider more on the distance

                float OriginalNNDist = NeighborClusterDist[NN * NeighborNum]; 
                assert(OriginalNNDist >= 0);
                assert(OriginalClusterCost > 0);
                float ShiftNNDist = faiss::fvec_L2sqr(TrainSet + NN * Dimension, Centroids +  AssignmentID[i] * Dimension, Dimension);
                if(Visualize) std::cout << "ShiftNNDist: " << ShiftNNDist << " OriginNNDist: " << OriginalNNDist << " ShiftVectorCost: " << ShiftClusterCost << " OriginalVectorCost: " << OriginalClusterCost <<  "\n";
                // The smaller prop means more shift cases
                if (((prop *  (ShiftNNDist - OriginalNNDist)) / OriginalNNDist) > (float (OriginalClusterCost - ShiftClusterCost) / OriginalClusterCost)){
                    ShiftFlag = false;
                }
                else{
                    AssignmentDist[NN] = ShiftNNDist;
                }
            }
            //std::cout << "Check whether to update\n";
*/
 }


/*

            if (UseQuantize){
                std::map<std::pair<uint32_t, uint32_t>, uint32_t> VisitedQuantizationMap;

                for (uint32_t j = 0; j < EfSearch; j++){
                    // Use each cluster as target cluster
                    uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + j];

                    if (NewNeighborList.find(TargetClusterID) != NewNeighborList.end()){
                        for (uint32_t k = 1; k < EfSearch; k++){
                            uint32_t NNClusterID = QCID[QueryIdx * EfSearch + k];

                            if (NewNeighborList[TargetClusterID].find(NNClusterID) != NewNeighborList[TargetClusterID].end()){
                                float QueryNLDist = FetchQueryNLDist(QCDist[QueryIdx * EfSearch + j], QCDist[QueryIdx * EfSearch + k], CentroidsDist[TargetClusterID * nc + NNClusterID], NewNeighborList[TargetClusterID][NNClusterID].second[0]);
                                NLQueueList.emplace(std::make_pair(std::make_pair(j, k), QueryNLDist));
                            }
                        }
                    }
                }

                while(!NLQueueList.empty()){
                    if (VisitedItem > MaxItem){break;} 

                    uint32_t TargetClusterIndice = NLQueueList.top().first.first; 
                    uint32_t NNClusterIndice = NLQueueList.top().first.second;
                    float Dist = NLQueueList.top().second;
                    NLQueueList.pop();

                    uint32_t TargetClusterID = QCID[QueryIdx * EfSearch + TargetClusterIndice]; 
                    uint32_t NNClusterID = QCID[QueryIdx * EfSearch + NNClusterIndice];
                    uint32_t QuantizationIndice = 0;
                    size_t NeighborListSize = QuantBatchSize;

                    if (VisitedQuantizationMap.find(std::make_pair(TargetClusterID, NNClusterID)) != VisitedQuantizationMap.end()){
                        QuantizationIndice = VisitedQuantizationMap[std::make_pair(TargetClusterID, NNClusterID)];
                    }
                    else{
                        QuantizationIndice = 1;
                    }

                    if (QuantizationIndice < NewNeighborList[TargetClusterID][NNClusterID].first[1]){
                        VisitedQuantizationMap[std::make_pair(TargetClusterID, NNClusterID)] = QuantizationIndice + 1;
                        float QueryNLDist = FetchQueryNLDist(QCDist[QueryIdx * EfSearch + TargetClusterIndice], QCDist[QueryIdx * EfSearch + NNClusterIndice], CentroidsDist[TargetClusterID * nc + NNClusterID], NewNeighborList[TargetClusterID][NNClusterID].second[QuantizationIndice]);
                        NLQueueList.emplace(std::make_pair(std::make_pair(TargetClusterIndice, NNClusterIndice), QueryNLDist));
                    }
                    else{
                        NeighborListSize = NewNeighborList[TargetClusterID][NNClusterID].first[0] - QuantBatchSize * (QuantizationIndice - 1);
                    }

                    float CDist = QCDist[QueryIdx * EfSearch + NNClusterIndice] - CNorms[NNClusterID];

                    uint32_t Start_Indice = 2 + (QuantizationIndice - 1) * QuantBatchSize;
                    for (size_t temp = Start_Indice; temp < Start_Indice + NeighborListSize; temp++){

                        uint32_t VectorID = NewNeighborList[TargetClusterID][NNClusterID].first[temp];

                        if (VisitedVecSets[NNClusterIndice].count(VectorID) == 0){
                            VisitedVecSets[NNClusterIndice].insert(VectorID);
                            VisitedItem ++;

                            float VNorm = BaseNorms[VectorID];
                            float ProdDist = 0;

                            for (size_t k = 0; k < PQ->code_size; k++){
                                ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodes[VectorID * PQ->code_size + k]];
                            }

                            float Dist = CDist + VNorm - 2 * ProdDist;

                            if (Dist < QueryDists[QueryIdx * K]){
                                faiss::maxheap_pop(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K);
                                faiss::maxheap_push(K, QueryDists + QueryIdx * K, QueryIds + QueryIdx * K, Dist, VectorID);
                            }
                        }
                    }
                }
            }
            else{
                */