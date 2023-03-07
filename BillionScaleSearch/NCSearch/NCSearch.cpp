#include "./NCSearch.h"


// This is the function for learning the number of clusters as well as the centroids setting
// Our analysis: the cost function for each vector is: alpha * log(M) + (Zi + Li), here Zi is the size of cluster with vector i, Li is the neighbor vectors to be visited
// By summing up all vectors, we get the final estimation result: alpha * N * log(M) + sum[c](Zc^2 + Lc), here Lc is the whole number of neighbor vectors of cluster c
// The upper bound of Lc is Zc * MaxNCLZ, but only a small fraction of vectors will touch a small fraction of neighbor vectors, we use this as a parameter K
// The search cost is: alpha * N * log(M) + sum[c](Zc^2 + K * Zc) = alpha * N * log(M) + sum[c](Zc^2) + K * N, sum[c](Zc) = N
// Pursuing the minimum cost: Z1 = Z2 = ... = ZM = (N / M), the cost: alpha * N * log(M) + (N^2 / M) + K * N, by differetiate: M = N / alpha
// We continually split the cluster with largest search cost. Estimation on the search cost: inner cost: Zc^2, neighbor cost: we get the vectors locate close enough.
// The distance threshold is the avg dist between query and groundtruth.
// Each cluster is splited to (SC) / (AVG SC). We continue the split until the search cost doesnot increase

// The NC Search for inverted index
// The parameter search cost: F(M) =   
uint32_t HeuristicNCList(size_t nb, float alpha, float DistBound, float * TrainSet, float * Centroids, size_t Dimension, size_t CentrainSize, 
                        bool verbose, bool optimize, size_t BoundSize, size_t CheckBatch, size_t MaxM, size_t NLevel, size_t OptSize, float Lambda){
    /*The Input Parameter*/
    // bool  AddiFunc;
    // bool  ControlStart;
    /*                   */

    uint32_t M = std::floor(float(nb) / alpha);
    float Scale = float(nb) / CentrainSize;
    if (verbose){std::cout << "Initialized with " << M << " clusters \n";}

    // Train the initial centroids;
    hierarkmeans(TrainSet, Dimension, CentrainSize, M, Centroids, NLevel, optimize, OptSize, Lambda);

    // Compute the distance between train vectors and the centroids
    std::vector<float> Distances(BoundSize * CentrainSize, 0);
    std::vector<int64_t> Labels(BoundSize * CentrainSize, 0);
    std::vector<std::vector<uint32_t>> NeighborVectorID(M); NeighborVectorID.reserve(MaxM);
    std::vector<std::vector<uint32_t>> ClusterList(M); ClusterList.reserve(MaxM);


    // Todo: If this is too slow, we capture the results from previous training
    // Todo: Check whether faiss knn_L2sqr are in ascending order
    faiss::float_maxheap_array_t res = {size_t(CentrainSize), size_t(BoundSize), Labels.data(), Distances.data()};
    faiss::knn_L2sqr (TrainSet, Centroids, Dimension, CentrainSize, M, &res);


    // Initialize the neighbor vectors of the clusters for neighbor cost estimation
    for (uint32_t i = 0; i < CentrainSize; i++){
        ClusterList[Labels[i * BoundSize]].emplace_back(i);
        for (size_t j = 1; j < BoundSize; j++){
            if (Distances[i * BoundSize + j] - Distances[i * BoundSize] < DistBound){
                // The other cluster will consider i as a neighbor vector
                NeighborVectorID[Labels[i * BoundSize + j]].emplace_back(i);
            }
        }
    }

    // Maintain a list with the VCDist, free other data
    std::vector<float> VCDist(CentrainSize);
    for (size_t i = 0; i < CentrainSize; i++){VCDist[i] = Distances[i * BoundSize];}
    std::vector<float>().swap(Distances);
    std::vector<int64_t>().swap(Labels);


    // Initialize the vector cost, for each cluster, the cost is the sum of inner cost (cluster size) and neighbor cost (number of neighbor vectors)
    double SumVectorCost = 0;
    auto comp = [](std::pair<float, uint32_t> Element1, std::pair<float, uint32_t> Element2){return Element1.first < Element2.first;};
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(comp)> ClusterCost(comp);

    for (uint32_t i = 0; i < M; i++){
        double VectorCost = Scale * Scale * ClusterList[i].size() * ClusterList[i].size() +  Scale * NeighborVectorID[i].size();
        ClusterCost.emplace(std::make_pair(VectorCost, i));
        SumVectorCost += VectorCost;
    }

    bool NextSplit = M + CheckBatch < MaxM ? true : false;
    double AvgVectorCost = SumVectorCost / M;

    if (verbose){std::cout << "Start spliting the clusters with largest cost\n";}
    while(NextSplit){
        uint32_t OriginM = M;
        double CostDiff = 0;
        std::vector<uint32_t> ClusterIDBatch(CheckBatch, 0);
        std::vector<double> ClusterCostBatch(CheckBatch, 0);
        for (size_t i = 0; i < CheckBatch; i++){
            ClusterIDBatch[i] = ClusterCost.top().second;
            ClusterCostBatch[i] = ClusterCost.top().first;
            std::cout << ClusterCostBatch[i] << " ";
            CostDiff -= ClusterCostBatch[i];
            ClusterCost.pop();
        }
        std::cout << std::endl;
        std::vector<uint32_t> SplitM(CheckBatch);
        std::vector<std::vector<float>> SplitCentroids(CheckBatch);
        std::vector<std::vector<std::vector<uint32_t>>> SplitClusterList(CheckBatch);
        std::vector<std::vector<std::vector<uint32_t>>> SplitNeighborVectorID(CheckBatch);
        //std::cout << "Process the clusters\n";
//#pragma omp parallel for
        for (size_t i = 0; i < CheckBatch; i++){
            // Train Split Centroids
            SplitM[i] = std::ceil(std::sqrt(ClusterCostBatch[i] / AvgVectorCost));

            SplitCentroids[i].resize(SplitM[i] * Dimension);
            SplitClusterList[i].resize(SplitM[i]);
            SplitNeighborVectorID[i].resize(SplitM[i]);

            size_t SplitTrainSize = ClusterList[ClusterIDBatch[i]].size();
            std::vector<float> SplitTrainSet(Dimension * SplitTrainSize);
            for (size_t j = 0; j < SplitTrainSize; j++){
                memcpy(SplitTrainSet.data() + j * Dimension, TrainSet + ClusterList[ClusterIDBatch[i]][j] * Dimension, Dimension * sizeof(float));
            }
            optkmeans(SplitTrainSet.data(), Dimension, SplitTrainSize, SplitM[i], SplitCentroids[i].data(), false, false, optimize, Lambda, OptSize);
            //std::cout << "Get the vector distance \n"; 

            // Assign vectors locally in cluster
            std::vector<int64_t> SplitLabels(SplitM[i] * SplitTrainSize, 0);
            std::vector<float> SplitDists(SplitM[i] * SplitTrainSize, 0);
            faiss::float_maxheap_array_t SplitRes = {size_t(SplitTrainSize), size_t(SplitM[i]), SplitLabels.data(), SplitDists.data()};
            faiss::knn_L2sqr (SplitTrainSet.data(), SplitCentroids[i].data(), Dimension, SplitTrainSize, SplitM[i], &SplitRes);
            // Compute the local ID and the change in neighbor search cost
            //std::cout << "Assign the vectors\n";
            for (size_t j = 0; j < SplitTrainSize; j++){
                SplitClusterList[i][SplitLabels[j * SplitM[i]]].emplace_back(ClusterList[ClusterIDBatch[i]][j]);
                VCDist[ClusterList[ClusterIDBatch[i]][j]] = SplitDists[j * SplitM[i]];
                for (size_t k = 1; k < SplitM[i]; k++){
                    // Update the neighbor vector info from the inner vectors
                    if (SplitDists[j * SplitM[i] + k] - SplitDists[j * SplitM[i]] < DistBound){
                        SplitNeighborVectorID[i][SplitLabels[j * SplitM[i] + k]].emplace_back(ClusterList[ClusterIDBatch[i]][j]);
                    }
                }
            }

            //std::cout<< "Update the splited clusters\n";
            // Update the neighbor vector cost of the original large dataset
            // Only consider the neighbor vectors of splited cluster
            faiss::float_maxheap_array_t VectorRes = {1, size_t(SplitM[i]), SplitLabels.data(), SplitDists.data()};
            for (size_t j = 0; j < NeighborVectorID[ClusterIDBatch[i]].size(); j++){
                uint32_t VectorID = NeighborVectorID[ClusterIDBatch[i]][j];
                faiss::knn_L2sqr(TrainSet + VectorID * Dimension, SplitCentroids[i].data(), Dimension, 1, SplitM[i], &VectorRes);
                for (size_t k = 0; k < SplitM[i]; k++){
                    if (SplitDists[k] - VCDist[VectorID] < DistBound){
                        SplitNeighborVectorID[i][SplitLabels[k]].emplace_back(VectorID);
                    }
                }
            }
        }


        // Update the split results: we replace the original centroid with the first generated centroid, and place the others at last
        // Update the M, centroids, neighborvectorID, clusterlist, clustercost, costdiff
        for (size_t i = 0; i < CheckBatch; i++){
            //std::cout << "Update the cluster list " << i << " " << CheckBatch <<  " " << SplitM[i] << std::endl;;
            //std::cout << "Copy the centroids\n";
            memcpy(Centroids + ClusterIDBatch[i] * Dimension, SplitCentroids[i].data() + 0 * Dimension, Dimension * sizeof(float));
            //std::cout << "Swap the neighbor ID\n";
            std::vector<uint32_t>().swap(NeighborVectorID[ClusterIDBatch[i]]); 
            //std::cout << "Resize the neighbor ID" << SplitNeighborVectorID[i][0].size() << std::endl;
            NeighborVectorID[ClusterIDBatch[i]].resize(SplitNeighborVectorID[i][0].size());
            //std::cout << "Copy the neighbor ID\n";
            memcpy(NeighborVectorID[ClusterIDBatch[i]].data(), SplitNeighborVectorID[i][0].data(), SplitNeighborVectorID[i][0].size() * sizeof(uint32_t));
            //std::cout << "Copy the cluster list" << SplitClusterList[i][0].size() << std::endl;
            std::vector<uint32_t>().swap(ClusterList[ClusterIDBatch[i]]); ClusterList[ClusterIDBatch[i]].resize(SplitClusterList[i][0].size());
            memcpy(ClusterList[ClusterIDBatch[i]].data(), SplitClusterList[i][0].data(), SplitClusterList[i][0].size() * sizeof(uint32_t));
            //std::cout << "Update the cost\n";
            double VectorCost = Scale * SplitClusterList[i][0].size() * Scale * SplitClusterList[i][0].size() + Scale * SplitNeighborVectorID[i][0].size();
            ClusterCost.emplace(std::make_pair(VectorCost, ClusterIDBatch[i]));
            CostDiff += VectorCost;

            for (size_t j = 1; j < SplitM[i]; j++){
                //std::cout << "Enhance the cluster list size " << j << " " << SplitM[i] << std::endl;;
                memcpy(Centroids + M * Dimension, SplitCentroids[i].data() + j * Dimension, Dimension * sizeof(float));
                NeighborVectorID.resize(M + 1); NeighborVectorID[M].resize(SplitNeighborVectorID[i][j].size());
                memcpy(NeighborVectorID[M].data(), SplitNeighborVectorID[i][j].data(), SplitNeighborVectorID[i][j].size() * sizeof(uint32_t));

                ClusterList.resize(M + 1); ClusterList[M].resize(SplitClusterList[i][j].size());
                memcpy(ClusterList[M].data(), SplitClusterList[i][j].data(), SplitClusterList[i][j].size() * sizeof(uint32_t));

                double VectorCost = Scale * SplitClusterList[i][j].size() * Scale * SplitClusterList[i][j].size() + Scale * SplitNeighborVectorID[i][j].size();
                ClusterCost.emplace(std::make_pair(VectorCost, M));
                CostDiff += VectorCost;
                M++;
            }
        }
        std::cout << "Cost Diff: " <<CostDiff << " M: " << M << "\n";
        SumVectorCost += CostDiff;
        CostDiff += alpha * (std::log(float(M) / OriginM)) * nb;
        std::cout << "Cost Diff: " <<CostDiff;
        AvgVectorCost = SumVectorCost / M;
        if (CostDiff >= 0 || M + CheckBatch > MaxM){
            NextSplit = false;
        }
    }
    return M;
}

/*
 * zlib License
 *
 * Regularized Incomplete Beta Function
 *
 * Copyright (c) 2016, 2017 Lewis Van Winkle
 * https://codeplea.com/incomplete-beta-function-c
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

double incbeta(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return 1.0/0.0;

    /*The continued fraction converges nicely for x < (a+1)/(a+b+2)*/
    if (x > (a+1.0)/(a+b+2.0)) {
        return (1.0-incbeta(b,a,1.0-x)); /*Use the fact that beta is symmetrical.*/
    }

    /*Find the first part before the continued fraction.*/
    const double lbeta_ab = lgamma(a)+lgamma(b)-lgamma(a+b);
    const double front = exp(log(x)*a+log(1.0-x)*b-lbeta_ab) / a;

    /*Use Lentz's algorithm to evaluate the continued fraction.*/
    double f = 1.0, c = 1.0, d = 0.0;

    int i, m;
    for (i = 0; i <= 200; ++i) {
        m = i/2;

        double numerator;
        if (i == 0) {
            numerator = 1.0; /*First numerator is 1.0.*/
        } else if (i % 2 == 0) {
            numerator = (m*(b-m)*x)/((a+2.0*m-1.0)*(a+2.0*m)); /*Even term.*/
        } else {
            numerator = -((a+m)*(a+b+m)*x)/((a+2.0*m)*(a+2.0*m+1)); /*Odd term.*/
        }

        /*Do an iteration of Lentz's algorithm.*/
        d = 1.0 + numerator * d;
        if (fabs(d) < TINY) d = TINY;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (fabs(c) < TINY) c = TINY;

        const double cd = c*d;
        f *= cd;

        /*Check for stop.*/
        if (fabs(1.0-cd) < STOP) {
            return front * (f-1.0);
        }
    }

    return 1.0/0.0; /*Needed more loops, did not converge.*/
}

// The volumn of a cap in n-dimensional space
// See the link https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres

double CapVolumn(double n, double r, double a){
            
    if (a >= 0){
        //std::cout << "Cap value: " <<  " " << n << " " << r << " " << a << " " << std::pow(M_PI, n / 2) << " " << 
        //std::tgamma((n/2 + 1)) << " " << std::pow(r, n/2)  << " " << incbeta((n+1)/2, 0.5, 1-((a*a)/(r*r))) << "\n";

        return (0.5 * std::pow(M_PI, n / 2) * (std::pow(r, n/2) * incbeta((n+1)/2, 0.5, 1-((a*a)/(r*r)))) * (std::pow(r, n/2) / std::tgamma(n/2 + 1)));
    }
    else{
        return ((std::pow(r, n/2) / std::tgamma(n/2 + 1)) * std::pow(M_PI, n / 2) * std::pow(r, n/2)  - CapVolumn(n, r, -a));
    }
}

double SphereVolumn(double n, double r){
    
    //std::cout << "Volumn value: " << n << " " << std::tgamma(n/2 + 1) << "\n";
    return (std::pow(r, n / 2) / std::tgamma(n/2 + 1) * std::pow(M_PI, n / 2) * std::pow(r, n / 2));
}

float ComputeRadius(uint32_t * ClusterList, uint32_t ClusterSize, float * TrainsetDist, float RadiusProp){
    assert(ClusterSize > 0);
    size_t ClusterRadiusNum = std::ceil(ClusterSize * RadiusProp);
    std::priority_queue<float> RaduisQueue;
    for (size_t j = 0; j < ClusterSize; j++){
        if(RaduisQueue.size() < ClusterRadiusNum){
            RaduisQueue.emplace(TrainsetDist[ClusterList[j]]);
        }
        else if (TrainsetDist[ClusterList[j]] < RaduisQueue.top())
        {
            assert(!RaduisQueue.empty());  RaduisQueue.pop(); RaduisQueue.emplace(TrainsetDist[ClusterList[j]]);
        }
    }
    return RaduisQueue.top();
}

double NeighborVectorCost(size_t Dimension, float * TrainVector, float * Centroid, size_t NeighborSize, float NeighborRadius, float ClusterDist, float VectorDist, float BoundDist){
    double NeighborVectorProp;
    if (std:: sqrt(VectorDist) + std::sqrt(BoundDist) < std::sqrt(ClusterDist) - std::sqrt(NeighborRadius)){
        return 0;
    }
    double d = faiss::fvec_L2sqr(TrainVector, Centroid, Dimension);
    NeighborVectorProp = (BoundDist + NeighborRadius - d) / BoundDist;
    //std::cout << "BoundDist: " << BoundDist << " NieghborRadius: " << NeighborRadius << " d: " << d << " NeighborVectorProp: " << NeighborVectorProp << " NeighborSize:" << NeighborSize << "\n";
    if (NeighborVectorProp <= 0){
        return 0;
    }
    else if (0.0 < NeighborVectorProp  && NeighborVectorProp < 1.0){
        return NeighborVectorProp * NeighborSize;
    }
    else{
        return NeighborSize;
    }

    /*
    if (NeighborRadius + BoundDist < d){
        return 0;
    }
    // The whole sphere2 is in sphere1
    else if (d + NeighborRadius <= BoundDist)
    {
        NeighborVectorCost = NeighborSize; 
    }
    else if(d + BoundDist <= NeighborRadius){
        std::cout << "\n Assignment Error: Should assigned to the neighbor cluster " << d << " " << BoundDist << " " << NeighborRadius;
    }
    else if(BoundDist + NeighborRadius > d){

        double c1 = (d * d + BoundDist * BoundDist - NeighborRadius * NeighborRadius) / (2 * d); 
        double c2 = (d * d - BoundDist * BoundDist + NeighborRadius * NeighborRadius) / (2 * d);
        double InterSec1 = CapVolumn(Dimension, BoundDist, c1);
        double InterSec2 = CapVolumn(Dimension, NeighborRadius, c2);
        std::cout << "Intersection: d: " << d * d << " BoundDist: " << BoundDist * BoundDist << " Radius: " << NeighborRadius * NeighborRadius << "\n";
        NeighborVectorCost = (InterSec1 + InterSec2) / Volumn * NeighborSize;
    }
    return NeighborVectorCost;
    */
}


// Check the average search cost of vectors in each cluster
// For acceleration, we only compute a subset of vectors in each cluster.
// 
void UpdateCandidateListSize(size_t M, size_t Dimension, size_t Scale, std::vector<std::vector<uint32_t>>& ClusterList, 
                            hnswlib::HierarchicalNSW * Graph, float * TrainSet, float ClusterNum, float * ClusterVectorCost){
    long RandomSeed = 1234;

    // The proportion of checked vectors
    float CheckProp = M < 10e6 ? 0.02 : 0.01;
    size_t CheckLowerBound = 20;
    std::cout << "Candidate list size with Number of cluster visited: " << ClusterNum <<  " The clusterVectorCost with checkprop: " << CheckProp <<"\n";;

#pragma omp parallel for
    for (size_t i = 0; i < M; i++){
        size_t ClusterSize = ClusterList[i].size();

        std::vector<int> RandomId(ClusterSize);
        faiss::rand_perm(RandomId.data(), ClusterSize, RandomSeed+1);
        size_t SampleClusterSize = std::ceil(CheckProp * ClusterSize) > CheckLowerBound ? std::ceil(CheckProp * ClusterSize) : CheckLowerBound < ClusterSize ? CheckLowerBound : ClusterSize;
        size_t CandidateListSize = 0;

        for (size_t VectorIndice = 0; VectorIndice < SampleClusterSize; VectorIndice++){
            std::vector<uint32_t> VectorLabel(ClusterNum);
            std::vector<float> VectorDist(ClusterNum);

            uint32_t VectorID = ClusterList[i][RandomId[VectorIndice]];
            auto result = Graph->searchKnn(TrainSet + VectorID * Dimension, ClusterNum);
            for (size_t j = 0; j < ClusterNum; j++){
                VectorLabel[ClusterNum - j - 1] = result.top().second;
                VectorDist[ClusterNum - j - 1] = result.top().first;
                result.pop();
            }

            for (size_t j = 0; j < ClusterNum; j++){
                CandidateListSize += float(ClusterSize * ClusterList[VectorLabel[j]].size()) / SampleClusterSize;
            }
        }
        ClusterVectorCost[i] = Scale * Scale * CandidateListSize;
    }

/*
    float AvgClusterVectorCost = 0;
    for (size_t i = 0; i < M; i++){
        AvgClusterVectorCost += ClusterVectorCost[i];
    }
    std::cout << "Check Prop: " << CheckProp << " Average Cluster Cost: " << AvgClusterVectorCost  / M << "\n";
*/
}

// The return result: AchieveTargetRecall, ClusterNum, CandidatelistSize, Centime, Vetime

std::tuple<bool, size_t, float, float, float> UpdateVisitedClusters(std::ofstream & RecordFile, size_t RecallK, size_t nq, size_t PQ_M, size_t nbits, size_t SubTrainSize, size_t PQ_TrainSize, float & TargetRecall, float MaxCandidateSize, size_t Dimension,
                    uint32_t * TrainsetLabel, int * SubsetID, float * TrainSet, int64_t * QueryGtLabel, hnswlib::HierarchicalNSW * Graph, float * QuerySet,
                    float M,  std::vector<std::vector<uint32_t>> & SubClusterList){

    time_recorder TRecorder = time_recorder();

    std::vector<std::unordered_set<uint32_t>> GtSets(nq);

    for (size_t i = 0; i < nq; i++){
        for (size_t j = 0; j < RecallK; j++){
            GtSets[i].insert(QueryGtLabel[RecallK * i + j]);
        }
    }
    // Compute the residual of sub trainset
    std::vector<float> SubResidual(SubTrainSize * Dimension, 0);
#pragma omp parallel for
    for (size_t i =0; i < SubTrainSize; i++){
        uint32_t ClusterID = TrainsetLabel[SubsetID[i]];
        faiss::fvec_madd(Dimension, TrainSet + SubsetID[i] * Dimension, -1.0,  Graph->getDataByInternalId(ClusterID), SubResidual.data() + i * Dimension);
    }
    faiss::ProductQuantizer * PQ = new faiss::ProductQuantizer(Dimension, PQ_M, nbits);

    // Train the PQ quantizer with a small subset
    PQ->verbose = true;
    PQ_TrainSize = PQ_TrainSize < SubTrainSize ? PQ_TrainSize : SubTrainSize;
    PQ->train(PQ_TrainSize, SubResidual.data());
    TRecorder.print_time_usage("PQ quantizer training completed");

    std::vector<uint8_t> SubPQCodes(SubTrainSize * PQ->code_size);
    PQ->compute_codes(SubResidual.data(), SubPQCodes.data(), SubTrainSize);
    std::vector<float> RecoveredSubResidual(SubTrainSize * Dimension);
    PQ->decode(SubPQCodes.data(), RecoveredSubResidual.data(), SubTrainSize);
    std::vector<float> RecoveredVector(SubTrainSize * Dimension);

#pragma omp parallel for
    for (size_t i = 0; i < SubTrainSize; i++){
        uint32_t ClusterID = TrainsetLabel[SubsetID[i]];
        faiss::fvec_madd(Dimension, Graph->getDataByInternalId(ClusterID), 1.0, RecoveredSubResidual.data() + i * Dimension, RecoveredVector.data() + i * Dimension);
    }
    TRecorder.print_time_usage("Vector quantization completed");
    std::vector<float> SubsetNorm(SubTrainSize);
    std::vector<float> CenNorm(Graph->maxelements_);
    faiss::fvec_norms_L2sqr(SubsetNorm.data(), RecoveredVector.data(), Dimension, SubTrainSize);
#pragma omp parallel for
    for (size_t i = 0; i < Graph->maxelements_; i++){
        CenNorm[i] = faiss::fvec_norm_L2sqr(Graph->getDataByInternalId(i), Dimension);
    }
    TRecorder.print_time_usage("Vector norm and centroid norm computation completed");


    float ResultIdx = -1;
    bool ValidLine = true;
    while(ValidLine){

        std::vector<int64_t> ResultID(RecallK * nq, 0);
        std::vector<float> ResultDist(RecallK * nq, 0);
        std::vector<float> PQTable(nq * PQ->ksub * PQ->M, 0);
        float VisitedVec = 0;
        float VisitedGt = 0;
        bool AchieveTargetRecall = true;

        bool UpdateClusterNum = true;
        bool IncreaseClusterNum = false;
        bool DecreaseClusterNum = false;
        size_t MaxRepeatTimes = 3;
        float ClusterNum = size_t(M / 200);
        size_t ClusterBatch = std::ceil(float(ClusterNum) / 10);
        float MinimumCoef = 0.98;

        // We want the search time on centroids and vectors change with the clusternum, to against the search load
        size_t PreviousClusterNum = ClusterNum;
        float PreviousRecordTime1 = 0;
        float PreviousRecordTime3 = 0;
        size_t RepeatTimes = 0;
        std::vector<float> ClusterNumList;
        std::vector<float> CanLengthList;
        std::vector<float> CenSearchTime;
        std::vector<float> VecSearchTime;

        while(UpdateClusterNum){
            
            std::vector<uint32_t> QueryLabel(nq * ClusterNum);
            std::vector<float> QueryDist(nq * ClusterNum);
            TRecorder.reset();
        
            
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                //std::cout << "1: \n";

                auto result = Graph->searchBaseLayer(QuerySet + QueryIdx * Dimension, ClusterNum);
                for (size_t i = 0; i < ClusterNum; i++){
                    QueryLabel[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().second;
                    QueryDist[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().first;
                    result.pop();
                }
                //std::cout << "2: \n";
            }
            
            TRecorder.recordTimeConsumption1();

            TRecorder.reset();
            //std::cout << "3: \n";

            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx ++){
                PQ->compute_inner_prod_table(QuerySet + QueryIdx * Dimension, PQTable.data() + QueryIdx * PQ->ksub * PQ->M);
            }
            TRecorder.recordTimeConsumption2();
            TRecorder.reset();
            //std::cout << "4: \n";
            
            VisitedVec = 0;
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                faiss::maxheap_heapify(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK);
                for (size_t i = 0; i < ClusterNum; i++){
                    uint32_t ClusterID = QueryLabel[QueryIdx * ClusterNum + i];
                    float CDist = QueryDist[QueryIdx * ClusterNum + i] - CenNorm[ClusterID];
                    VisitedVec += SubClusterList[ClusterID].size();
                    //std::cout << "5: \n";

                    for (size_t j = 0; j < SubClusterList[ClusterID].size(); j++){
                        //std::cout << "5.1: " << QueryIdx << " " << ClusterID << " " << j << " " << SubClusterList[ClusterID].size() << "\n";
                        uint32_t VectorID = SubClusterList[ClusterID][j];
                        //std::cout << "5.2: \n";
                        float VNorm = SubsetNorm[VectorID];
                        //std::cout << "5.3: \n";
                        float ProdDist = 0;
                        //std::cout << "6: \n";
                        for (size_t k = 0; k < PQ->code_size; k++){
                            ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + SubPQCodes[VectorID * PQ->code_size + k]];
                        }
                        float Dist = CDist + VNorm - 2 * ProdDist;
                        //std::cout << "7: \n";

/*
                        std::cout << "Correct QC Dist: " << faiss::fvec_L2sqr(QuerySet + QueryIdx * Dimension, Graph->getDataByInternalId(ClusterID), Dimension) << " Result: " << QueryDist[QueryIdx * ClusterNum + i] << std::endl;
                        std::cout << "Correct CNorm: " << faiss::fvec_norm_L2sqr(Graph->getDataByInternalId(ClusterID), Dimension) << " Result: " << CenNorm[ClusterID] << std::endl; 
                        std::vector<float> RecoveredResidual(Dimension);
                        PQ->decode(SubPQCodes.data() + VectorID * PQ->code_size, RecoveredResidual.data(),1);
                        float CorrectProdDist = faiss::fvec_inner_product(RecoveredResidual.data(), QuerySet + QueryIdx * Dimension, Dimension);
                        std::cout << "Correct Prod Dist: " << CorrectProdDist << " Result ProdDist: " << ProdDist << std::endl;
                        std::vector<float> OPQBase(Dimension);
                        faiss::fvec_madd(Dimension, RecoveredResidual.data(), 1.0, Graph->getDataByInternalId(ClusterID), OPQBase.data());
                        float CorrectBaseNorm = faiss::fvec_norm_L2sqr(OPQBase.data(), Dimension);
                        float CorrectResultDist = faiss::fvec_L2sqr(OPQBase.data(), QuerySet + QueryIdx * Dimension, Dimension);
                        std::cout << "Precise Base Norm " << faiss::fvec_norm_L2sqr(TrainSet + SubsetID[VectorID] * Dimension, Dimension) << " Recorded Base Norm " << VNorm <<  " Correct Base Norm: " << CorrectBaseNorm << std::endl;
                        std::cout << "Precise Dist " << faiss::fvec_L2sqr(QuerySet + QueryIdx * Dimension, TrainSet + SubsetID[VectorID] * Dimension, Dimension) << " Result Dist: " << Dist <<  " Correct Result Dist: " << CorrectResultDist << std::endl;
*/
                        if (Dist < ResultDist[QueryIdx * RecallK]){
                            faiss::maxheap_pop(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK);
                            faiss::maxheap_push(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK, Dist, VectorID);
                            //std::cout << "8: \n";
                        }
                    }
                }
            }
            //std::cout << "9: \n";

            TRecorder.recordTimeConsumption3();

            
            if (PreviousClusterNum < ClusterNum && RepeatTimes < MaxRepeatTimes){
                if (TRecorder.TempDuration1 < PreviousRecordTime1 || TRecorder.TempDuration3 < PreviousRecordTime3){
                    RepeatTimes ++;
                    continue;
                }
            }
            else if (PreviousClusterNum > ClusterNum && RepeatTimes < MaxRepeatTimes){
                if (TRecorder.TempDuration1 > PreviousRecordTime1 || TRecorder.TempDuration3 > PreviousRecordTime3){
                    RepeatTimes ++;
                    continue;
                }
            }

            VisitedGt = 0;
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                for (size_t i = 0; i < RecallK; i++){
                    if (GtSets[QueryIdx].count(ResultID[QueryIdx * RecallK + i]) != 0){
                        VisitedGt ++;
                    }
                }
            }

            std::cout << "'Numcluster: " << ClusterNum << " Cluster Batch: " << ClusterBatch << " Visited num of GT in top " << RecallK << " : " << (VisitedGt) / nq << " Candidate List Size: " << (VisitedVec) / nq <<  " / " << SubTrainSize <<  " Target recall: "<< TargetRecall  << " Search Time: " << (TRecorder.TempDuration1 + TRecorder.TempDuration2 + TRecorder.TempDuration3) / (nq * 1000) <<  " Cen Search Time: " <<  TRecorder.TempDuration1 / (nq * 1000)  << " Table time: " <<  TRecorder.TempDuration2 / (nq * 1000) << " Vec Search Time: " << TRecorder.TempDuration3 / (nq * 1000)  << " with repeat times: " << RepeatTimes << " / " << MaxRepeatTimes << "',\n";
            RecordFile << "'Numcluster: " << ClusterNum << " Cluster Batch: " << ClusterBatch << " Visited num of GT in top " << RecallK << " : " << (VisitedGt) / nq << " Candidate List Size: " << (VisitedVec) / nq <<  " / " << SubTrainSize <<  " Target recall: "<< TargetRecall  << " Search Time: " << (TRecorder.TempDuration1 + TRecorder.TempDuration2 + TRecorder.TempDuration3) / (nq * 1000) << " Cen Search Time: " <<  TRecorder.TempDuration1 / (nq * 1000) << " Table time: " <<  TRecorder.TempDuration2 / (nq * 1000) << " Vec Search Time: " << TRecorder.TempDuration3 / (nq * 1000)  << " with repeat times: " << RepeatTimes << " / " << MaxRepeatTimes << "',\n";

            ClusterNumList.emplace_back(ClusterNum); CanLengthList.emplace_back((VisitedVec) / nq); CenSearchTime.emplace_back(TRecorder.TempDuration1 / (nq * 1000)); VecSearchTime.emplace_back(TRecorder.TempDuration3 / (nq * 1000));

            if ((VisitedVec) / nq > MaxCandidateSize){
                UpdateClusterNum = false;
                AchieveTargetRecall = false;
            }
            else if (VisitedGt / nq > TargetRecall){
                ResultIdx = ClusterNumList.size() - 1;
                if (IncreaseClusterNum){
                    if (ClusterBatch > 1){
                        ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                        ClusterBatch = std::ceil(float(ClusterBatch) / 10);
                        ClusterNum += ClusterBatch;
                    }
                    else{
                        UpdateClusterNum = false;
                    }
                }
                else{
                    DecreaseClusterNum = true;
                    ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                }
            }
            else if (VisitedGt / nq < TargetRecall){
                if (DecreaseClusterNum){
                    if (ClusterBatch > 1){
                        ClusterNum += ClusterBatch;
                        ClusterBatch = std::ceil(float(ClusterBatch) / 10);
                        ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                    }
                    else{
                        ClusterNum += 1;
                        UpdateClusterNum = false;
                    }
                }
                else{
                    IncreaseClusterNum = true;
                    ClusterNum += ClusterBatch;
                }
            }
            else{
                ResultIdx = ClusterNumList.size() - 1;
                UpdateClusterNum = false;
            }

            PreviousRecordTime1 = TRecorder.TempDuration1;
            PreviousRecordTime3 = TRecorder.TempDuration3;
            RepeatTimes = 0;
        }

        std::pair<float, std::pair<float, float>> Result1 =  LeastSquares(ClusterNumList, CenSearchTime);
        std::pair<float, std::pair<float, float>> Result2 = LeastSquares(CanLengthList, VecSearchTime);

        if (ResultIdx < 0){ResultIdx = CanLengthList.size() - 1;} assert(ClusterNumList[ResultIdx] == ClusterNum);
        float Centime = Result1.second.first * ClusterNumList[ResultIdx] + Result1.second.second;
        float Vectime = Result2.second.first * CanLengthList[ResultIdx] + Result2.second.second;
        std::cout << "The search time on centroids and vectors: |" << Centime << "| |" << Vectime << "|\n";
        if (Result1.first > MinimumCoef && Result2.first > MinimumCoef){
            return std::make_tuple(AchieveTargetRecall, ClusterNumList[ResultIdx], CanLengthList[ResultIdx], Centime, Vectime);
        }
        else{
            RecordFile << "Regression failed with " << Result1.first << " and " << Result2.first << ", repeat the time estimation process\n\n";
            std::cout << "Regression failed with " << Result1.first << " and " << Result2.first << ", repeat the time estimation process\n\n";
        }
    }
    delete PQ;
    return std::make_tuple(0, 0, 0, 0, 0);
}

// Todo: 
// 1. Maintain a small subset from the trainset to estimate the number of vectors to be visited and the search time of vectors
// 2. Shrink the checkbatch: just like the lr training in deep learning     : Reduce the training learning rate in NC and numcluster: reduce them by 1/10 until we reach the termination point
// 3. Make the search time record robust to machine overhead

// Target:
// Make the model robust to initial NC and checkbatch setting
// Ensure the result is correct: near-optimal results

// We update the number of centroids by computing the search cost 
// The search cost comes from two sides: the central cluster and the neighor cluster 
// 
uint32_t HeuristicINI(std::ofstream & RecordFile, size_t nb, size_t nq, size_t RecallK, float SubsetProp, size_t PQ_M, size_t nbits, size_t PQ_TrainSize, float * TrainSet, float * QuerySet, float * ResultCentroids, std::string PathCentroid,
std::string PathTrainsetLabel, size_t Dimension, size_t TrainSize, float TargetRecall, float MaxCandidateSize, bool verbose, bool optimize, size_t ClusterBoundSize, size_t CheckBatch, size_t MaxM, size_t IniM, size_t NLevel, size_t GraphM, size_t GraphEf, size_t OptSize, float Lambda){

    uint32_t MaxFailureTimes = 10000 / CheckBatch;

    //The ratio between the baseset and the trainset
    uint32_t M = IniM;
    float Scale = float(nb) / TrainSize;
    time_recorder TRecorder = time_recorder();

    if (verbose){ std::cout << "Initialized with " << M << " clusters\n";}
    float * Centroids = new float[MaxM * Dimension];

    std::vector<float> TrainsetDist(TrainSize, 0);
    std::vector<uint32_t> TrainsetLabel(TrainSize, 0);

    // Initialize the centroids
    std::cout << "Initializing the cluster centroids\n";
    hnswlib::HierarchicalNSW * Graph;

    if (exists(PathCentroid)){
        std::ifstream IniCentroidInput(PathCentroid, std::ios::binary);
        readXvec<float>(IniCentroidInput, Centroids, Dimension, M, true, true);
        std::cout << "Load pretrained centroids from " << PathCentroid << "\n";
    }
    else{
        hierarkmeans(TrainSet, Dimension, TrainSize, M, Centroids, NLevel, optimize, true, OptSize, Lambda, 30, true, TrainsetLabel.data(), TrainsetDist.data());
        std::ofstream IniCentroidOutput(PathCentroid, std::ios::binary);
        for (size_t i = 0; i < M; i++){
            IniCentroidOutput.write((char *) & Dimension, sizeof(uint32_t));
            IniCentroidOutput.write((char *) (Centroids + i * Dimension), Dimension * sizeof(float));
        }
        IniCentroidOutput.close();
    }
    Graph = new hnswlib::HierarchicalNSW(Dimension, M, GraphM, 2 * GraphM, GraphEf);
    for (size_t i = 0; i < M; i++){Graph->addPoint(Centroids + i * Dimension);}
    RecordFile << "Graph data: efConstruction " << Graph->efConstruction_ << " efSearch: " << Graph->efSearch << " M: " << Graph->maxM_ << " Num of nodes: " << Graph->maxelements_ << "\n";

    // Assign the train vector
    if (exists(PathTrainsetLabel)){
        std::ifstream IniLabelInput(PathTrainsetLabel, std::ios::binary);
        IniLabelInput.read((char *) (TrainsetLabel.data()), TrainSize * sizeof(uint32_t));
    }
    else{
#pragma omp parallel for
        for (size_t i = 0; i < TrainSize; i++){
            auto Result = Graph->searchKnn(TrainSet + i * Dimension, 1);
            TrainsetDist[i] = Result.top().first;
            TrainsetLabel[i] = Result.top().second;
        }
        std::ofstream IniLabelOutput(PathTrainsetLabel, std::ios::binary);
        IniLabelOutput.write((char *) TrainsetLabel.data(), TrainSize  * sizeof(uint32_t));
        IniLabelOutput.close();
    }

    // Update the centroids with cluster size
    TRecorder.print_time_usage("Cluster Centroids loaded");
    TRecorder.record_time_usage(RecordFile, "Cluster Centroids loaded");

    // The traininig vector assignment for inverted index
    std::vector<std::vector<uint32_t>> ClusterList(M);
    for (uint32_t i = 0; i < TrainSize; i++){
        assert(TrainsetLabel[i] < M);
        ClusterList[TrainsetLabel[i]].emplace_back(i);
    }

    float m = TrainSize / M; float diffprop = 0; float dist = 0;
    for (size_t i = 0; i < M; i++){
        diffprop += std::abs(ClusterList[i].size() - m);
    }
    for (size_t i = 0; i < TrainSize; i++){
        dist += TrainsetDist[i];
    }
    std::cout << "Distance between vectors and centroids in initialization:  " << dist / TrainSize << " Training Diff Prop: " << diffprop / M << " / " << m <<"\n";

    // Only use the subset for query parameter determination, the top SubTrainSize vectors are the subset
    uint32_t SubTrainSize = SubsetProp * TrainSize;
    std::vector<int> SubsetID(TrainSize);

    faiss::rand_perm(SubsetID.data(), TrainSize, 1234+1);

    std::vector<std::vector<uint32_t>> SubClusterList(M);
    for (uint32_t i = 0; i < SubTrainSize; i++){
        uint32_t VectorID = SubsetID[i];
        SubClusterList[TrainsetLabel[VectorID]].emplace_back(i);
    }

    TRecorder.print_time_usage("Sample the subset for training");
    TRecorder.record_time_usage(RecordFile, "Sample the subset for training");
    // Compute the Gt of queries in the trainset, used to estimate the candidate list with different nc setting
    std::vector<int64_t> QueryGtLabel(nq * RecallK);
    std::vector<float> QueryGtDist(nq * RecallK);

    // Check the GT in neighbor clusters

    ClusterBoundSize = ClusterBoundSize > M / 20 ? ClusterBoundSize : M / 20;
#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        auto result = Graph->searchKnn(QuerySet + i * Dimension, ClusterBoundSize);

        faiss::maxheap_heapify(RecallK, QueryGtDist.data() + i * RecallK, QueryGtLabel.data() + i * RecallK);
        for (size_t j = 0; j < ClusterBoundSize; j++){
            uint32_t ClusterID = result.top().second; result.pop();
            size_t ClusterSize = SubClusterList[ClusterID].size();
            for (size_t k = 0; k < ClusterSize; k++){
                uint32_t VectorID = SubClusterList[ClusterID][k];
                float Dist = faiss::fvec_L2sqr(QuerySet + i * Dimension, TrainSet + SubsetID[VectorID] * Dimension, Dimension);
                if (Dist < QueryGtDist[i * RecallK]){
                    faiss::maxheap_pop(RecallK, QueryGtDist.data() + i * RecallK, QueryGtLabel.data() + i * RecallK);
                    faiss::maxheap_push(RecallK, QueryGtDist.data() + i * RecallK, QueryGtLabel.data() + i * RecallK, Dist, VectorID);
                }
            }
        }
    }


    // Use the original GT in base set, only for test
/*
    std::string PathGT = "/home/yujianfu/Desktop/Dataset/SIFT1M/SIFT1M_groundtruth.ivecs";
    std::ifstream GTInput(PathGT, std::ios::binary);
    std::vector<uint32_t> QueryGT(nq * 100);
    readXvec<uint32_t> (GTInput, QueryGT.data(), 100, nq, true, true);
    for (size_t i = 0; i < nq; i++){
        for (size_t j = 0; j < RecallK; j++){
            QueryGtLabel[i * RecallK + j] = QueryGT[i * 100 + j];
        }
    }
*/

    TRecorder.print_time_usage("Complete query GT computation for recall@" + std::to_string(RecallK) + " with boundsize: " + std::to_string(ClusterBoundSize));
    TRecorder.record_time_usage(RecordFile, "Complete query GT computation for recall@" + std::to_string(RecallK) + " with boundsize: " + std::to_string(ClusterBoundSize));

    // Determine the number of clusters to be checked: need to consider the quantization error: build the quantizer and quantize the base set
    std::tuple<bool, size_t, float, float, float> TimeResult = UpdateVisitedClusters(RecordFile, RecallK, nq, PQ_M, nbits, SubTrainSize, PQ_TrainSize, TargetRecall, MaxCandidateSize, Dimension, TrainsetLabel.data(), SubsetID.data(), TrainSet, QueryGtLabel.data(), Graph, QuerySet, M, SubClusterList);

    TRecorder.print_time_usage("Update number of clusters to be visited");
    TRecorder.record_time_usage(RecordFile, "Update number of clusters to be visited");

    // Update the search cost candidate list size to check which cluster to be splited
    std::vector<float> ClusterVectorCost(M, 0);
    UpdateCandidateListSize(M, Dimension, Scale, ClusterList, Graph, TrainSet, std::get<1>(TimeResult), ClusterVectorCost.data());

    TRecorder.print_time_usage("Update Cluster Neighbor Cost");
    TRecorder.record_time_usage(RecordFile, "Update Cluster Neighbor Cost");
    

    delete Graph; Graph = nullptr;

    // Initialize the vector cost, for each cluster, the cost is the sum of inner cost (cluster size) and neighbor cost (number of neighbor vectors)
    auto comp = [](std::pair<float, uint32_t> Element1, std::pair<float, uint32_t> Element2){return Element1.first < Element2.first;};
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(comp)> ClusterCostQueue(comp);
    // Backup of the index infomation
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(comp)> ClusterCostQueueBackup(comp);
    std::vector<uint32_t> TrainsetLabelBackup;
    std::vector<float> TrainsetDistBackup;
    std::vector<float> ClusterVectorCostBackup;


    float SumVectorCost = 0;
    float SumInnerCost = 0;
    float OptPerTime = std::numeric_limits<float>::max();
    uint32_t OptM = M; 

    for (uint32_t i = 0; i < M; i++){
        ClusterCostQueue.emplace(std::make_pair(ClusterVectorCost[i], i));
        SumVectorCost += ClusterVectorCost[i];
        float InnerCost = Scale * Scale * ClusterList[i].size() * ClusterList[i].size();
        SumInnerCost += InnerCost;
    }
    std::cout << "The average candidate list size: " << SumVectorCost / nb << " The average inner candidate list size: " << SumInnerCost / nb << " The Ratio: " << (SumVectorCost / SumInnerCost ) << "\n";

    if (std::get<3>(TimeResult) + std::get<4>(TimeResult) <= OptPerTime){
        memcpy(ResultCentroids, Centroids, M * Dimension * sizeof(float));
        OptPerTime = std::get<3>(TimeResult) + std::get<4>(TimeResult);
    }

    RecordFile <<  "M: " << M <<  " Search Cluster Num: " << std::get<1>(TimeResult) << " PerCentroidTime: " << std::get<3>(TimeResult) << " Per Vec Time: " << std::get<4>(TimeResult) << " CandidateListSize: " << std::get<2>(TimeResult) <<  " Search Time: " << std::get<3>(TimeResult) + std::get<4>(TimeResult) << "\n\n";
    std::cout  <<  "M: " << M << " Search Cluster Num: " << std::get<1>(TimeResult) <<  " PerCentroidTime: " << std::get<3>(TimeResult) << " Per Vec Time: " << std::get<4>(TimeResult) << " CandidateListSize: " << std::get<2>(TimeResult) << " Search Time: " << std::get<3>(TimeResult) + std::get<4>(TimeResult) <<   "\n\n";

    bool NextSplit = M + CheckBatch < MaxM ? true : false;
    bool ShrinkBatch = true;
    bool OptInfoUpdated = false;
    float AvgVectorCost = SumVectorCost / M;
    uint32_t FailureTimes = 0;

    std::cout << "The avg vector cost by cluster: " << AvgVectorCost << "\n";

    if (verbose){std::cout << "Start spliting the clusters with with batch size: " << CheckBatch << "\n";}
    assert(CheckBatch <= M);
    
    // Record the time and candidate list, clusternum with the changing NC
    std::vector<size_t> NCRecord; std::vector<float> CenTimeRecord; std::vector<float> VecTimeRecord; std::vector<float> NumClusterRecord; std::vector<float> CansizeRecord;

    while (NextSplit)
    {
        TRecorder.reset();
        //uint32_t OriginM = M;
        float OriginSumVectorCost = SumVectorCost;
        SumVectorCost = 0;
        SumInnerCost = 0;

        std::vector<uint32_t> ClusterIDBatch(CheckBatch, 0);
        std::vector<double> ClusterCostBatch(CheckBatch, 0);
        for (size_t i = 0; i < CheckBatch; i++){
            ClusterIDBatch[i] = ClusterCostQueue.top().second;
            ClusterCostBatch[i] = ClusterCostQueue.top().first;
            ClusterCostQueue.pop();
        }
        while(!ClusterCostQueue.empty()){
            ClusterCostQueue.pop();
        }


        std::cout << "\n\nStart Split the clusters with NC = " << M << " and  spit batch = " << CheckBatch<<"\n"; 
/*
        std::cout << "\nThe ID of selected split clusters : [";
        RecordFile << "\nThe ID of selected split clusters : [";
        for (size_t i = 0; i < CheckBatch; i++){
            std::cout << ClusterIDBatch[i] << ", ";
            RecordFile << ClusterIDBatch[i] << ", ";
        }
        std::cout << "]\nThe size of selected split clusters: [";
        RecordFile << "]\nThe size of selected split clusters: [";
        for (size_t i = 0; i < CheckBatch; i++){
            std::cout << ClusterList[ClusterIDBatch[i]].size() << ", ";
            RecordFile << ClusterList[ClusterIDBatch[i]].size() << ", ";
        }
        std::cout << "]\nThe vector cost of the selected split clusters: [";
        RecordFile << "]\nThe vector cost of the selected split clusters: [";
        for (size_t i = 0; i < CheckBatch; i++){
            std::cout << ClusterCostBatch[i] << ", ";
            RecordFile << ClusterCostBatch[i] << ", ";
        }
        RecordFile << "]\n\n";
        std::cout << "]\n\n";
*/
        std::vector<uint32_t> SplitM(CheckBatch);
        std::vector<std::vector<float>> SplitCentroids(CheckBatch);
        std::vector<std::vector<std::vector<uint32_t>>> SplitClusterList(CheckBatch);

#pragma omp parallel for 
        for (size_t i = 0; i < CheckBatch; i++){ 
            // Train Split Centroids
            SplitM[i] = std::ceil(std::sqrt(ClusterCostBatch[i] / (AvgVectorCost)));
            assert(SplitM[i] > 0);
            SplitCentroids[i].resize(SplitM[i] * Dimension, 0);
            SplitClusterList[i].resize(SplitM[i]);

            size_t SplitTrainSize = ClusterList[ClusterIDBatch[i]].size();
            std::vector<float> SplitTrainSet(Dimension * SplitTrainSize);
            //std::cout << "Copy training vectors\n"; 
            for (size_t j = 0; j < SplitTrainSize; j++){
                memcpy(SplitTrainSet.data() + j * Dimension, TrainSet + ClusterList[ClusterIDBatch[i]][j] * Dimension, Dimension* sizeof(float));
            }
            std::vector<uint32_t> SplitLabels(SplitTrainSize, 0);
            std::vector<float> SplitDists(SplitTrainSize, 0);

            //std::cout << "Train the clusters with " << SplitM[i] << " centroids: " << ClusterCostBatch[i] << " " << AvgVectorCost << " " << SplitTrainSize << "\n"; 
            if (SplitTrainSize < SplitM[i]){std::cout << "Error: Split cluster with " << SplitTrainSize << " vectors to " << SplitM[i] << " clusters\n"; exit(0);};
            optkmeans(SplitTrainSet.data(), Dimension, SplitTrainSize, SplitM[i], SplitCentroids[i].data(), false, false, optimize, Lambda, OptSize, false, true, false, 30, true, SplitLabels.data(), SplitDists.data());

            // Assign the local ID and local dist
            //std::cout << "Update the assignment\n";
            for (size_t j = 0; j < SplitTrainSize; j++){
                SplitClusterList[i][SplitLabels[j]].emplace_back(ClusterList[ClusterIDBatch[i]][j]);
                TrainsetDist[ClusterList[ClusterIDBatch[i]][j]] = SplitDists[j];
            }

            //std::cout << "The result cluster size: ";
            for (size_t j = 0; j < SplitM[i]; j++){
                if(SplitClusterList[i][j].size() == 0){
                    std::cout << SplitTrainSize << " ";
                    for (size_t k = 0; k < SplitTrainSize; k++){
                        std::cout << SplitClusterList[i][j].size() << " ";
                    }
                    std::cout << "\n"; exit(0);
                }
            }
            //std::cout << "\n";
        }

        TRecorder.print_time_usage("Split the clusters");
        TRecorder.record_time_usage(RecordFile, "Split the clusters");

        // Clear the subclusterlist 
        for (size_t i = 0; i < M; i++) std::vector<uint32_t>().swap(SubClusterList[i]);


        //std::cout << "Split Result of the selected clusters: \n";
        
        // Update the split results: we replace the original centroid with the first new-generated centroid, and place the others at last
        // Update the M, centroids, clusterlist
        for (size_t i = 0; i < CheckBatch; i++){
            memcpy(Centroids + ClusterIDBatch[i] * Dimension, SplitCentroids[i].data() + 0 * Dimension, Dimension * sizeof(float));
            std::vector<uint32_t>().swap(ClusterList[ClusterIDBatch[i]]); ClusterList[ClusterIDBatch[i]].resize(SplitClusterList[i][0].size());
            memcpy(ClusterList[ClusterIDBatch[i]].data(), SplitClusterList[i][0].data(), SplitClusterList[i][0].size() * sizeof(uint32_t));

            //std::cout << "[ " << ClusterIDBatch[i] << " , " <<  SplitClusterList[i][0].size() << " ";
            //RecordFile << "[ " << ClusterIDBatch[i] << " , " <<  SplitClusterList[i][0].size() << " ";


            for (size_t j = 1; j < SplitM[i]; j++){
                memcpy(Centroids + M * Dimension, SplitCentroids[i].data() + j * Dimension, Dimension * sizeof(float));
                ClusterList.resize(M + 1); ClusterList[M].resize(SplitClusterList[i][j].size());
                memcpy(ClusterList[M].data(), SplitClusterList[i][j].data(), SplitClusterList[i][j].size() * sizeof(uint32_t));
                for (size_t k = 0; k < ClusterList[M].size(); k++){
                    TrainsetLabel[ClusterList[M][k]] = M;
                }
                //std::cout <<  M << " , " <<  SplitClusterList[i][j].size() << " ] , ";
                //RecordFile <<  M << " , " <<  SplitClusterList[i][j].size() << " ] , ";
                M++;
            }
        }
        //RecordFile << "\n";std::cout << "\n";
        TRecorder.print_time_usage("Update the train vector assignment");
        TRecorder.record_time_usage(RecordFile, "Update the train vector assignment");


        // Update the assignment list of the subset
        for (size_t i = 0; i < M; i++){
            for (size_t j = 0; j < ClusterList[i].size(); j++){
                assert(TrainsetLabel[ClusterList[i][j]] == i);
            }
        }

        SubClusterList.resize(M);
        for (uint32_t i = 0; i < SubTrainSize; i++){
            uint32_t VectorID = SubsetID[i];
            SubClusterList[TrainsetLabel[VectorID]].emplace_back(i);
        }

        // Report the split features
        double m =  float(TrainSize) / M;
        double accum = 0.0;
        double diffprop = 0.0;
        uint32_t MinSize = std::numeric_limits<uint32_t>::max();
        uint32_t MaxSize = 0;
        for (size_t i = 0; i < M; i++){
            accum += (ClusterList[i].size() - m) * (ClusterList[i].size() - m);
            diffprop += std::abs(ClusterList[i].size() - m);
            if (ClusterList[i].size() < MinSize) MinSize = ClusterList[i].size();
            if (ClusterList[i].size() > MaxSize) MaxSize = ClusterList[i].size();
        }
        double dist = std::accumulate(TrainsetDist.begin(), TrainsetDist.end(), 0.0);

        // Report the training result
        printf("Number of clusters: (%d), Vector - Centroid Dist (%.2f), Diff Prop: (%.1lf) / (%.0lf) MinSize: %d  MaxSize: %d, \n", M, dist / TrainSize, diffprop/M, m, MinSize, MaxSize);

/*
        // Update the centroids to avoid the error accumulation in local kmeans
        optkmeans(TrainSet, Dimension, TrainSize, M, Centroids, true, true, true, Lambda, OptSize, true, true, false, 2, true, TrainsetLabel.data(), TrainsetDist.data());
        for (size_t i = 0; i < M; i++){
            // Delete all cluster lists
            std::vector<uint32_t>().swap(ClusterList[i]);
        }
        for (size_t i = 0; i < TrainSize; i++){
            ClusterList[TrainsetLabel[i]].emplace_back(i);
        }
*/      
        // Rebuild the HNSW graph with new centroids for search
        Graph = new hnswlib::HierarchicalNSW(Dimension, M, GraphM, 2 * GraphM, GraphEf);
        for (size_t i = 0; i < M; i++){Graph->addPoint(Centroids + i * Dimension);}
        TRecorder.print_time_usage("Construct new graph for splitted centroids");
        TRecorder.record_time_usage(RecordFile, "Construct new graph for splitted centroids");


        // Update the clusternum parameter
        TimeResult = UpdateVisitedClusters(RecordFile, RecallK, nq, PQ_M, nbits, SubTrainSize, PQ_TrainSize, TargetRecall, MaxCandidateSize, Dimension, TrainsetLabel.data(),SubsetID.data(), TrainSet, QueryGtLabel.data(),Graph, QuerySet, M, SubClusterList);
        TRecorder.print_time_usage("Update the number of clusters to be visited for target recall");
        TRecorder.record_time_usage(RecordFile, "Update the number of clusters to be visited for target recall");

/*
        // Test the centroid search time
        QueryDist.resize(nq * TimeResult.first.second);
        QueryLabel.resize(nq * TimeResult.first.second);

        SearchTrecorder.reset();
        for (uint32_t i = 0; i < nq; i++){
            auto Result = Graph->searchBaseLayer(QuerySet + i * Dimension, TimeResult.first.second);
            for (size_t j = 0; j < TimeResult.first.second; j++){
                QueryDist[i * TimeResult.first.second + TimeResult.first.second - j - 1] = Result.top().first;
                QueryLabel[i * TimeResult.first.second + TimeResult.first.second - j - 1] = Result.top().second;
                Result.pop();
            }
        }
        PerCentroidTime = SearchTrecorder.getTimeConsumption() / (1000 * nq);
*/
        /*
        double PerCentroidCost = alpha * std::log(float(M + beta));
        double CentroidCostDiff =  alpha * (std::log(float(M + beta) / (OriginM + beta))) * nb;
        */

       float PerTime = std::get<3>(TimeResult) + std::get<4>(TimeResult);
        NCRecord.emplace_back(M); CenTimeRecord.emplace_back(std::get<3>(TimeResult)); VecTimeRecord.emplace_back(std::get<4>(TimeResult)); CansizeRecord.emplace_back(std::get<2>(TimeResult));
        RecordFile << "\nThis is index with |" << M << "| centroids with search time: |" << PerTime << "| PerCentroidTime: |" << std::get<3>(TimeResult) << "| Per Vecor Time: |" << std::get<4>(TimeResult) <<  "| ClusterNum: " << std::get<1>(TimeResult) << " Candidate List Size: " << std::get<2>(TimeResult) << std::endl;
        std::cout << "\nThis is index with |" << M << "| centroids with search time: |" << PerTime << "| PerCentroidTime: |" << std::get<3>(TimeResult) << "| Per Vecor Time: |" << std::get<4>(TimeResult) << "| ClusterNum: " << std::get<1>(TimeResult) << " Candidate List Size: " << std::get<2>(TimeResult) << std::endl;

        // If the next one achieves maxfailure times or the NC grows over the limitation, no need to update cluister cost
        if (!( FailureTimes == (MaxFailureTimes - 1) && std::get<0>(TimeResult) && PerTime >= OptPerTime) && M + CheckBatch <= MaxM){
            // Update the candidate list size
            ClusterVectorCost.resize(M);
            UpdateCandidateListSize(M, Dimension, Scale, ClusterList, Graph, TrainSet, std::get<1>(TimeResult), ClusterVectorCost.data());
            TRecorder.print_time_usage("Update the cluster search cost");
            TRecorder.record_time_usage(RecordFile, "Update the cluster search cost");
        }

        // This iteration offers new optimal centroids
        if (std::get<0>(TimeResult) && PerTime < OptPerTime){
            memcpy(ResultCentroids, Centroids, M * Dimension * sizeof(float));
            OptPerTime = PerTime;
            OptM = M;
            FailureTimes = 0;
            RecordFile << "This is new optimal index \n\n";
            std::cout << "This is new optimal index \n\n";
            if (ShrinkBatch){
                ClusterVectorCostBackup.resize(M);
                TrainsetLabelBackup.resize(TrainSize);
                TrainsetDistBackup.resize(TrainSize);
                memcpy(TrainsetLabelBackup.data(), TrainsetLabel.data(), TrainSize * sizeof(uint32_t));
                memcpy(ClusterVectorCostBackup.data(), ClusterVectorCost.data(), M * sizeof(float));
                memcpy(TrainsetDistBackup.data(), TrainsetDist.data(), TrainSize * sizeof(float));
                OptInfoUpdated = true;
                std::cout << "***[Opt Index backup saved]***\n";
            }
        }
        //  This iteration cannot achieve the target recall
        else if(! std::get<0>(TimeResult)){
            memcpy(ResultCentroids, Centroids, M * Dimension * sizeof(float));
            OptPerTime = PerTime;
            OptM = M;

            RecordFile << "This is *below recall target* index\n\n";
            std::cout << "This is *below recall target* index\n\n";
        }
        // This iteration offer non-optimal centroids
        else{
            FailureTimes ++;
            RecordFile << "This is *not* optimal index, the optimal NC is: |" << OptM << "|, Optimal search time: |" << OptPerTime << "| Failure Time: " << FailureTimes << " / " << MaxFailureTimes << "\n\n"; 
            std::cout << "These are *not* optimal index, the optimal NC is: |" << OptM << "|, Optimal search time: |" << OptPerTime << "| Failure Time: " << FailureTimes << " / " << MaxFailureTimes <<"\n\n"; 
        }

        // Stop increasing NC
        if (M + CheckBatch > MaxM || FailureTimes == MaxFailureTimes){
            if (ShrinkBatch && OptInfoUpdated){
                for (size_t i = 0; i < M; i++) {std::vector<uint32_t>().swap(SubClusterList[i]);std::vector<uint32_t>().swap(ClusterList[i]);}
                M = OptM;
                CheckBatch /= MaxFailureTimes > 5 ? MaxFailureTimes : 5;

                ShrinkBatch = false;
                memcpy(TrainsetLabel.data(), TrainsetLabelBackup.data(), TrainSize * sizeof(uint32_t));
                memcpy(ClusterVectorCost.data(), ClusterVectorCostBackup.data(), M * sizeof(float));
                memcpy(TrainsetDist.data(), TrainsetDistBackup.data(), TrainSize * sizeof(float));
                memcpy(Centroids, ResultCentroids, M * Dimension * sizeof(float));
                SubClusterList.resize(M); ClusterList.resize(M);

                for (uint32_t j = 0; j < TrainSize; j++){
                    assert(TrainsetLabel[j] < M);
                    ClusterList[TrainsetLabel[j]].emplace_back(j);
                }
                for (uint32_t j = 0; j < SubTrainSize; j++){
                    uint32_t VectorID = SubsetID[j];
                    SubClusterList[TrainsetLabel[VectorID]].emplace_back(j);
                }
                FailureTimes = 0;
                std::cout << "***[Achieve maximum NC or maximum failure times, back to the previous optimal setting and update the checkbatch to :" << CheckBatch << "]***\n";
            }
            else{
                NextSplit = false;
            }
        }

        if (NextSplit){
            // Calculate the inner cost and neighbor cost
            assert(ClusterCostQueue.size() == 0);
            for (uint32_t i = 0; i < M; i++){
                ClusterCostQueue.emplace(std::make_pair(ClusterVectorCost[i], i));
                SumVectorCost += ClusterVectorCost[i];
                float InnerCost = Scale * Scale * ClusterList[i].size() * ClusterList[i].size();
                SumInnerCost += InnerCost;
            }
            AvgVectorCost = SumVectorCost / M;
            std::cout << "For all base vectors, The average candidate list size: " << SumVectorCost / nb << " The average inner candidate list size: " << SumInnerCost / nb << " The Ratio: " << (SumVectorCost / SumInnerCost ) << " Origin PerVectorCost : " << OriginSumVectorCost / nb << " New PerVectorCost: " << SumVectorCost / nb    <<  " AvgVectorCost by cluster: " << AvgVectorCost << "\n";
            RecordFile << "For all base vectors, The average candidate list size: " << SumVectorCost / nb << " The average inner candidate list size: " << SumInnerCost / nb << " The Ratio: " << (SumVectorCost / SumInnerCost ) << " Origin PerVectorCost : " << OriginSumVectorCost / nb << " New PerVectorCost: " << SumVectorCost / nb    <<  " AvgVectorCost by cluster: " << AvgVectorCost << "\n";
        }

        delete Graph;
        Graph = nullptr;
    }

    RecordFile << "\nNC = [";
    for (size_t i = 0; i < NCRecord.size(); i++){
        RecordFile << NCRecord[i] << ", ";
    }
    RecordFile << "]\n";
    RecordFile << "Canlist = [";
    for (size_t i = 0; i < CansizeRecord.size(); i++){
        RecordFile << CansizeRecord[i] << ", ";
    }
    RecordFile << "]\n";
    RecordFile << "CenTime = [";
    for (size_t i = 0; i < CenTimeRecord.size(); i++){
        RecordFile << CenTimeRecord[i] << ", ";
    }
    RecordFile << "]\n";
    RecordFile << "VecTime = [";
    for (size_t i = 0; i < VecTimeRecord.size(); i++){
        RecordFile << VecTimeRecord[i] << ", ";
    }
    RecordFile << "]\n";
    RecordFile << "SumTime = [";
    for (size_t i = 0; i < VecTimeRecord.size(); i++){
        RecordFile << CenTimeRecord[i] + VecTimeRecord[i] << ", ";
    }

    RecordFile << "]\n";
    delete Centroids;
    return OptM;
}


// The section for changing the ClusterNum
// Input parameters: (1) scalar value (2) data vector (3) path (4) point, data structure
std::tuple<bool, size_t, float, float, float> BillionUpdateRecall(
    size_t nb, size_t nq, size_t Dimension, size_t nc, size_t RecallK, float TargetRecall, float MaxCandidateSize, size_t ngt,
    float * QuerySet, uint32_t * QueryGtLabel, float * CenNorms, uint32_t * Base_ID_seq,
    std::string Path_base, 
    std::ofstream & RecordFile, hnswlib::HierarchicalNSW * HNSWGraph, faiss::ProductQuantizer * PQ, std::vector<std::vector<uint32_t>> & BaseIds)
{

    time_recorder TRecorder = time_recorder();

    // Accumulate the vector quantization in the clusters to be visited by the queries
    std::cout << "Check the recall performance on different num clusters to be visited\n";
    std::vector<bool> QuantizeLabel(nc, false);
    std::vector<std::vector<uint8_t>> BaseCodeSubset(nc);
    std::vector<std::vector<float>> BaseRecoverNormSubset(nc);

    std::vector<int64_t> ResultID(RecallK * nq, 0);
    std::vector<float> ResultDist(RecallK * nq, 0);

    std::vector<std::unordered_set<uint32_t>> GtSets(nq);
    for (size_t i = 0; i < nq; i++){
        for (size_t j = 0; j < RecallK; j++){
            GtSets[i].insert(QueryGtLabel[ngt * i + j]);
        }
    }

    bool ValidResult = false;
    float MinimumCoef = 0.95;
    size_t MaxRepeatTimes = 3;
    const size_t Assignment_num_batch = 20;
    const size_t Assignment_batch_size = nb / Assignment_num_batch;


    size_t MaxClusterNum = std::ceil(MaxCandidateSize / (nb / nc));
    std::vector<float> MaxQueryDist(nq * MaxClusterNum);
    std::vector<uint32_t> MaxQueryLabel(nq * MaxClusterNum);
    for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
        auto result = HNSWGraph->searchBaseLayer(QuerySet + QueryIdx * Dimension, MaxClusterNum);
        for (size_t i = 0; i < MaxClusterNum; i++){
            MaxQueryLabel[QueryIdx * MaxClusterNum + MaxClusterNum - i - 1] = result.top().second;
            MaxQueryDist[QueryIdx * MaxClusterNum + MaxClusterNum - i - 1] = result.top().first;
            result.pop();
        }
    }

    size_t NumLoadCluster = 0;
    for (size_t i  = 0; i < nq * MaxClusterNum; i++){
        if (!QuantizeLabel[MaxQueryLabel[i]]){
            uint32_t ClusterLabel = MaxQueryLabel[i];
            QuantizeLabel[ClusterLabel] = true;
            BaseCodeSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size() * PQ->code_size);
            BaseRecoverNormSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size());
            NumLoadCluster ++;
        }
    }
    std::cout << "Search " << NumLoadCluster << " / " << nc << " clusters in the index for evaluation with max ClusterNum: " << MaxClusterNum << "\n";
    std::ifstream BaseInput(Path_base, std::ios::binary);
    std::vector<float> Base_batch(Assignment_batch_size * Dimension);

    for (size_t i = 0; i < Assignment_num_batch; i++){
        TRecorder.reset();
        readXvec<DataType>(BaseInput, Base_batch.data(), Dimension, Assignment_batch_size, false, false);
        TRecorder.print_record_time_usage(RecordFile, "Load the " + std::to_string(i + 1) + " / " + std::to_string(Assignment_num_batch) + " batch");
#pragma omp parallel for
        for (size_t j = 0; j < Assignment_batch_size; j++){
            uint32_t ClusterLabel = Base_ID_seq[i * Assignment_batch_size + j];
            if (QuantizeLabel[ClusterLabel]){
                uint32_t ClusterInnerIdx = 0;
                while (BaseIds[ClusterLabel][ClusterInnerIdx] != i * Assignment_batch_size + j)
                {
                    ClusterInnerIdx ++;
                }
                assert(ClusterInnerIdx < BaseIds[ClusterLabel].size());

                std::vector<float> BaseResidual(Dimension);
                std::vector<float> RecoverResidual(Dimension);
                std::vector<float> RecoverVector(Dimension);
                faiss::fvec_madd(Dimension, Base_batch.data() + j * Dimension, -1.0,  HNSWGraph->getDataByInternalId(ClusterLabel), BaseResidual.data());

                PQ->compute_code(BaseResidual.data(), BaseCodeSubset[ClusterLabel].data() + ClusterInnerIdx * PQ->code_size);
                PQ->decode(BaseCodeSubset[ClusterLabel].data() + ClusterInnerIdx * PQ->code_size, RecoverResidual.data());
                faiss::fvec_madd(Dimension, RecoverResidual.data(), 1.0, HNSWGraph->getDataByInternalId(ClusterLabel), RecoverVector.data());
                BaseRecoverNormSubset[ClusterLabel][ClusterInnerIdx] = faiss::fvec_norm_L2sqr(RecoverVector.data(), Dimension);
            }
        }
        TRecorder.print_record_time_usage(RecordFile, "Process the " + std::to_string(i + 1) + " / " + std::to_string(Assignment_num_batch) + " batch");
    }
    BaseInput.close();

    while(!ValidResult){
        size_t ClusterNum = size_t(MaxCandidateSize / (2 * (nb / nc)));
        size_t ClusterBatch = std::ceil(float(ClusterNum) / 10);
        std::vector<float> ClusterNumList;
        std::vector<float> CanLengthList;
        std::vector<float> CenSearchTime;
        std::vector<float> VecSearchTime;

        bool AchieveTargetRecall = true;
        bool UpdateClusterNum = true;
        bool IncreaseClusterNum = false;
        bool DecreaseClusterNum = false;
        float ResultIndice = -1;
        float VisitedVec = 0;
        size_t PreviousClusterNum = ClusterNum;
        float PreviousRecordTime1 = 0;
        float PreviousRecordTime3 = 0;
        size_t RepeatTimes = 0;

        // Change different ClusterNum
        while (UpdateClusterNum)
        {
            // Record the time of graph search on centroids
            std::vector<float> QueryDist(nq * ClusterNum);
            std::vector<uint32_t> QueryLabel(nq * ClusterNum);
            TRecorder.reset();
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                auto result = HNSWGraph->searchBaseLayer(QuerySet + QueryIdx * Dimension, ClusterNum);
                for (size_t i = 0; i < ClusterNum; i++){
                    QueryLabel[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().second;
                    QueryDist[QueryIdx * (ClusterNum) +  ClusterNum - i - 1] = result.top().first;
                    result.pop();
                }
            }

            TRecorder.recordTimeConsumption1();


/*
            size_t NumLoadCluster = 0;
            // Load and quantize the vectors to be visited
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                // Do parallel for the search result of one query, as there will be no repeat

                for (size_t i = 0; i < ClusterNum; i++){
                    if (!QuantizeLabel[QueryLabel[QueryIdx * ClusterNum + i]]){
                        NumLoadCluster ++;
                        uint32_t ClusterLabel = QueryLabel[QueryIdx * ClusterNum + i];
                        // The vectors are not quantized, read and quantize the base vectors
                        QuantizeLabel[ClusterLabel] = true;
                        BaseCodeSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size() * PQ->code_size);
                        BaseRecoverNormSubset[ClusterLabel].resize(BaseIds[ClusterLabel].size());

#pragma omp parallel for
                        for (size_t j = 0; j < BaseIds[ClusterLabel].size(); j++){
                            std::ifstream BaseInput(Path_base, std::ios::binary);
                            std::vector<float> BaseVector(Dimension);
                            BaseInput.seekg(BaseIds[ClusterLabel][j] * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);

                            readXvecFvec<DataType>(BaseInput, BaseVector.data(), Dimension, 1);
                            std::vector<float> BaseResidual(Dimension);
                            std::vector<float> RecoverResidual(Dimension);
                            faiss::fvec_madd(Dimension, BaseVector.data(), -1.0,  HNSWGraph->getDataByInternalId(ClusterLabel), BaseResidual.data());
                            PQ->compute_code(BaseResidual.data(), BaseCodeSubset[ClusterLabel].data() + j * PQ->code_size);
                            PQ->decode(BaseCodeSubset[ClusterLabel].data() + j * PQ->code_size, RecoverResidual.data());
                            std::vector<float> RecoverVector(Dimension);
                            faiss::fvec_madd(Dimension, RecoverResidual.data(), 1.0, HNSWGraph->getDataByInternalId(ClusterLabel), RecoverVector.data());
                            BaseRecoverNormSubset[ClusterLabel][j] = faiss::fvec_norm_L2sqr(RecoverVector.data(), Dimension);

                            //std::cout << faiss::fvec_norm_L2sqr(BaseResidual.data(), Dimension) << " " << faiss::fvec_norm_L2sqr(RecoverResidual.data(), Dimension) << " " << faiss::fvec_L2sqr(BaseResidual.data(), RecoverResidual.data(), Dimension) << " | "; 
                            BaseInput.close();
                        }
                    }
                }
            }
            TRecorder.print_time_usage("Load and quantize the base vectors in |" + std::to_string(NumLoadCluster) + "| clusters ");
*/

            // Prepare the PQ table for queries
            TRecorder.reset();
            std::vector<float> PQTable(nq * PQ->ksub * PQ->M, 0);
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx ++){
                PQ->compute_inner_prod_table(QuerySet + QueryIdx * Dimension, PQTable.data() + QueryIdx * PQ->ksub * PQ->M);
            }
            TRecorder.recordTimeConsumption2();

            // Search the quantized vectors and examine the recall target
            VisitedVec = 0;
            TRecorder.reset();
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                faiss::maxheap_heapify(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK);
                //std::cout << "4: \n";
                for (size_t i = 0; i < ClusterNum; i++){
                    uint32_t ClusterID = QueryLabel[QueryIdx * ClusterNum + i];
                    float CDist = QueryDist[QueryIdx * ClusterNum + i] - CenNorms[ClusterID];
                    VisitedVec += BaseRecoverNormSubset[ClusterID].size();
                    //std::cout << "5: \n";
                    for (size_t j = 0; j < BaseRecoverNormSubset[ClusterID].size(); j++){

                        float VNorm = BaseRecoverNormSubset[ClusterID][j];
                        float ProdDist = 0;
                        for (size_t k = 0; k < PQ->code_size; k++){
                            ProdDist += PQTable[QueryIdx * PQ->ksub * PQ->M + PQ->ksub * k + BaseCodeSubset[ClusterID][j * PQ->code_size + k]];
                        }
                        float Dist = CDist + VNorm - 2 * ProdDist;

/*
                        std::vector<float> BaseVector(Dimension);
                        std::ifstream BaseInput(Path_base, std::ios::binary);
                        BaseInput.seekg(BaseIds[ClusterID][j] * (Dimension * sizeof(DataType) + sizeof(uint32_t)) + sizeof(uint32_t), std::ios::beg);
                        BaseInput.read((char *) BaseVector.data(), Dimension * sizeof(DataType));
                        std::cout << "ID: " << BaseIds[ClusterID][j] << " ClusterID: " << ClusterID << "\n";
                        std::cout << "Correct QC Dist: " << faiss::fvec_L2sqr(QuerySet + QueryIdx * Dimension, CentroidHNSW->getDataByInternalId(ClusterID), Dimension) << " Result: " << QueryDist[QueryIdx * ClusterNum + i] << std::endl;
                        std::cout << "Correct CNorm: " << faiss::fvec_norm_L2sqr(CentroidHNSW->getDataByInternalId(ClusterID), Dimension) << " Result: " << CenNorms[ClusterID] << std::endl; 
                        std::vector<float> Residual(Dimension);
                        faiss::fvec_madd(Dimension, BaseVector.data(), -1.0, CentroidHNSW->getDataByInternalId(ClusterID), Residual.data());
                        std::vector<uint8_t> ResidualCode(PQ->code_size);
                        PQ->compute_code(Residual.data(), ResidualCode.data());
                        for (size_t temp = 0; temp < PQ->code_size; temp++){
                            assert(ResidualCode[temp] == BaseCodeSubset[ClusterID][j * PQ->code_size + temp]);
                        }
                        
                        std::vector<float> RecoveredResidual(Dimension);
                        PQ->decode(BaseCodeSubset[ClusterID].data() + j * PQ->code_size, RecoveredResidual.data(),1);
                        float CorrectProdDist = faiss::fvec_inner_product(RecoveredResidual.data(), QuerySet + QueryIdx * Dimension, Dimension);
                        std::cout << "Correct Prod Dist: " << CorrectProdDist << " Result ProdDist: " << ProdDist << std::endl;
                        std::vector<float> OPQBase(Dimension);
                        faiss::fvec_madd(Dimension, RecoveredResidual.data(), 1.0, CentroidHNSW->getDataByInternalId(ClusterID), OPQBase.data());
                        float CorrectBaseNorm = faiss::fvec_norm_L2sqr(OPQBase.data(), Dimension);
                        float CorrectResultDist = faiss::fvec_L2sqr(OPQBase.data(), QuerySet + QueryIdx * Dimension, Dimension);
                        std::cout << "Precise Base Norm " << faiss::fvec_norm_L2sqr(BaseVector.data(), Dimension) << " Recorded Base Norm " << BaseRecoverNormSubset[ClusterID][j] <<  " Correct Base Norm: " << CorrectBaseNorm << std::endl;
                        std::cout << "Precise Dist " << faiss::fvec_L2sqr(QuerySet + QueryIdx * Dimension, BaseVector.data(), Dimension) << " Result Dist: " << Dist <<  " Correct Result Dist: " << CorrectResultDist << std::endl;
                        std::cout << "Norm of the QC residual: " << faiss::fvec_norm_L2sqr(Residual.data(), Dimension) << " Norm of the PQ residual: " << faiss::fvec_norm_L2sqr(RecoveredResidual.data(), Dimension) << " Distance between two residual: " << faiss::fvec_L2sqr(Residual.data(), RecoveredResidual.data(), Dimension) << "\n";
*/
                        if (Dist < ResultDist[QueryIdx * RecallK]){
                            faiss::maxheap_pop(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK);
                            faiss::maxheap_push(RecallK, ResultDist.data() + QueryIdx * RecallK, ResultID.data() + QueryIdx * RecallK, Dist, BaseIds[ClusterID][j]);
                        }
                    }
                }
            }
            TRecorder.recordTimeConsumption3();

            if (PreviousClusterNum < ClusterNum && RepeatTimes < MaxRepeatTimes){
                if (TRecorder.TempDuration1 < PreviousRecordTime1 || TRecorder.TempDuration3 < PreviousRecordTime3){
                    RepeatTimes ++;
                    continue;
                }
            }
            else if (PreviousClusterNum > ClusterNum && RepeatTimes < MaxRepeatTimes){
                if (TRecorder.TempDuration1 > PreviousRecordTime1 || TRecorder.TempDuration3 > PreviousRecordTime3){
                    RepeatTimes ++;
                    continue;
                }
            }

            float VisitedGt = 0;
            for (size_t QueryIdx = 0; QueryIdx < nq; QueryIdx++){
                for (size_t i = 0; i < RecallK; i++){
                    if (GtSets[QueryIdx].count(ResultID[QueryIdx * RecallK + i]) != 0){
                        VisitedGt ++;
                    }
                }
            }
            std::cout << "'Numcluster: " << ClusterNum << " Cluster Batch: " << ClusterBatch << " Visited num of GT in top " << RecallK << " NN: " << (VisitedGt) / nq << " Candidate List Size: " << (VisitedVec) / nq  << " / " << MaxCandidateSize <<  " Target recall: "<< TargetRecall  << " Search Time: " << (TRecorder.TempDuration1 + TRecorder.TempDuration2 + TRecorder.TempDuration3) / (nq * 1000) <<  " Cen Search Time: " <<  TRecorder.TempDuration1 / (nq * 1000)  << " Table time: " <<  TRecorder.TempDuration2 / (nq * 1000) << " Vec Search Time: " << TRecorder.TempDuration3 / (nq * 1000)  << " with repeat times: " << RepeatTimes << " / " << MaxRepeatTimes << "',\n";
            RecordFile << "'Numcluster: " << ClusterNum << " Cluster Batch: " << ClusterBatch << " Visited num of GT in top " << RecallK << " NN: " << (VisitedGt) / nq << " Candidate List Size: " << (VisitedVec) / nq <<  " / " << MaxCandidateSize << " Target recall: "<< TargetRecall  << " Search Time: " << (TRecorder.TempDuration1 + TRecorder.TempDuration2 + TRecorder.TempDuration3) / (nq * 1000) << " Cen Search Time: " <<  TRecorder.TempDuration1 / (nq * 1000) << " Table time: " <<  TRecorder.TempDuration2 / (nq * 1000) << " Vec Search Time: " << TRecorder.TempDuration3 / (nq * 1000)  << " with repeat times: " << RepeatTimes << " / " << MaxRepeatTimes << "',\n";

            //std::cout << "6: \n";
            ClusterNumList.emplace_back(ClusterNum); CanLengthList.emplace_back((VisitedVec) / nq); CenSearchTime.emplace_back(TRecorder.TempDuration1 / (nq * 1000)); VecSearchTime.emplace_back(TRecorder.TempDuration3 / (nq * 1000));

            // Determine how should the clusternum change
            if (ClusterNum >= MaxClusterNum){
                UpdateClusterNum = false;
                AchieveTargetRecall = false;
            }

            else if (VisitedGt / nq > TargetRecall){
                ResultIndice = ClusterNumList.size() - 1;
                if (IncreaseClusterNum){
                    if (ClusterBatch > 1){
                        ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                        ClusterBatch = std::ceil(float(ClusterBatch) / 10);
                        ClusterNum += ClusterBatch;
                    }
                    else{
                        UpdateClusterNum = false;
                    }
                }
                else{
                    DecreaseClusterNum = true;
                    ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                }
            }
            else if (VisitedGt / nq < TargetRecall){
                if (DecreaseClusterNum){
                    if (ClusterBatch > 1){
                        ClusterNum += ClusterBatch;
                        ClusterBatch = std::ceil(float(ClusterBatch) / 10);
                        ClusterNum = ClusterNum > ClusterBatch ? ClusterNum - ClusterBatch : 1;
                    }
                    else{
                        ClusterNum += 1;
                        UpdateClusterNum = false;
                    }
                }
                else{
                    IncreaseClusterNum = true;
                    ClusterNum += ClusterBatch;
                }
            }
            else{
                ResultIndice = ClusterNumList.size() - 1;
                UpdateClusterNum = false;
            }
            //std::cout << "7: \n";
            PreviousRecordTime1 = TRecorder.TempDuration1;
            PreviousRecordTime3 = TRecorder.TempDuration3;
            RepeatTimes = 0;
        }

        //std::cout << "8: \n";
        
        std::pair<float, std::pair<float, float>> Result1 =  LeastSquares(ClusterNumList, CenSearchTime);
        std::pair<float, std::pair<float, float>> Result2 = LeastSquares(CanLengthList, VecSearchTime);
        if (ResultIndice < 0){ResultIndice = CanLengthList.size() - 1;} assert(ClusterNumList[ResultIndice] == ClusterNum);
        float Centime = Result1.second.first * ClusterNumList[ResultIndice] + Result1.second.second;
        float Vectime = Result2.second.first * CanLengthList[ResultIndice] + Result2.second.second;
        std::cout << "The search time on centroids and vectors: |" << Centime << "| |" << Vectime << "|\n";
        if (Result1.first > MinimumCoef && Result2.first > MinimumCoef){
            return std::make_tuple(AchieveTargetRecall, ClusterNumList[ResultIndice], CanLengthList[ResultIndice], Centime, Vectime);
        }
        else{
            RecordFile << "Regression failed with " << Result1.first << " and " << Result2.first << ", repeat the time estimation process\n\n";
            std::cout << "Regression failed with " << Result1.first << " and " << Result2.first << ", repeat the time estimation process\n\n";
        }
    }
    return std::make_tuple(0, 0, 0, 0, 0); 
}


// Input parameters: (1) scalar value (2) data vector (3) path (4) point, data structure
void BillionUpdateCost(
    size_t ClusterNum, size_t nc, float CheckProp, size_t LowerBound, size_t Dimension,
    float * ClusterVectorCost,
    std::string Path_base,
    std::vector<std::vector<uint32_t>> & BaseIds, hnswlib::HierarchicalNSW * Graph
){

/*
    // The proportion of checked vectors
    std::cout << "Update cluster search cost with Number of cluster visited: " << ClusterNum <<  " The clusterVectorCost with checkprop: " << CheckProp <<"\n";
#pragma omp parallel for
    for (size_t i = 0; i < NC; i++){
        long RandomSeed = 1234;
        std::ifstream BaseInput(Path_base, std::ios::binary);

        size_t ClusterSize = BaseIds[i].size();
        std::vector<int> RandomId(ClusterSize);
        faiss::rand_perm(RandomId.data(), ClusterSize, RandomSeed+1);
        size_t SampleClusterSize = std::ceil(CheckProp * ClusterSize) > LowerBound ? std::ceil(CheckProp * ClusterSize) : LowerBound < ClusterSize ? LowerBound : ClusterSize;
        size_t CandidateListSize = 0;

        for (size_t VectorIndice = 0; VectorIndice < SampleClusterSize; VectorIndice++){
            std::vector<uint32_t> VectorLabel(ClusterNum);
            std::vector<float> VectorDist(ClusterNum);
            uint32_t VectorID = BaseIds[i][RandomId[VectorIndice]];

            std::vector<float> BaseVector(Dimension);
            BaseInput.seekg(VectorID * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
            readXvecFvec<DataType>(BaseInput, BaseVector.data(), Dimension, 1);

            auto result = Graph->searchKnn(BaseVector.data(), ClusterNum);
            for (size_t j = 0; j < ClusterNum; j++){
                VectorLabel[ClusterNum - j - 1] = result.top().second;
                VectorDist[ClusterNum - j - 1] = result.top().first;
                result.pop();
            }

            for (size_t j = 0; j < ClusterNum; j++){
                CandidateListSize += float(ClusterSize * BaseIds[VectorLabel[j]].size()) / SampleClusterSize;
            }
        }
        ClusterVectorCost[i] = CandidateListSize;
        BaseInput.close();
    }
*/
    assert(Dimension > 0 && LowerBound > 0 && CheckProp > 0 && exists(Path_base));
    std::cout << "Update cluster search cost based on centroids with Number of cluster visited: " << ClusterNum << "\n";
#pragma omp parallel for
    for (size_t i = 0; i < nc; i++){
        std::vector<uint32_t> VectorLabel(ClusterNum);
        std::vector<float> VectorDist(ClusterNum);

        auto result = Graph->searchKnn(Graph->getDataByInternalId(i), ClusterNum);
        for (size_t j = 0; j < ClusterNum; j++){
            VectorLabel[ClusterNum - j - 1] = result.top().second;
            VectorDist[ClusterNum - j - 1] = result.top().first;
            result.pop();
        }
        size_t CandidateListSize = 0;
        for(size_t j = 0; j < ClusterNum; j++){
            CandidateListSize += BaseIds[i].size() * BaseIds[VectorLabel[j]].size();
        }
        ClusterVectorCost[i] = CandidateListSize;
    }
}

// Input parameters: (1) scalar value (2) data vector (3) path (4) point, data structure
void BillionUpdateCentroids(
    size_t Dimension, size_t NCBatch, float AvgVectorCost, bool optimize, float Lambda, size_t OptSize, size_t & NC,
    float * ClusterCostBatch, uint32_t * ClusterIDBatch, float * Centroids, uint32_t * Base_ID_seq,
    std::string Path_base,
    std::vector<std::vector<uint32_t>> & BaseIds
){
    std::vector<uint32_t> SplitM(NCBatch);
    std::vector<std::vector<float>> SplitCentroids(NCBatch);
    std::vector<std::vector<std::vector<uint32_t>>> SplitBaseIds(NCBatch);

    std::cout << "Loading the base vectors for cluster split training\n";
    std::vector<std::vector<float>> SplitTrainSets(NCBatch);
    std::ifstream BaseInput(Path_base, std::ios::binary);
    for (size_t i = 0; i < NCBatch; i++){
        size_t SplitTrainSize =  BaseIds[ClusterIDBatch[i]].size();
        SplitTrainSets[i].resize(Dimension * SplitTrainSize);
        for (size_t j = 0; j < SplitTrainSize; j++){
            BaseInput.seekg(BaseIds[ClusterIDBatch[i]][j] * (Dimension * sizeof(DataType) + sizeof(uint32_t)), std::ios::beg);
            readXvecFvec<DataType>(BaseInput, SplitTrainSets[i].data() + j * Dimension, Dimension, 1);
        }
    }
    BaseInput.close();

#pragma omp parallel for 
    for (size_t i = 0; i < NCBatch; i++){ 
        // Train Split Centroids
        SplitM[i] = std::ceil(std::sqrt(ClusterCostBatch[i] / (AvgVectorCost)));
        assert(SplitM[i] > 0);
        SplitCentroids[i].resize(SplitM[i] * Dimension, 0);
        SplitBaseIds[i].resize(SplitM[i]);

        size_t SplitTrainSize = BaseIds[ClusterIDBatch[i]].size();

        std::vector<uint32_t> SplitLabels(SplitTrainSize, 0);
        std::vector<float> SplitDists(SplitTrainSize, 0);

        //std::cout << "Train the clusters with " << SplitM[i] << " centroids: " << ClusterCostBatch[i] << " " << AvgVectorCost << " " << SplitTrainSize << "\n"; 
        if (SplitTrainSize < SplitM[i]){std::cout << "Error: Split cluster with " << SplitTrainSize << " vectors to " << SplitM[i] << " clusters\n"; exit(0);};
        optkmeans(SplitTrainSets[i].data(), Dimension, SplitTrainSize, SplitM[i], SplitCentroids[i].data(), false, false, optimize, Lambda, OptSize, false, true, false, 30, true, SplitLabels.data(), SplitDists.data());

        // Assign the local ID and local dist
        //std::cout << "Update the assignment\n";
        for (size_t j = 0; j < SplitTrainSize; j++){
            SplitBaseIds[i][SplitLabels[j]].emplace_back(BaseIds[ClusterIDBatch[i]][j]);
        }
    }

    for (size_t i = 0; i < NCBatch; i++){
        memcpy(Centroids + ClusterIDBatch[i] * Dimension, SplitCentroids[i].data() + 0 * Dimension, Dimension * sizeof(float));
        std::vector<uint32_t>().swap(BaseIds[ClusterIDBatch[i]]); BaseIds[ClusterIDBatch[i]].resize(SplitBaseIds[i][0].size());
        memcpy(BaseIds[ClusterIDBatch[i]].data(), SplitBaseIds[i][0].data(), SplitBaseIds[i][0].size() * sizeof(uint32_t));


        for (size_t j = 1; j < SplitM[i]; j++){
            memcpy(Centroids + NC * Dimension, SplitCentroids[i].data() + j * Dimension, Dimension * sizeof(float));
            BaseIds.resize(NC + 1); BaseIds[NC].resize(SplitBaseIds[i][j].size());
            memcpy(BaseIds[NC].data(), SplitBaseIds[i][j].data(), SplitBaseIds[i][j].size() * sizeof(uint32_t));
            for (size_t k = 0; k < BaseIds[NC].size(); k++){
                Base_ID_seq[BaseIds[NC][k]] = NC;
            }
            NC++;
        }
    }

}
