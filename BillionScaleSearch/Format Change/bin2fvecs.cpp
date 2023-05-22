#include <stdio.h>
#include <iostream>
#include <fstream>
#include<assert.h>
#include<vector>

typedef uint8_t DataType;


int main(){

    std::string InputPath = "/data/yujian/Dataset/SIFT100M/SIFT100M_query.i8bin";
    std::string OutputPath = "/data/yujian/Dataset/SIFT100M/SIFT100M_query.fvecs";

    size_t nb = 10000;
    uint32_t Dimension = 128;

    std::ifstream Input(InputPath, std::ios::binary);
    std::ofstream Output(OutputPath, std::ios::binary);

    uint32_t n_points = 0;
    uint32_t n_dim = 0;

    Input.read((char *) & n_points, sizeof(uint32_t));
    Input.read((char *) & n_dim, sizeof(uint32_t));

    if (n_dim != Dimension){
        std::cout << "The loaded dimension: " << n_dim << " The loaded number of points: " << n_points << "\n";
    }
    assert(n_points == nb && n_dim == Dimension);

    std::vector<DataType> OriginDataVector(Dimension);
    std::vector<float> DataVector(Dimension);
    for (size_t i = 0; i < nb; i++){
        Input.read((char *) OriginDataVector.data(), sizeof(DataType) * Dimension);
        for (size_t j = 0; j < Dimension; j++){
            DataVector[j] = OriginDataVector[j];
        }
        Output.write((char *) & Dimension, sizeof(uint32_t));
        Output.write((char *) DataVector.data(), sizeof(float) * Dimension);

        if (i % 100000 == 0){
            std::cout << "Processing " << i <<  " / " << nb << "\r";
        }
    }
}





