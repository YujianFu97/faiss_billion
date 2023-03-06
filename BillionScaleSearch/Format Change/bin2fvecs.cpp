#include <stdio.h>
#include <iostream>
#include <fstream>
#include<assert.h>
#include<vector>

typedef int8_t DataType;


int main(){

    std::string InputPath = "/data/yujian/Dataset/Space1B/space1b_base.i8bin";
    std::string OutputPath = "/data/yujian/Dataset/Space1B/Space1b_base.fvecs";

    std::ifstream Input(InputPath, std::ios::binary);
    std::ofstream Output(OutputPath, std::ios::binary);

    size_t nb = 1e9;
    uint32_t Dimension = 100;

    uint32_t Dim = 100;
    uint32_t n_points = 0;
    uint32_t n_dim = 0;
    Input.read((char *) & n_points, sizeof(uint32_t));
    Input.read((char *) & n_dim, sizeof(uint32_t));
    assert(n_points == nb && n_dim == Dimension);

    std::vector<float> DataVector(Dimension);
    for (size_t i = 0; i < nb; i++){
        Input.read((char *) DataVector.data(), sizeof(DataType) * Dimension);
        Output.write((char *) & Dimension, sizeof(uint32_t));
        Output.write((char *) DataVector.data(), sizeof(DataType) * Dimension);

        if (i % 100000 == 0){
            std::cout << "Processing " << i <<  " / " << nb << "\r";
        }
    }
}





