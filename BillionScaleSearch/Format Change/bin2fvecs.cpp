#include <stdio.h>
#include <iostream>
#include <fstream>
#include<assert.h>
#include<vector>

typedef uint8_t DataType;


int main(){

    std::string InputPath = "/data/yujian/Dataset/SIFT100M/SIFT100M.ubin";
    std::string OutputPath = "/data/yujian/Dataset/SIFT100M/SIFT100M.fvecs";

    std::ifstream Input(InputPath, std::ios::binary);
    std::ofstream Output(OutputPath, std::ios::binary);

    size_t nb = 1e8;
    uint32_t Dimension = 128;

    uint32_t n_points = 0;
    uint32_t n_dim = 0;

    Input.read((char *) & n_points, sizeof(uint32_t));
    Input.read((char *) & n_dim, sizeof(uint32_t));

    if (n_dim != Dimension){
        std::cout << "The loaded dimension: " << n_dim << " The loaded number of points: " << n_points << "\n";
    }
    assert(n_points == nb && n_dim == Dimension);

    std::vector<float> DataVector(Dimension);
    for (size_t i = 0; i < nb; i++){
        Input.read((char *) DataVector.data(), sizeof(DataType) * Dimension);
        Output.write((char *) & Dimension, sizeof(uint32_t));
        Output.write((char *) DataVector.data(), sizeof(float) * Dimension);

        if (i % 100000 == 0){
            std::cout << "Processing " << i <<  " / " << nb << "\r";
        }
    }
}





