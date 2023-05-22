#include <stdio.h>
#include <iostream>
#include <fstream>
#include<assert.h>
#include<vector>

typedef float DataType;


int main(){

    std::string InputPath = "/home/yujianfu/Desktop/Dataset/SIFT/SIFT1M/SIFT1M_base.fvecs";
    std::string OutputPath = "/home/yujianfu/Desktop/Dataset/SIFT/SIFT1M/SIFT1M_base.fbin";

    size_t nb = 1000000;
    uint32_t Dimension = 128;
    uint32_t n_dim = 0;

    std::ifstream Input(InputPath, std::ios::binary);
    std::ofstream Output(OutputPath, std::ios::binary);



    std::vector<DataType> DataVector(Dimension);
    std::vector<float> DataVectorFloat(Dimension);

    for (size_t i = 0; i < nb; i++){
        Input.read((char *) & n_dim, sizeof(uint32_t));
        assert(n_dim == Dimension);
        Input.read((char *) DataVector.data(), sizeof(DataType) * Dimension);
        for (size_t j = 0; j < Dimension; j++){
            DataVectorFloat[j] = DataVector[j];
        }
        Output.write((char *) DataVectorFloat.data(), sizeof(float) * Dimension);
        if ((i % (nb / 100)) == 0){
            std::cout << "Processed " << i << " / " << nb << "\r";
        }
    }
}





