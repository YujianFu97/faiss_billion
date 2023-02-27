#include "complib/include/codecfactory.h"

using namespace SIMDCompressionLib;
int main(){

    SIMDCompressionLib::IntegerCODEC & codec = *CODECFactory::getFromName("fastpfor");
    size_t Num = 10000;
    std::vector<uint32_t> Data(Num);
    for (uint32_t i = 0; i < Num; i++){
        Data[i] = i % 21;
    }
    std::vector<uint32_t> DataCopy(Data);
    std::vector<uint32_t> CompressOutput(Num + 1024);
    size_t CompressedSize = CompressOutput.size();
    std::cout << CompressOutput.size() << std::endl;
    codec.encodeArray(Data.data(), Data.size(), CompressOutput.data(), CompressedSize);
    std::cout << CompressOutput.size() << std::endl;
    CompressOutput.resize(CompressedSize);
    std::cout << CompressOutput.size() << std::endl;
    CompressOutput.shrink_to_fit();
    std::cout << CompressOutput.size() << std::endl;

  cout << setprecision(3);
  cout << "You are using "
       << 32.0 * static_cast<double>(CompressOutput.size()) /
              static_cast<double>(Data.size())
       << " bits per integer. " << endl;


  vector<uint32_t> mydataback(Num);
  size_t recoveredsize = mydataback.size();
  //
  codec.decodeArray(CompressOutput.data(), CompressOutput.size(),
                    mydataback.data(), recoveredsize);
  mydataback.resize(recoveredsize);
  
}