#include <vector>
#include <chrono>
#include <iostream>

int main() {
    const size_t s = 1000000000; // 1 billion
    size_t n_test = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_test; i++){
        //static bool myvec[s] = {false};
        bool * MyVec = new bool[s];
        //std::vector<bool> myvec(s, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start) / n_test;
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}