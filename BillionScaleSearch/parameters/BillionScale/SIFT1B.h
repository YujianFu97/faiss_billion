#include <cstdio>
#include <string>

const size_t nb = 1000000000;
const size_t Dimension = 128;
const std::string Dataset = "SIFT1B";
std::string PathFolder = "/data/yujian/Dataset/";
//std::string PathFolder = "/home/yujian/Yujian/Dataset/";
//std::string PathFolder = "/home/y/yujianfu/Dataset/";

const uint32_t ngt = 1000;
const size_t nq = 10000;
const size_t nt = 100000000;

// For OptKmeans
float Lambda = 400;
size_t OptSize = 80;
