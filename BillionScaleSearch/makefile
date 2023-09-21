# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CXX          = g++ -std=c++14
CXXFLAGS     = -fPIC -m64 -funroll-loops -fprefetch-loop-arrays -mavx -flto -Wno-sign-compare -g -Wall -Wextra -Ofast -lrt -DNDEBUG -DHAVE_CXX0X -openmp -march=native -w -ftree-vectorize -ftree-vectorizer-verbose=0
CPPFLAGS     = -DFINTEGER=int  -fopenmp 
LDFLAGS      = -fopenmp  
LIBS         = -lopenblas

NEITEST = NeighborProp
INITEST = InvertedIndex
ABKMEANS = AblationOPTKmeans
HNSWPARA = TestHNSWPara
ABPS = AblationPS
ABNL = AblationNL
KNNDIST = KthNNDist
CANLIST = CandidateListSize
TRAINGT = TrainsetGroundtruth
COMPTEST = CompTest
NL = NL
HNSW = EvaluateHNSW
FAISS = EvaluateFaiss

nl: ${NL}
abkmeans: $(ABKMEANS)
ini: $(INITEST)
hnswpara: $(HNSWPARA)
abps: $(ABPS)
abnl: $(ABNL)
nei: $(NEITEST)
knnd: $(KNNDIST)
can: ${CANLIST}
trgt: $(TRAINGT)
comptest: $(COMPTEST)
hnsw: ${HNSW}
faiss: ${FAISS}


% :	%.cpp ./utils/utils.h ./hnswlib_old/hnswalg.cpp ../build/faiss/libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../ $(LIBS) ./complib/*.o  ./FastPFor/build/*.a

clean:
	rm -f $(MAINTEST)
	rm -f $(INITEST)
	rm -f $(ABKMEANS)
	rm -f $(HNSWPARA)
	rm -f $(ABPS)
	rm -f $(ABNL)
	rm -f ${NEITEST}
	rm -f $(KNNDIST)
	rm -f $(CANLIST)
	rm -f $(TRAINGT)
	rm -f $(COMPTEST)
	rm -f ${NL}
	rm -f ${HNSW}
	rm -f ${FAISS}

.PHONY: all file clean
