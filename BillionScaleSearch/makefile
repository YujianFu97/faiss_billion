# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CXX          = g++ -std=c++11
CXXFLAGS     = -fPIC -m64 -mavx -Wno-sign-compare -g -O3 -Wall -Wextra
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

% :	%.cpp ./Optkmeans/Optkmeans.cpp ./Index/BIndex.cpp ./utils/utils.h ./hnswlib/hnswalg.cpp ./NCSearch/NCSearch.cpp ../build/faiss/libfaiss.a 
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

.PHONY: all file clean
