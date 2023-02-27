#include "Index/BIndex.h"
#include "parameters/MillionScale/ParameterResults.h"

// Basic Theory

int main(){
    BIndex * Index = new BIndex(Dimension, nb, nc, nt, SavingIndex, Recording, Retrain, UseOPQ, M_PQ, CodeBits);

    Index->TestDistBound(K, ngt, nq, PathQuery, PathGt, PathBase);

    return 0;
}