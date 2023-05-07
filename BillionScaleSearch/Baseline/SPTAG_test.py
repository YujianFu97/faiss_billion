import os
import shutil
import numpy as np
import sptag

vector_number = 1000
vector_dimension = 100

# Randomly generate the database vectors. Currently SPTAG only support int8, int16 and float32 data type.
x = np.random.rand(vector_number, vector_dimension).astype(np.float32) 

# Prepare metadata for each vectors, separate them by '\n'. Currently SPTAG python wrapper only support '\n' as the separator
m = ''
for i in range(vector_number):
    m += str(i) + '\n'



index = sptag.AnnIndex('BKT', 'Float', vector_dimension)

# Set the thread number to speed up the build procedure in parallel 
index.SetBuildParam("NumberOfThreads", '4', "Index")

# Set the distance type. Currently SPTAG only support Cosine and L2 distances. Here Cosine distance is not the Cosine similarity. The smaller Cosine distance it is, the better.
index.SetBuildParam("DistCalcMethod", 'L2', "Index") 

if (os.path.exists("sptag_index")):
    shutil.rmtree("sptag_index")
if index.BuildWithMetaData(x, m, vector_number, False, False):
    index.Save("sptag_index") # Save the index to the disk

os.listdir('sptag_index')