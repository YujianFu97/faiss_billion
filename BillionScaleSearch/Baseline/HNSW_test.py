import os
import hnswlib
import numpy as np
import time
from memory_profiler import memory_usage

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def bvecs_read(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def gtbin_read(fname):
    x = np.fromfile(fname, dtype='int32')
    num_points = x[0]
    num_gt = x[1]
    gt = x[2:2 + num_points * num_gt].view(np.uint32).reshape(-1, num_gt)
    dist = x[2 + num_points * num_gt : ].view('float32').reshape(-1, num_gt)
    return gt, dist
    
    
"""
Example of index building, search and serialization/deserialization
"""
data = "SIFT"
data_size = "100M"
dim = 128
num_elements = 100000000

M = 16
ef_construction = 200
ef_search_ini = 100
K = 1
ef_search_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 
                  380, 390, 400]

data_path = "/data/yujian/Dataset/"
dataset_path = data_path + data + "/" + data + data_size + "/" + data + data_size + "_base.fvecs"
index_path = data_path + data + "/" + data + data_size + "/" + data + data_size + "_HNSW_efcon_"+str(ef_construction)+"_M_"+str(M)+".bin"
queryset_path = data_path + data + "/" + data + data_size + "/" + data + data_size + "_query.fvecs"
gt_path= data_path + data + "/" + data + data_size + "/" + data + data_size + "_GT_100"

def BuildIndex():
    assert(os.path.exists(gt_path))
    gt, dist = gtbin_read(gt_path)

    if not (os.path.exists(index_path)):
        # Declaring index
        HNSW = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Load the data for construction
        assert(os.path.exists(dataset_path))
        print("Loading base dataset from ", dataset_path)
        dataset = fvecs_read(dataset_path)

        # Initing index
        # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
        # during insertion of an element.
        # The capacity can be increased by saving/loading the index, see below.
        #
        # ef_construction - controls index search speed/build speed tradeoff
        #
        # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
        # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
        HNSW.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        HNSW.set_ef(ef_search_ini)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        start_time = time.time()
        print("Adding ", (len(dataset)), " elements from the database set with ef_Cons = ", ef_construction, " M = ", M)
        HNSW.set_num_threads(64)
        HNSW.add_items(dataset)
        end_time = time.time()
        print("The time construction of HNSW graph: ", end_time - start_time, "s")

        # Serializing and deleting the index:
        print("Saving index to '%s'" % index_path)
        HNSW.save_index(index_path)

    # Reiniting, loading the index
    HNSW = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
    # Load the constructed index
    print("\nLoading index from "+index_path)
    assert(os.path.exists(index_path))
    HNSW.load_index(index_path, max_elements = num_elements)

    assert(os.path.exists(queryset_path))
    print("Loading the query vectors")
    queryset = fvecs_read(queryset_path)
    # Query the elements for themselves and measure recall:
    HNSW.set_num_threads(1)

    for ef_search in ef_search_list:
        print("Searching the query vectors")
        HNSW.set_ef(ef_search)
        start = time.perf_counter()
        labels, distances = HNSW.knn_query(queryset, k=K)
        end = time.perf_counter()
        elapsed = end - start
        print("The time consumption for each query: ", 1000 * elapsed / len(queryset), "ms")

        correct = 0
        for i in range(len(queryset)):
            gt_set = set(gt[i][:K])
            label_set = set(labels[i][:K])
            correct += len(gt_set.intersection(label_set))

        print("Recall for query set: Number of quries:", len(queryset), "Recall", K, "@", K, "=", correct / (len(queryset) * K) ," efQuery = ", ef_search)

mem_usage = memory_usage(BuildIndex)
#print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
print('Maximum memory usage: %s' % max(mem_usage), " KB ")