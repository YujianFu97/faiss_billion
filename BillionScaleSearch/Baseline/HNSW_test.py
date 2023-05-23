import os
import hnswlib
import numpy as np
import time

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

dim = 128
num_elements = 100000000
M = 16
ef_construction = 100
ef_search = 10
K = 1
data = "SIFT"
data_size = "100M"
dataset_path = "/data/yujian/Dataset/"

dataset_path = dataset_path + data + "/" + data + data_size + "/" + data + data_size + "_base.fvecs"
index_path = dataset_path + data + "/" + data + data_size + "/" + data + data_size + "_HNSW_efcon_"+str(ef_construction)+"_M_"+str(M)+".bin"
queryset_path = dataset_path + data + "/" + data + data_size + "/" + data + data_size + "_query.fvecs"
gt_path= dataset_path + data + "/" + data + data_size + "/" + data + data_size + "_GT_100"

print(gt_path)
assert(os.path.exists(gt_path))
gt, dist = gtbin_read(gt_path)
print(gt)
exit(0)
if (os.path.exists(index_path)):
    # Declaring index
    HNSW = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

    # Load the data for construction
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
    HNSW.set_ef(ef_search)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    print("Adding %d elements from the database set" % (len(dataset)))
    HNSW.add_items(dataset)
    
    # Serializing and deleting the index:
    print("Saving index to '%s'" % index_path)
    HNSW.save_index(index_path)

# Reiniting, loading the index
HNSW = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
# Load the constructed index
print("\nLoading index from "+index_path)
HNSW.load_index("first_half.bin", max_elements = num_elements)

queryset = fvecs_read(queryset_path)
# Query the elements for themselves and measure recall:
HNSW.set_num_threads(1)
start = time.perf_counter()
labels, distances = HNSW.knn_query(queryset, k=K)


print("Recall for query set: Number of quries:",   np.mean(labels.reshape(-1) == np.arange(len(queryset))), "\n")
