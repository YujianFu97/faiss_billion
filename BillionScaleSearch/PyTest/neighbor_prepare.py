import numpy as np
import faiss
import struct
       
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

dataset = 'SIFT1M'
path_base = '/home/yujianfu/Desktop/Dataset/' + dataset + '/' + dataset + '_base.fvecs'
path_neighbor = '/home/yujianfu/Desktop/Dataset/' + dataset + '/' + dataset + '_neighbor.ivecs'

if __name__ == '__main__':
    
    base_data = fvecs_read(path_base)
    d = np.shape(base_data)[1]
    res = faiss.StandardGpuResources()  # use a single GPU
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)


    gpu_index_flat.add(base_data)         # add vectors to the index
    print(gpu_index_flat.ntotal)

    k = 50 + 1                         # we want to see 4 nearest neighbors
    D, I = gpu_index_flat.search(base_data, k)  # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries

    


    '''
    neighbor_file = open(path_neighbor, 'wb')
    nb = np.shape(base_data)[0]
    for i in range (nb):
        neighbor_file.write(struct.pack('I', k))
        for j in range(k):
            neighbor_file.write(struct.pack('i', I[i, j]))
    neighbor_file.close()

    neighbor = ivecs_read(path_neighbor)
    print(neighbor)
    '''



