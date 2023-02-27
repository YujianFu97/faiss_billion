import numpy as np
import faiss

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


x = bvecs_read("/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs")
n = x.shape[0]
x = x[:int(n/2),:]
x = np.ascontiguousarray(x.astype('float32'))

#x = np.random.rand(10000000,128)
ncentroids = 1000000
niter = 30
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu= True)
kmeans.train(x)

