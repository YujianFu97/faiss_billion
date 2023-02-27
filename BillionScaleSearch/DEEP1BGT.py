import numpy as np
import struct

def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


filename = "/home/yujianfu/Downloads/groundtruth.public.10K.ibin"
data = read_ibin(filename)

neighbor_file = open("/home/yujianfu/Downloads/DEEP1B_groundtruth.ivecs", 'wb')
nb = np.shape(data)[0]
k = np.shape(data)[1]
print(nb, k)
for i in range (nb):
    neighbor_file.write(struct.pack('I', k))
    for j in range(k):
        neighbor_file.write(struct.pack('i', data[i, j]))
neighbor_file.close()



