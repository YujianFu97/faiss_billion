import struct
import numpy as np

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


def load_dataset(filename):
    with open(filename, 'rb') as f:
        num_points, num_dimensions = struct.unpack('II', f.read(8))
        
        if filename.endswith('.fbin'):
            dtype = 'float32'
        elif filename.endswith('.u8bin'):
            dtype = 'uint8'
        elif filename.endswith('.i8bin'):
            dtype = 'int8'
        else:
            raise ValueError("Invalid file extension")
        
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape((num_points, num_dimensions))
    return data

def load_ground_truth(filename):
    with open(filename, 'rb') as f:
        num_queries, K = struct.unpack('II', f.read(8))
        neighbor_ids = np.frombuffer(f.read(num_queries * K * 4), dtype='uint32')
        neighbor_ids = neighbor_ids.reshape((num_queries, K))
        distances = np.frombuffer(f.read(num_queries * K * 4), dtype='float32')
        distances = distances.reshape((num_queries, K))
    
    return neighbor_ids, distances

def vecs_write(fname, data):
    n_vectors, dim = data.shape
    dim_bytes = np.array([dim], dtype=np.uint32).tobytes()

    with open(fname, 'wb') as f:
        for vec in data:
            f.write(dim_bytes + vec.tobytes())


if __name__ == "__main__":

    # Test dataset loading
    #dataset_file = '/dev/shm/yujian/Data/SIFT1B/query.public.10K.u8bin'  # Replace with actual filename
    dataset_file = '/data00/yujian/Data/ANNS/DEEP1B/base.1B.fbin'
    data = load_dataset(dataset_file)
    print(data[100])
    print(f"Dataset shape: {data.shape}")

    # Test vecs_write
    #vecs_filename = dataset_file.rsplit('.', 1)[0] + '.vecs'  # Change the extension to .vecs
    vecs_filename = '/dev/shm/yujian/Data/DEEP1B_base.vecs'
    vecs_write(vecs_filename, data)
    print(f"Written to {vecs_filename}")

    data = fvecs_read(vecs_filename)
    print(data[100])
    print(f"Dataset shape: {data.shape}")

    exit()

    # Test ground truth loading
    ground_truth_file = '/dev/shm/yujian/Data/GT_1B/bigann-1B'  # Replace with actual filename
    neighbor_ids, distances = load_ground_truth(ground_truth_file)
    print(f"Neighbor IDs shape: {neighbor_ids.shape}")
    print(f"Distances shape: {distances.shape}")
    gt_filename = ground_truth_file + "-gt.ivecs"
    gt_dist_filename = ground_truth_file + "-gt-dist.fvecs"
    vecs_write(gt_filename, neighbor_ids)
    vecs_write(gt_dist_filename, distances)
    exit()

