import numpy as np

'''
def fvecs_read(filename):
    ffile = np.fromfile(filename, dtype = np.float32)
    if ffile.size == 0:
        return zeros((0, 0))
    dimension = ffile.view(np.int32)[0]
    assert dimension > 0
    ffile = ffile.reshape(-1, 1+dimension)
    if not all(ffile.view(np.int32)[:, 0] == dimension):
        raise IOError('Non-uniform vector sizes in ' + filename)
    ffile = ffile[:, 1:]
    ffile = ffile.copy()
    return ffile
'''

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