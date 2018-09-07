import numpy as np
import struct



def arrange(a, spc=3):
    """Convenience function which takes an array of images, and arranges them into a grid."""
    num_imgs, dim1, dim2 = a.shape
    num_rows = int(np.sqrt(num_imgs))
    num_cols = (num_imgs - 1) // num_rows + 1

    m = num_rows * dim1 + (num_rows + 1) * spc
    n = num_cols * dim2 + (num_cols + 1) * spc
    result = np.zeros((m, n))
    idx = 0
    for i in range(num_rows):
        rstart = i * dim1 + (i + 1) * spc
        for j in range(num_cols):
            cstart = j * dim2 + (j + 1) * spc
            if idx < num_imgs:
                result[rstart:rstart+dim1, cstart:cstart+dim2] = a[idx, :, :]
            idx += 1

    return result


mnist_cache = {}


def read_mnist_images(fname):
    if fname in mnist_cache:
        return mnist_cache[fname]
    
    data = open(fname).read()
    magic = struct.unpack('>i', data[:4])[0]
    assert magic == 2051
    num_images = struct.unpack('>i', data[4:8])[0]
    num_rows = struct.unpack('>i', data[8:12])[0]
    num_cols = struct.unpack('>i', data[12:16])[0]
    
    pixels = struct.unpack('B' * num_images * num_rows * num_cols, data[16:])
    pixels = np.array(pixels, dtype=float).reshape((num_images, num_rows * num_cols)) / 255.

    mnist_cache[fname] = pixels
    return pixels


def read_mnist_labels(fname):
    if fname in mnist_cache:
        return mnist_cache[fname]
    
    data = open(fname).read()
    magic = struct.unpack('>i', data[:4])[0]
    assert magic == 2049
    num_items = struct.unpack('>i', data[4:8])[0]
    labels = struct.unpack('B' * num_items, data[8:])

    mnist_cache[fname] = np.array(labels)
    return np.array(labels)

