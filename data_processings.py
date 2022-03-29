import numpy as np
import gzip


def load_mnist(data_file, label_file):
    with gzip.open(data_file, 'rb') as f:
        magic_num = int(f.read(4).hex(), 16)  # 读取出的四个字节是16进制，需要转成10进制
        image_num = int(f.read(4).hex(), 16)
        image_width = int(f.read(4).hex(), 16)
        image_height = int(f.read(4).hex(), 16)
        img_data = np.frombuffer(f.read(), dtype='uint8')  # 将剩余所有数据一次读取至numpy数组中
        img_data = img_data.reshape(image_num, image_width, image_height)

    with gzip.open(label_file, 'rb') as f:
        magic_num = int(f.read(4).hex(), 16)
        label_num = int(f.read(4).hex(), 16)
        label_data = np.frombuffer(f.read(), dtype='uint8')

    return img_data, label_data


def linearInd2Binary(ind, n):
    """
    covert the labels to binary form
    :param ind:
    :param n:
    :return:
    """
    l = len(ind)
    result = np.zeros((l, n))
    for i in range(l):
        result[i, ind[i] - 1] = 1
    return result


def standardize_cols(X, mu=None, sigma=None):
    """
    standardize the data
    """
    if mu is None and sigma is None:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 2e-16] = 1

    X = (X - mu) / sigma
    return X, mu, sigma

