import numpy as np


def ravel_data(x: np.ndarray):
    return x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
