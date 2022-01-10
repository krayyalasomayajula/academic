import numpy as np

def premute(x , in_shape, out_shape):
    np.moveaxis(x, [0, 1, 2], [-1, -2, -3])