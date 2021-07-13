import numpy as np

def rescale(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    if abs(max - min) < 1e-5:
        return x
    result = (x - min) / (max - min)
    return result
