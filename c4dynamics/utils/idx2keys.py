# pragma: no cover


import numpy as np


def idx2keys(xdict, idx):
    return np.array(list(xdict.keys()))[idx]
