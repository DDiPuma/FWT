from math import ceil, log

import numpy as np


def pairs(arr):
    return zip(arr[::2], arr[1::2])


def zero_pad(signal):
    power = log(len(signal), 2)
    power = ceil(power)

    if len(signal) == 2**power:
        return signal
    else:
        return np.concatenate(signal, np.zeros((2**power - len(signal), 1)))

