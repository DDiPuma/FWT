from math import ceil, log

import numpy as np


def pairs(arr):
    return zip(arr[::2], arr[1::2])


def strided_pair_indices(stride):
    return [(i, i+stride) for i in range(stride)]


def zero_pad(signal):
    power = log(len(signal), 2)
    power = ceil(power)

    if len(signal) == 2**power:
        return signal.copy()
    else:
        return np.concatenate(signal, np.zeros((2**power - len(signal), 1)))

"""
https://wiki.python.org/moin/NumericAndScientificRecipes
Find 2^n that is equal to or greater than.
"""
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
