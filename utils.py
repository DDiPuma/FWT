from math import ceil, log

import numpy as np


def pairs(arr):
    """
    Return an iterable of tuples of adjacent pairs from an array.

    Notes
    -----
    I wouldn't run this on an array whose size is not divisible by two.
    """
    return zip(arr[::2], arr[1::2])


def zero_pad(signal):
    """
    Extend a signal to length power of two by appending zeros to the end of it.
    """
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
    """
    Calculate the next power of two beyond an integer.
    """
    n = 1
    while n < i: n *= 2
    return n
