from math import log

import numpy as np

from utils import pairs, zero_pad


# Heavily based on Wavelets Made Easy
def fast_1d_haar_transform(signal):
    s = zero_pad(signal)
    num_sweeps = int(log(len(signal), 2))

    a = s[:]
    new_a = s[:]

    for _ in range(num_sweeps):
        calculations = [((first+second)/2, (first-second)/2)
                        for first, second in pairs(new_a)]
        new_a, c = zip(*calculations)
        new_a = np.array(new_a)
        c = np.array(c)
        a[:len(new_a)] = new_a[:]
        a[len(new_a):len(new_a)+len(c)] = c[:]

    return a


def fast_2d_haar_transform(matrix):
    first_transform = np.array([fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = [fast_1d_haar_transform(col) for col in first_transform.T]
    return np.array(second_transform_T).T

