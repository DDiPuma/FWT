from functools import lru_cache
from math import log

import numpy as np

from utils import nextpow2, pairs, zero_pad


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


# Heavily based on Wavelets Made Easy, Algorithm 1.16
def inplace_fast_1d_haar_transform(signal):
    s = zero_pad(signal)
    n = int(log(len(s), 2))

    I = 1
    J = 2
    M = len(s)

    for _ in range(n):
        M = M // 2
        for K in range(M):
            s[J*K], s[J*K+I] = (s[J*K] + s[J*K+I]) / 2, (s[J*K] - s[J*K+I]) / 2
        I = J
        J *= 2

    return s

# Heavily based on Wavelets Made Easy, Algorithm 1.19
def inplace_inverse_fast_1d_haar_transform(signal):
    s = zero_pad(signal)

    n = int(log(len(s), 2))

    J = len(s)
    I = J // 2
    M = 1

    for _ in range(n):
        for K in range(M):
            s[J*K], s[J*K+I] = s[J*K] + s[J*K+I], s[J*K] - s[J*K+I]
        J = I
        I = I//2
        M *= 2

    return s


def fast_2d_haar_transform(matrix):
    first_transform = np.array([fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = np.array([fast_1d_haar_transform(col) for col in first_transform.T])
    return second_transform_T.T


def inplace_fast_2d_haar_transform(matrix):
    first_transform = np.array([inplace_fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = np.array([inplace_fast_1d_haar_transform(col) for col in first_transform.T])
    return second_transform_T.T


def inplace_inverse_fast_2d_haar_transform(matrix):
    first_transform = np.array([inplace_inverse_fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = np.array([inplace_inverse_fast_1d_haar_transform(col) for col in first_transform.T])
    return second_transform_T.T


HAAR_MATRIX_2x2 = np.array([[1, 1], [1, -1]])

"""
https://www.mathworks.com/matlabcentral/fileexchange/45247-code-for-generating-haar-matrix?fbclid=IwAR3iNpupw8mx7l763E8RDpyQdkmj7TGKOpJAs3FMfTCRa0Fh85AAQ_aUbV0
Output: 2**Nx2**N Haar matrix
"""
@lru_cache(20)
def haar_matrix(N):
    if N == 1:
        return HAAR_MATRIX_2x2
    else:
        top = np.kron(haar_matrix(N-1), np.array([1, 1]))
        bottom = np.kron(np.identity(2**(N-1)), np.array([1, -1]))
        return np.vstack((top, bottom))

