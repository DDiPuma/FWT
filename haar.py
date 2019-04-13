from functools import lru_cache
from math import log

import numpy as np

from utils import nextpow2, pairs, zero_pad


# Heavily based on Wavelets Made Easy
def ordered_fast_1d_haar_transform(signal):
    s = zero_pad(signal)
    num_sweeps = int(log(len(s), 2))

    a = s.copy()
    new_a = s.copy()

    for _ in range(num_sweeps):
        calculations = [((first+second)/2, (first-second)/2)
                        for first, second in pairs(new_a)]
        new_a, c = zip(*calculations)
        new_a = np.array(new_a)
        c = np.array(c)
        a[:len(new_a)] = new_a[:]
        a[len(new_a):len(new_a)+len(c)] = c[:]

    return a


def ordered_inverse_fast_1d_haar_transform(signal):
    s = zero_pad(signal)
    num_sweeps = int(log(len(s), 2))

    a = s.copy()

    # This algorithm starts by modifying 2 entries, then 4, then 8, and so on
    # The new entries are sums and differences of pairs of elements
    # BUT the pairs of elements are strided apart
    # The stride is 1, 2, 4, ...
    for stride_pow in range(num_sweeps):
        stride = 2**stride_pow
        size_a = 2*stride
        new_a = a[:size_a].copy()

        for i in range(stride):
            new_a[2*i], new_a[2*i+1] = a[i]+a[i+stride], a[i]-a[i+stride]

        a[:size_a] = new_a[:]

    return a


def ordered_fast_2d_haar_transform(matrix):
    first_transform = np.array([ordered_fast_1d_haar_transform(row.copy()) for row in matrix])
    second_transform_T = np.array([ordered_fast_1d_haar_transform(col.copy()) for col in first_transform.T])
    return second_transform_T.T


def ordered_inverse_fast_2d_haar_transform(matrix):
    first_transform = np.array([ordered_inverse_fast_1d_haar_transform(row.copy()) for row in matrix])
    second_transform_T = np.array([ordered_inverse_fast_1d_haar_transform(col.copy()) for col in first_transform.T])
    return second_transform_T.T


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
        I = I // 2
        M *= 2

    return s


def inplace_fast_2d_haar_transform(matrix):
    if matrix.shape[0] == 1:
        return matrix.copy()

    first_transform = np.array([inplace_fast_1d_haar_transform(row.copy()) for row in matrix])
    second_transform_T = np.array([inplace_fast_1d_haar_transform(col.copy()) for col in first_transform.T])
    transform = second_transform_T.T

    return transform


def inplace_inverse_fast_2d_haar_transform(matrix):
    first_transform = np.array([inplace_inverse_fast_1d_haar_transform(row.copy()) for row in matrix])
    second_transform_T = np.array([inplace_inverse_fast_1d_haar_transform(col.copy()) for col in first_transform.T])
    return second_transform_T.T


HAAR_MATRIX_2x2 = np.array([[1, 1], [1, -1]], dtype=float)

"""
https://www.mathworks.com/matlabcentral/fileexchange/45247-code-for-generating-haar-matrix?fbclid=IwAR3iNpupw8mx7l763E8RDpyQdkmj7TGKOpJAs3FMfTCRa0Fh85AAQ_aUbV0
Output: 2**Nx2**N Haar matrix
"""
@lru_cache(20)
def haar_matrix(N):
    if N == 1:
        return HAAR_MATRIX_2x2
    else:
        top = np.kron(haar_matrix(N-1), np.array([1, 1], dtype=float))
        bottom = np.kron(np.identity(2**(N-1)), np.array([1, -1], dtype=float))
        return np.vstack((top, bottom))


@lru_cache(20)
def haar_transform_matrix(N):
    # Just normalize the Haar matrix
    H = haar_matrix(N)
    H_normalized = H.copy()
    for idx in range(H.shape[0]):
        row = H[idx, :]
        scale = np.linalg.norm(row, ord=2) 
        H_normalized[idx, :] = row / scale
    return H_normalized


def matrix_1d_haar_transform(signal):
    s = zero_pad(signal)
    N = int(log(len(s), 2))

    return haar_transform_matrix(N) @ s


def matrix_inverse_1d_haar_transform(signal):
    s = zero_pad(signal)
    N = int(log(len(s), 2))

    return  haar_transform_matrix(N).T @ s


def matrix_2d_haar_transform(signal):
    N = int(log(signal.shape[0], 2))

    H = haar_transform_matrix(N)

    return H @ signal @ H.T


def matrix_inverse_2d_haar_transform(signal):
    N = int(log(signal.shape[0], 2))

    H = haar_transform_matrix(N)

    return H.T @ signal @ H

