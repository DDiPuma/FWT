from math import log

import numpy as np

from utils import pairs, zero_pad, nextpow2


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
    n = int(log(len(signal), 2))
    s = zero_pad(signal)
    
    I = 1
    J = 2
    M = int(2**n)
    
    for L in range(1, n+1):
        M = M//2
        for K in range(M):
            a = (s[J*K] + s[J*K+I]) / 2
            c = (s[J*K] - s[J*K+I]) / 2
            s[J*K] = a
            s[J*K+I] = c
        I = J
        J = 2*J
        
    return s

# Heavily based on Wavelets Made Easy, Algorithm 1.19
def inplace_inverse_fast_1d_haar_transform(signal):
    n = int(log(len(signal), 2))
    s = zero_pad(signal)
    
    I = int(2**(n-1))
    J = 2*I
    M = 1
    
    for L in range(n+1, 1, -1):
        for K in range(M):
            a1 = s[J*K] + s[J*K+I]
            a2 = s[J*K] - s[J*K+I]
            s[J*K] = a1
            s[J*K+I] = a2
        J = I
        I = I//2
        M = 2*M

    return s


def fast_2d_haar_transform(matrix):
    first_transform = np.array([fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = [fast_1d_haar_transform(col) for col in first_transform.T]
    return np.array(second_transform_T).T


def inplace_fast_2d_haar_transform(matrix):
    first_transform = np.array([inplace_fast_1d_haar_transform(row) for row in matrix])
    second_transform_T = [inplace_fast_1d_haar_transform(col) for col in first_transform.T]
    return np.array(second_transform_T).T


"""
https://www.mathworks.com/matlabcentral/fileexchange/45247-code-for-generating-haar-matrix?fbclid=IwAR3iNpupw8mx7l763E8RDpyQdkmj7TGKOpJAs3FMfTCRa0Fh85AAQ_aUbV0
Input: N must be a power of 2
Output: NxN Haar matrix
"""
def generate_haar_matrix(N):
    p = [0, 0]
    q = [0, 1]
    n = nextpow2(N)
    
    for i in range(1, n-2):
        p.extend(2**i*[i])
        t = list(range(1, 2**i+1))
        q.extend(t)

    Hr = [[0 for _ in range(N)] for _ in range(N)]
    Hr[0][:] = N*[1]

    for i in range(1, N):
        P = p[i]
        Q = q[i]
        for j in range(int((N*(Q-1)/(2**P))), int(N*((Q-0.5)/(2**P)))):
            Hr[i][j] = 2**(P/2)
        for j in range(int(N*((Q-0.5)/(2**P))), int(N*(Q/(2**P)))):
            Hr[i][j] = -(2**(P/2))
            
#     print(Hr) # Easier to look at version
    Hr = [[i*(N**-.5) for i in j] for j in Hr]
    return Hr

