from math import log

from utils import pairs, transpose, zero_pad


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
        a[:len(new_a)] = new_a[:]
        a[len(new_a):len(new_a)+len(c)] = c[:]

    return a


def fast_2d_haar_transform(matrix):
    first_transform = [fast_1d_haar_transform(row) for row in matrix]
    first_transform_T = transpose(first_transform)
    second_transform_T = [fast_1d_haar_transform(col) for col in first_transform_T]
    return transpose(second_transform_T)

