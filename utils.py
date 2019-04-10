from itertools import repeat
from math import ceil, log

def pairs(arr):
    return zip(arr[::2], arr[1::2])


def transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def zero_pad(signal):
    power = log(len(signal), 2)
    power = ceil(power)

    if len(signal) == 2**power:
        return signal
    else:
        return signal + list(repeat(0, 2**power - len(signal)))

