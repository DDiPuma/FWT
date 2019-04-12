from haar import *
import matplotlib.pyplot as plt
import timeit
import functools
from random import shuffle


def one_dim_forward_benchmark():
    ordered_timings = []
    inplace_timings = []
    matrix_timings = []
    data_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    for data_size in data_sizes:
        data_array = list(range(data_size))
        ordered_timer = timeit.Timer(functools.partial(ordered_fast_1d_haar_transform, data_array[:], 0, len(data_array) - 1))
        ordered_timings.append(ordered_timer.timeit(5))
        inplace_timer = timeit.Timer(functools.partial(inplace_timings, data_array[:]))
        inplace_timings.append(inplace_timer.timeit(5))
        matrix_timer = timeit.Timer(functools.partial(matrix_1d_haar_transform, data_array[:], 0, len(data_array)))
        matrix_timings.append(matrix_timer.timeit(5))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_timings, inplace_timings, matrix_timings):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    ax = plt.subplot2grid((2, 3), (1, 0), colspan=4, rowspan=2)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    plt.subplot2grid((2, 3), (0, 0))
    plt.plot(data_sizes, ordered_timings)
    plt.title("Ordered Haar Transform", y=1.08)
    plt.xlabel("Input Size (Length of Array)")
    plt.ylabel("Input Length")
    plt.subplot2grid((2, 3), (0, 1))
    plt.plot(data_sizes, inplace_timings)
    plt.title("In-Place Haar Transform", y=1.08)
    plt.xlabel("Input Size (Length of Array)")
    plt.ylabel("Input Length")
    plt.subplot2grid((2, 3), (0, 2))
    plt.plot(data_sizes, matrix_timings)
    plt.title("Matrix Haar Transform", y=1.08)
    plt.xlabel("Input Size (Length of Array)")
    plt.ylabel("Input Length")

    fig.set_size_inches(w=12, h=10)
    plt.show()
