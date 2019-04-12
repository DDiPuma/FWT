import functools
import timeit

import matplotlib.pyplot as plt

from haar import *


def one_dim_forward_benchmark():
    ordered_time_averages = []
    inplace_time_averages = []
    matrix_time_averages = []

    ordered_time_mins = []
    inplace_time_mins = []
    matrix_time_mins = []

    ordered_time_maxes = []
    inplace_time_maxes = []
    matrix_time_maxes = []

    ordered_time_std_devs = []
    inplace_time_std_devs = []
    matrix_time_std_devs = []

    data_sizes = [2**N for N in range(1, 13)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []
        matrix_timings = []

        for trial in range(25):
            data_array = np.random.randint(256, size=data_size)

            ordered_timer = timeit.Timer(functools.partial(ordered_fast_1d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(3))

            inplace_timer = timeit.Timer(functools.partial(inplace_fast_1d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(3))

            matrix_timer = timeit.Timer(functools.partial(matrix_1d_haar_transform, data_array))
            matrix_timings.append(matrix_timer.timeit(3))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        matrix_time_averages.append(np.mean(matrix_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        matrix_time_maxes.append(max(matrix_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        matrix_time_mins.append(min(matrix_timings))
        ordered_time_std_devs.append(np.std(ordered_timings))
        inplace_time_std_devs.append(np.std(inplace_timings))
        matrix_time_std_devs.append(np.std(matrix_timings))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages, matrix_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.plot(np.log2(data_sizes), np.log2(ordered_time_averages), "b-", label="ordered (fast)")
    plt.plot(np.log2(data_sizes), np.log2(inplace_time_averages), "r-", label="inplace (fast)")
    plt.plot(np.log2(data_sizes), np.log2(matrix_time_averages), "k-", label="matrix (slow)")
    plt.xlabel("$log_2$ size (Length of Input Array)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


if __name__ == '__main__':
    one_dim_forward_benchmark()
