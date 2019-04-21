import functools
import timeit

import matplotlib.pyplot as plt

from haar import *
import compression


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

    data_sizes = [2**N for N in range(1, 14)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []
        matrix_timings = []

        for trial in range(25):
            data_array = np.random.randint(256, size=data_size).astype(np.float32)

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
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))
        matrix_time_std_devs.append(np.std(np.log2(matrix_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages, matrix_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(matrix_time_averages), yerr=matrix_time_std_devs, fmt="k-", label="matrix (slow)")
    plt.xlabel("$log_2$ size (Length of Input Array)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def fast_one_dim_forward_benchmark():
    ordered_time_averages = []
    inplace_time_averages = []

    ordered_time_mins = []
    inplace_time_mins = []

    ordered_time_maxes = []
    inplace_time_maxes = []

    ordered_time_std_devs = []
    inplace_time_std_devs = []

    data_sizes = [2**N for N in range(1, 20)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []

        for trial in range(25):
            data_array = np.random.randint(256, size=data_size).astype(np.float32)

            ordered_timer = timeit.Timer(functools.partial(ordered_fast_1d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(3))

            inplace_timer = timeit.Timer(functools.partial(inplace_fast_1d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(3))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Fast Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.xlabel("$log_2$ size (Length of Input Array)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()

def one_dim_inverse_benchmark():
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

    data_sizes = [2**N for N in range(1, 14)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []
        matrix_timings = []

        for trial in range(25):
            data_array = np.random.randint(256, size=data_size).astype(np.float32)

            ordered_timer = timeit.Timer(functools.partial(ordered_inverse_fast_1d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(3))

            inplace_timer = timeit.Timer(functools.partial(inplace_inverse_fast_1d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(3))

            matrix_timer = timeit.Timer(functools.partial(matrix_inverse_1d_haar_transform, data_array))
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
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))
        matrix_time_std_devs.append(np.std(np.log2(matrix_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages, matrix_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Inverse Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(matrix_time_averages), yerr=matrix_time_std_devs, fmt="k-", label="matrix (slow)")
    plt.xlabel("$log_2$ size (Length of Input Array)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def fast_one_dim_inverse_benchmark():
    ordered_time_averages = []
    inplace_time_averages = []

    ordered_time_mins = []
    inplace_time_mins = []

    ordered_time_maxes = []
    inplace_time_maxes = []

    ordered_time_std_devs = []
    inplace_time_std_devs = []

    data_sizes = [2**N for N in range(1, 20)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []

        for trial in range(25):
            data_array = np.random.randint(256, size=data_size).astype(np.float32)

            ordered_timer = timeit.Timer(functools.partial(ordered_inverse_fast_1d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(3))

            inplace_timer = timeit.Timer(functools.partial(inplace_inverse_fast_1d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(3))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("1D Fast Inverse Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.xlabel("$log_2$ size (Length of Input Array)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def two_dim_forward_benchmark():
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

    data_sizes = [2**N for N in range(1, 12)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []
        matrix_timings = []

        for trial in range(10):
            data_array = np.random.randint(256, size=(data_size,data_size)).astype(np.float32)

            ordered_timer = timeit.Timer(functools.partial(ordered_fast_2d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(2))

            inplace_timer = timeit.Timer(functools.partial(inplace_fast_2d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(2))

            matrix_timer = timeit.Timer(functools.partial(matrix_2d_haar_transform, data_array))
            matrix_timings.append(matrix_timer.timeit(2))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        matrix_time_averages.append(np.mean(matrix_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        matrix_time_maxes.append(max(matrix_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        matrix_time_mins.append(min(matrix_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))
        matrix_time_std_devs.append(np.std(np.log2(matrix_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages, matrix_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("2D Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(matrix_time_averages), yerr=matrix_time_std_devs, fmt="k-", label="matrix (slow)")
    plt.xlabel("$log_2$ N (for N x N Input Matrix)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def two_dim_inverse_benchmark():
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

    data_sizes = [2**N for N in range(1, 12)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []
        matrix_timings = []

        for trial in range(10):
            data_array = np.random.randint(256, size=(data_size, data_size)).astype(np.float32)

            ordered_timer = timeit.Timer(functools.partial(ordered_inverse_fast_2d_haar_transform, data_array))
            ordered_timings.append(ordered_timer.timeit(2))

            inplace_timer = timeit.Timer(functools.partial(inplace_inverse_fast_2d_haar_transform, data_array))
            inplace_timings.append(inplace_timer.timeit(2))

            matrix_timer = timeit.Timer(functools.partial(matrix_inverse_2d_haar_transform, data_array))
            matrix_timings.append(matrix_timer.timeit(2))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        matrix_time_averages.append(np.mean(matrix_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        matrix_time_maxes.append(max(matrix_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        matrix_time_mins.append(min(matrix_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))
        matrix_time_std_devs.append(np.std(np.log2(matrix_timings)))

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime', 'Matrix Haar Transform Runtime')
    rows = ["Input Length = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages, matrix_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("2D Inverse Haar Transform Algorithm Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-", label="ordered (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-", label="inplace (fast)")
    plt.errorbar(np.log2(data_sizes), np.log2(matrix_time_averages), yerr=matrix_time_std_devs, fmt="k-", label="matrix (slow)")
    plt.xlabel("$log_2$ N (for N x N Input Matrix)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def compression_benchmark():
    ordered_time_averages = []
    inplace_time_averages = []

    ordered_time_mins = []
    inplace_time_mins = []

    ordered_time_maxes = []
    inplace_time_maxes = []

    ordered_time_std_devs = []
    inplace_time_std_devs = []

    data_sizes = [2**N for N in range(6, 11)]

    for data_size in data_sizes:
        ordered_timings = []
        inplace_timings = []

        for trial in range(3):
            data_array = np.random.randint(256, size=(data_size, data_size)).astype(np.float32)
            compressor = compression.HaarImageCompressor(compression_method="ordered", target_compression_ratio=0)
            compressor.uncompressed_image = data_array

            matrix_timer = timeit.Timer(functools.partial(compressor.compress_image))
            ordered_timings.append(matrix_timer.timeit(1))
            compressor.compression_method = "inplace"
            matrix_timer = timeit.Timer(functools.partial(compressor.compress_image))
            inplace_timings.append(matrix_timer.timeit(1))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))

    data_sizes = [(2**N)**2 for N in range(6, 11)]

    columns = ('Ordered Haar Transform Runtime', 'In-Place Haar Transform Runtime')
    rows = ["Number of Pixels (N) = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("Image Compression Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(np.log2(data_sizes), np.log2(ordered_time_averages), yerr=ordered_time_std_devs, fmt="b-",
                 label="Transform method = ordered")
    plt.errorbar(np.log2(data_sizes), np.log2(inplace_time_averages), yerr=inplace_time_std_devs, fmt="r-",
                 label="Transform method =  inplace")

    plt.xlabel("$log_2$ N (number of pixels)")
    plt.ylabel("$log_2$ time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def compression_ratio_benchmark():
    ordered_time_averages = []
    inplace_time_averages = []

    ordered_time_mins = []
    inplace_time_mins = []

    ordered_time_maxes = []
    inplace_time_maxes = []

    ordered_time_std_devs = []
    inplace_time_std_devs = []

    data_sizes = [2 ** N for N in range(0, 11)]

    for data_size in data_sizes:
        print(data_size)
        ordered_timings = []
        inplace_timings = []

        for trial in range(10):
            compressor = compression.HaarImageCompressor(compression_method="ordered", target_compression_ratio=data_size)
            compressor.load_image("cam.png")

            matrix_timer = timeit.Timer(functools.partial(compressor.compress_image))
            ordered_timings.append(matrix_timer.timeit(1))
            compressor.compression_method = "inplace"
            matrix_timer = timeit.Timer(functools.partial(compressor.compress_image))
            inplace_timings.append(matrix_timer.timeit(1))

        ordered_time_averages.append(np.mean(ordered_timings))
        inplace_time_averages.append(np.mean(inplace_timings))
        ordered_time_maxes.append(max(ordered_timings))
        inplace_time_maxes.append(max(inplace_timings))
        ordered_time_mins.append(min(ordered_timings))
        inplace_time_mins.append(min(inplace_timings))
        ordered_time_std_devs.append(np.std(np.log2(ordered_timings)))
        inplace_time_std_devs.append(np.std(np.log2(inplace_timings)))

    columns = ('Compression w/ Ordered Haar Transform Runtime', 'Compression w/ In-Place Haar Transform Runtime')
    rows = ["Target Compression Ratio = {}".format(x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(ordered_time_averages, inplace_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("Image Compression Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(data_sizes, ordered_time_averages, yerr=ordered_time_std_devs, fmt="b-",
                 label="Transform method = ordered")
    plt.errorbar(data_sizes, inplace_time_averages, yerr=inplace_time_std_devs, fmt="r-",
                 label="Transform method =  inplace")

    plt.xlabel("Compression Ratio")
    plt.ylabel("time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def iterative_vs_recursive_haar():
    iterative_time_averages = []
    recursive_time_averages = []

    iterative_time_mins = []
    recursive_time_mins = []

    iterative_time_maxes = []
    recursive_time_maxes = []

    iterative_time_std_devs = []
    recursive_time_std_devs = []

    data_sizes = [N for N in range(1, 14)]

    for data_size in data_sizes:
        iterative_timings = []
        recursive_timings = []

        for trial in range(2):
            data_array = np.random.randint(256, size=2**data_size).astype(np.float32)

            matrix_timer = timeit.Timer(functools.partial(matrix_1d_haar_transform, data_array))
            recursive_timings.append(matrix_timer.timeit(3))

            matrix_timer = timeit.Timer(functools.partial(haar_matrix_old, 2**data_size))
            iterative_timings.append(matrix_timer.timeit(3))


        iterative_time_averages.append(np.mean(iterative_timings))
        recursive_time_averages.append(np.mean(recursive_timings))
        iterative_time_maxes.append(max(iterative_timings))
        recursive_time_maxes.append(max(recursive_timings))
        iterative_time_mins.append(min(iterative_timings))
        recursive_time_mins.append(min(recursive_timings))
        iterative_time_std_devs.append(np.std(np.log2(iterative_timings)))
        recursive_time_std_devs.append(np.std(np.log2(recursive_timings)))

    columns = ('Recursive Matrix Generation', 'Iterative Matrix Generation')
    rows = ["Matrix Size = N x N = {} x {}".format(2**x, 2**x) for x in data_sizes]
    cell_text = []
    for time_tuple in zip(recursive_time_averages, iterative_time_averages):
        cell_text.append(["{} seconds".format(time_data) for time_data in time_tuple])

    fig = plt.figure(1)
    plt.suptitle("Matrix Generation Runtimes")
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)

    plt.subplot(211)
    plt.errorbar(data_sizes, np.log2(recursive_time_averages), yerr=recursive_time_std_devs, fmt="b-",
                 label="Recursive")
    plt.errorbar(data_sizes, np.log2(iterative_time_averages), yerr=iterative_time_std_devs, fmt="r-",
                 label="Iterative")

    plt.xlabel("N")
    plt.ylabel("time (s)")
    plt.legend()

    ax = plt.subplot(212)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='upper center')
    ax.axis("off")

    fig.set_size_inches(w=12, h=10)
    plt.show()


def plot_haar_vectors():
    matrix = haar_matrix(3)
    fig, ax = plt.subplots(4, 2)

    for i, (row, x,y) in enumerate(zip(matrix,[0,1,2,3,0,1,2,3],[0,0,0,0,1,1,1,1])):
        print(row)
        ax[x,y].set_title("Vector {}".format(i))
        ax[x,y].plot(row, drawstyle='steps-pre')
    plt.suptitle("Haar Wavelets in $R^8$")
    plt.show()
    # ax[0, 0].plot(range(10), 'b') #row=0, col=0
    # ax[1, 0].plot(range(10), 'b') #row=1, col=0
    # ax[2, 0].plot(range(10), 'b') #row=1, col=0
    # ax[3, 0].plot(range(10), 'b') #row=1, col=0
    # ax[0, 1].plot(range(10), 'b') #row=0, col=1
    # ax[1, 1].plot(range(10), 'b') #row=1, col=1
    # ax[2, 1].plot(range(10), 'b') #row=1, col=1
    # ax[3, 1].plot(range(10), 'b') #row=1, col=1

    
if __name__ == '__main__':
    # one_dim_forward_benchmark()
    # fast_one_dim_forward_benchmark()
    # one_dim_inverse_benchmark()
    # fast_one_dim_inverse_benchmark()
    # two_dim_forward_benchmark()
    # fast_two_dim_forward_benchmark()
    # two_dim_inverse_benchmark()
    # fast_two_dim_inverse_benchmark()
    # compression_benchmark()
    # compression_ratio_benchmark()
    # iterative_vs_recursive_haar()
    plot_haar_vectors()
