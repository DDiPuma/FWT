import haar
import imageio
import numpy as np


class HaarImageCompressor:
    uncompressed_image = None
    wavelet_coefficients = None
    compression_method = None
    forward_transform_dict = {"matrix": "matrix_2d_haar_transform", "inplace": "inplace_fast_2d_haar_transform",
                              "ordered": "ordered_fast_2d_haar_transform"}
    inverse_transform_dict = {"matrix": "matrix_inverse_2d_haar_transform", "inplace": "inplace_inverse_fast_2d_haar_transform",
                              "ordered": "ordered_inverse_fast_2d_haar_transform"}
    compressed_image = None
    target_compression_ratio = None
    actual_compression_ratio = 0

    def __init__(self, compression_method="matrix", target_compression_ratio=0):
        self.select_compression_method(compression_method)
        self.select_target_compression_ratio(target_compression_ratio)

    def load_image(self, file_path):
        self.uncompressed_image = imageio.imread(file_path)

    def select_compression_method(self, function_string):
        """
        :param function_string: function string should be string in ["matrix", "inplace", "ordered"]
        """
        if function_string in ["matrix", "inplace", "ordered"]:
            self.compression_method = function_string

    def select_target_compression_ratio(self, compression_ratio):
        self.target_compression_ratio = compression_ratio

    def compress_image_bad(self):
        self.wavelet_coefficients = getattr(haar, self.forward_transform_dict[self.compression_method])(self.uncompressed_image)
        # Intended to count the nonzero elements in the wavelet coefficients
        uncompressed_nonzero = np.count_nonzero(abs(self.wavelet_coefficients) != 0)
        error_tolerance = .1*self.target_compression_ratio
        threshold = np.percentile(self.wavelet_coefficients, 100/self.target_compression_ratio if self.target_compression_ratio != 0 else 1)

        if self.target_compression_ratio == 0:
            self.actual_compression_ratio = 0
            self.compressed_image = getattr(haar, self.inverse_transform_dict[self.compression_method])(self.wavelet_coefficients)
            return
        else:
            while True:
                # Intended to count the elements in the wavelet coefficients above the zeroing threshold
                compressed_nonzero = np.count_nonzero(abs(self.wavelet_coefficients) >= abs(threshold)) + 1
                temp_compression_ratio = uncompressed_nonzero/compressed_nonzero

                # Check if we are within an allowed error tolerance
                if abs(self.target_compression_ratio - temp_compression_ratio) < error_tolerance:
                    self.actual_compression_ratio = temp_compression_ratio

                    self.wavelet_coefficients = getattr(haar, self.forward_transform_dict[self.compression_method])(
                        self.uncompressed_image)

                    self.wavelet_coefficients[abs(self.wavelet_coefficients) < abs(threshold)] = 0
                    self.compressed_image = getattr(haar, self.inverse_transform_dict[self.compression_method])(self.wavelet_coefficients)
                    break
                elif self.target_compression_ratio > temp_compression_ratio:
                    threshold = threshold - .05*self.target_compression_ratio
                elif self.target_compression_ratio < temp_compression_ratio:
                    threshold = threshold + .05*self.target_compression_ratio

    def compress_image(self):
        self.wavelet_coefficients = getattr(haar, self.forward_transform_dict[self.compression_method])(
            self.uncompressed_image)

        if self.target_compression_ratio == 0:
            self.actual_compression_ratio = 0
            self.compressed_image = getattr(haar, self.inverse_transform_dict[self.compression_method])(
                self.wavelet_coefficients)
            return

        uncompressed_nonzero = np.count_nonzero(abs(self.wavelet_coefficients) != 0)
        target_nonzero = uncompressed_nonzero//self.target_compression_ratio
        threshold = np.partition(self.wavelet_coefficients[:], int(target_nonzero), None)[target_nonzero]
        compressed_nonzero = np.count_nonzero(self.wavelet_coefficients >= abs(threshold))

        self.actual_compression_ratio = uncompressed_nonzero/compressed_nonzero
        self.wavelet_coefficients[abs(self.wavelet_coefficients) < abs(threshold)] = 0
        self.compressed_image = getattr(haar, self.inverse_transform_dict[self.compression_method])(
            self.wavelet_coefficients)


def compressor_test():
    import matplotlib.pyplot as plt

    compressor = HaarImageCompressor(compression_method="matrix", target_compression_ratio=0)
    compressor.load_image("cam.png")

    plt.imshow(compressor.uncompressed_image, cmap='gray')
    plt.title("Uncompressed")
    plt.show()
    # plt.savefig("{}.png".format(0))

    for cr in [2, 4, 8, 15, 25, 50, 100, 250, 500, 1000]:
        compressor.select_target_compression_ratio(cr)
        compressor.compress_image()
        plt.imshow(compressor.compressed_image, cmap='gray')
        plt.title("Compressed, compression ratio of {:.2f}".format(compressor.actual_compression_ratio))
        plt.show()
        # plt.savefig("{}.png".format(cr))


if __name__ == '__main__':
    compressor_test()
