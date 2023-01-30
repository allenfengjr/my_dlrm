import zfpy
import numpy as np
my_array = np.arange(1, 20)
compressed_data = zfpy.compress_numpy(my_array)
decompressed_array = zfpy.decompress_numpy(compressed_data)

# confirm lossless compression/decompression
np.testing.assert_array_equal(my_array, decompressed_array)

def change_layout_method(data,layout):
    if layout == 'original':
        return data
    elif layout == "1D_sample":
        return data.flatten()
    elif layout == "1D_table":
        return data.transpose().flatten()
    else:
        exit("Please choose right layout")
        
class zfpCompressor:
    def compress(self,data,l_error_bound,layout):
        a_error_bound = l_error_bound*(data.max()-data.min())
        data_compressed = zfpy.compress_numpy(data, tolerance=a_error_bound)
        return data_compressed
    def decompress(self,data):
        return zfpy.decompress_numpy(data)

class EmbeddingCompressor:
    def __init__(self,compressor) -> None:
        self.compressor = compressor
    