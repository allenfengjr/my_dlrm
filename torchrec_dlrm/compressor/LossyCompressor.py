import zfpy
import numpy as np
my_array = np.arange(1, 20)
compressed_data = zfpy.compress_numpy(my_array)
decompressed_array = zfpy.decompress_numpy(compressed_data)

# confirm lossless compression/decompression
np.testing.assert_array_equal(my_array, decompressed_array)

def change_layout_method(data,layout):
    if layout == 'original':
    elif layout == "1D_sample":
    elif layout == "1D_table":
class zfpCompressor:
    def compress(self,data,l_error_bound,layout):

class EmbeddingCompressor:
    def __init__(self,compressor) -> None:
        self.compressor = compressor