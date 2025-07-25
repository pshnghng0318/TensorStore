import numpy as np
from astropy.io import fits
import zarr
import os
import shutil
import numcodecs

fits_file = "/images/set_QA/alma_band1_orionkl_bandscan_combined.fits"
zarr_dir = "alma16G.zarr"
#fits_file = "/images/set_QA/HD163296_CO_2_1.fits"
#zarr_dir = "image.zarr"
#fits_file = "i17_1G.fits"
#zarr_dir = "i17_1G.zarr"

if os.path.exists(zarr_dir):
    shutil.rmtree(zarr_dir)

#max_GB = 0.25 # 256 MB
max_GB = 0.0625 # 64 MB
max_bytes = max_GB * 1024**3

with fits.open(fits_file, memmap=True) as hdul:

    #data = hdul[1].data # for 2D image
    data = hdul[0].data # for 3D image
    shape = data.shape
    dtype = data.dtype
    bytes_per_element = data.dtype.itemsize
    freq_step = int(max(1, max_bytes // (shape[0] * shape[2] * shape[3] * bytes_per_element)))
    print(shape, dtype, bytes_per_element, freq_step)
    compressor = numcodecs.Blosc(cname='lz4', clevel=0)
    zarr_array = zarr.open(
        zarr_dir,
        mode='w',
        shape=shape,
        chunks=(shape[0], freq_step, shape[2], shape[3]),
        dtype=dtype,
        zarr_format=2,
        compressor=compressor
    )

    total_freq = shape[1]
    for start in range(0, total_freq, freq_step):
        end = min(start + freq_step, total_freq)
        print(start, end)

        chunk = data[:, start:end, :, :]
        zarr_array[:, start:end, :, :] = chunk

