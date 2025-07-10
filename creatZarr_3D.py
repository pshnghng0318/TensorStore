import numpy as np
import zarr
import shutil
import numcodecs
import os

zarr_dir = "as_128_3D.zarr"
if os.path.exists(zarr_dir):
    shutil.rmtree(zarr_dir)

shape = (1, 128, 7763, 4742)
chunk = (1, 64, 256, 256)
compressor = numcodecs.Blosc(cname='lz4', clevel=1)

store = zarr.DirectoryStore(zarr_dir)
root = zarr.group(store=store, overwrite=True)

zarr_array = root.create_dataset(
    name='data',
    shape=shape,
    chunks=chunk,
    dtype='<f4',
    #compressor=compressor,
    compressor=None,
    fill_value=0,
    overwrite=True
)

print("Dataset shape:", zarr_array.shape)
print("Dataset chunks:", zarr_array.chunks)


for ch in range(0, 128, 64):
    print(ch)
    for y in range(0, 7763, 256):
        for x in range(0, 4742, 256):
            dch = min(64, 128 - ch)
            dy = min(256, 7763 - y)
            dx = min(256, 4742 - x)
            chunk_data = np.random.rand(1, dch, dy, dx).astype(np.float32)
            zarr_array[0, ch:ch+dch, y:y+dy, x:x+dx] = chunk_data.squeeze(0)
