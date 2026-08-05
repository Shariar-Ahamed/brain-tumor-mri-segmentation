import os
from tqdm import tqdm
import numpy as np

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except ImportError:
    h5py = None  # type: ignore
    HAS_H5PY = False

def create_hdf5_cache(dataset_dict, h5_filepath, compression="gzip"):
    """
    Creates an HDF5 cache file containing preprocessed slices for fast loading in Colab Pro.
    dataset_dict: dict mapping slice_id -> {'image': np.array (4, H, W), 'mask': np.array (H, W)}
    """
    if not HAS_H5PY:
        raise ImportError("h5py library is not installed. Please install it using 'pip install h5py'.")
    os.makedirs(os.path.dirname(h5_filepath), exist_ok=True)
    
    with h5py.File(h5_filepath, "w") as h5f:
        for slice_id, data in tqdm(dataset_dict.items(), desc="Caching to HDF5"):
            grp = h5f.create_group(slice_id)
            grp.create_dataset("image", data=data["image"], dtype="float32", compression=compression)
            grp.create_dataset("mask", data=data["mask"], dtype="int64", compression=compression)
            
    print(f"HDF5 Cache successfully saved to: {h5_filepath}")

def read_hdf5_slice(h5_filepath, slice_id):
    """
    Reads a single slice from an existing HDF5 cache file.
    """
    if not HAS_H5PY:
        raise ImportError("h5py library is not installed.")
    with h5py.File(h5_filepath, "r") as h5f:
        img = h5f[slice_id]["image"][:]
        mask = h5f[slice_id]["mask"][:]
    return img, mask
