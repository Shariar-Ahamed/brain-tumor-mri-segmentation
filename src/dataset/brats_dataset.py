import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except ImportError:
    h5py = None  # type: ignore
    HAS_H5PY = False

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS Multimodal Brain Tumor MRI Slices.
    Supports 4 channels: T1, T1ce, T2, FLAIR.
    """
    def __init__(self, slice_data_dict=None, h5_filepath=None, slice_keys=None, transform=None):
        self.slice_data_dict = slice_data_dict
        self.h5_filepath = h5_filepath
        self.transform = transform

        if slice_keys is not None:
            self.slice_keys = slice_keys
        elif slice_data_dict is not None:
            self.slice_keys = list(slice_data_dict.keys())
        elif h5_filepath is not None and HAS_H5PY and h5py is not None:
            with h5py.File(h5_filepath, "r") as h5f:
                self.slice_keys = list(h5f.keys())
        else:
            self.slice_keys = []

    def __len__(self):
        return len(self.slice_keys)

    def __getitem__(self, idx):
        slice_key = self.slice_keys[idx]

        if self.slice_data_dict is not None and slice_key in self.slice_data_dict:
            item = self.slice_data_dict[slice_key]
            img = item['image'] # (4, H, W)
            mask = item['mask'] # (H, W)
        elif self.h5_filepath is not None and HAS_H5PY and h5py is not None:
            with h5py.File(self.h5_filepath, "r") as h5f:
                img = h5f[slice_key]["image"][:]
                mask = h5f[slice_key]["mask"][:]
        else:
            raise ValueError("No valid data source or h5py package available for BraTSDataset.")

        if self.transform is not None:
            # Albumentations expects (H, W, C)
            img_hwc = np.transpose(img, (1, 2, 0))
            augmented = self.transform(image=img_hwc, mask=mask)
            img = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']

        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return img_tensor, mask_tensor

