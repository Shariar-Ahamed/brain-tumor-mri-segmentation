import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
REVERSE_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 4}

def normalize_intensity(image):
    """
    Z-score normalization per channel for non-zero voxels.
    image shape: (C, H, W) or (C, H, W, D)
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        channel = image[c]
        non_zero_mask = channel > 0
        if np.any(non_zero_mask):
            mean = np.mean(channel[non_zero_mask])
            std = np.std(channel[non_zero_mask])
            if std > 0:
                normalized[c] = np.where(non_zero_mask, (channel - mean) / std, 0.0)
            else:
                normalized[c] = channel - mean
        else:
            normalized[c] = channel
    return normalized

def remap_brats_labels(mask):
    """
    Remaps BraTS segmentation labels (0, 1, 2, 4) into contiguous values (0, 1, 2, 3).
    0: Background
    1: NCR/NET (Necrotic and Non-Enhancing Tumor Core)
    2: ED (Peritumoral Edema)
    3: ET (Enhancing Tumor)
    """
    remapped = np.zeros_like(mask, dtype=np.int64)
    for orig_label, new_label in LABEL_MAP.items():
        remapped[mask == orig_label] = new_label
    return remapped

def restore_brats_labels(mask):
    """
    Restores contiguous labels (0, 1, 2, 3) back to BraTS original labels (0, 1, 2, 4).
    """
    restored = np.zeros_like(mask, dtype=np.int64)
    for new_label, orig_label in REVERSE_LABEL_MAP.items():
        restored[mask == new_label] = orig_label
    return restored

def resize_slice(image_slice, mask_slice, target_size=(192, 192)):
    """
    Resizes 2D image (C, H, W) and mask (H, W) to target_size.
    """
    if HAS_CV2 and cv2 is not None:
        C, H, W = image_slice.shape
        resized_img = np.zeros((C, target_size[0], target_size[1]), dtype=image_slice.dtype)
        for c in range(C):
            resized_img[c] = cv2.resize(image_slice[c], target_size, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        return resized_img, resized_mask
    else:
        # PyTorch fallback
        img_t = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0) # (1, C, H, W)
        mask_t = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        resized_img_t = F.interpolate(img_t, size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        resized_mask_t = F.interpolate(mask_t, size=target_size, mode='nearest').squeeze(0).squeeze(0)

        return resized_img_t.numpy().astype(image_slice.dtype), resized_mask_t.numpy().astype(mask_slice.dtype)

def patient_level_split(patient_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Performs leakage-free patient-level splitting.
    """
    np.random.seed(seed)
    shuffled_ids = np.array(sorted(patient_ids))
    np.random.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = shuffled_ids[:n_train].tolist()
    val_ids = shuffled_ids[n_train:n_train + n_val].tolist()
    test_ids = shuffled_ids[n_train + n_val:].tolist()

    return train_ids, val_ids, test_ids
