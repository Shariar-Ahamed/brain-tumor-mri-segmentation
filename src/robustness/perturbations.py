import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    HAS_SCIPY = True
except ImportError:
    gaussian_filter = None  # type: ignore
    HAS_SCIPY = False

def apply_gaussian_noise(image, noise_level=0.1):
    """
    Applies Gaussian noise to multi-channel input image (C, H, W).
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_img = image + noise
    return noisy_img

def apply_motion_blur(image, kernel_size=5):
    """
    Applies motion blur to multi-channel input image (C, H, W).
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size, dtype=np.float32)
    kernel = kernel / kernel_size

    C, H, W = image.shape
    blurred = np.zeros_like(image)
    if HAS_CV2 and cv2 is not None:
        for c in range(C):
            blurred[c] = cv2.filter2D(image[c], -1, kernel)
    else:
        # PyTorch conv2d fallback
        kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        img_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        pad = kernel_size // 2
        blurred_t = F.conv2d(img_t, kernel_t, padding=pad, groups=C)
        blurred = blurred_t.squeeze(0).numpy().astype(image.dtype)
    return blurred

def apply_brightness_shift(image, factor=1.2):
    """
    Adjusts brightness by multiplying by factor.
    """
    return image * factor

def apply_low_resolution(image, downscale_factor=2):
    """
    Simulates low-resolution scan by downsampling and upsampling back.
    """
    C, H, W = image.shape
    low_res_h, low_res_w = H // downscale_factor, W // downscale_factor
    
    degraded = np.zeros_like(image)
    if HAS_CV2 and cv2 is not None:
        for c in range(C):
            downsampled = cv2.resize(image[c], (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)
            degraded[c] = cv2.resize(downsampled, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # (1, C, H, W)
        down_t = F.interpolate(img_t, size=(low_res_h, low_res_w), mode='bilinear', align_corners=False)
        up_t = F.interpolate(down_t, size=(H, W), mode='bilinear', align_corners=False)
        degraded = up_t.squeeze(0).numpy().astype(image.dtype)
        
    return degraded
