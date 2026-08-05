import pytest
import numpy as np
import torch

from src.utils.seed import seed_everything
from src.utils.metrics import compute_dice_score, compute_iou_score, compute_precision_recall_specificity, compute_hd95
from src.dataset.preprocessing import normalize_intensity, remap_brats_labels, restore_brats_labels, resize_slice, patient_level_split
from src.dataset.brats_dataset import BraTSDataset
from src.models import get_model, ProposedUnifiedHybridModel
from src.training.losses import CombinedDiceCELoss
from src.xai.grad_cam import GradCAMPlusPlus
from src.robustness.perturbations import apply_gaussian_noise, apply_motion_blur, apply_brightness_shift, apply_low_resolution

def test_seed_reproducibility():
    seed_everything(42)
    val1 = np.random.rand(1)[0]
    seed_everything(42)
    val2 = np.random.rand(1)[0]
    assert val1 == val2

def test_preprocessing():
    # Test intensity normalization
    raw_img = np.random.rand(4, 64, 64).astype(np.float32) * 100
    norm_img = normalize_intensity(raw_img)
    assert norm_img.shape == (4, 64, 64)
    assert not np.isnan(norm_img).any()

    # Test label remapping
    raw_mask = np.array([0, 1, 2, 4])
    remapped = remap_brats_labels(raw_mask)
    assert np.array_equal(remapped, np.array([0, 1, 2, 3]))

    restored = restore_brats_labels(remapped)
    assert np.array_equal(restored, raw_mask)

    # Test slice resizing
    resized_img, resized_mask = resize_slice(raw_img, raw_mask.reshape(2, 2), target_size=(32, 32))
    assert resized_img.shape == (4, 32, 32)
    assert resized_mask.shape == (32, 32)

    # Test patient splitting
    p_ids = [f"patient_{i:03d}" for i in range(100)]
    train, val, test = patient_level_split(p_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(set(train).intersection(set(val))) == 0

def test_models_forward_pass():
    models_to_test = ["Proposed-Hybrid", "ResNet34-UNet", "U-Net++", "DeepLabV3+"]
    dummy_input = torch.randn(2, 4, 64, 64)

    for m_name in models_to_test:
        model = get_model(model_name=m_name, in_channels=4, num_classes=4)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 4, 64, 64), f"Model {m_name} output shape mismatch: {output.shape}"

def test_loss_and_backward():
    model = ProposedUnifiedHybridModel(in_channels=4, num_classes=4)
    dummy_input = torch.randn(2, 4, 64, 64)
    dummy_target = torch.randint(0, 4, (2, 64, 64))

    criterion = CombinedDiceCELoss()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    assert loss.item() > 0.0
    loss.backward()
    
    # Verify gradients computed
    assert model.final_conv.weight.grad is not None

def test_metrics():
    pred_logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))

    dice = compute_dice_score(pred_logits, target)
    assert 'Mean_Tumor_Dice' in dice
    assert 0.0 <= dice['Mean_Tumor_Dice'] <= 1.0

    iou = compute_iou_score(pred_logits, target)
    assert 'Mean_Tumor_IoU' in iou

    pr_rec = compute_precision_recall_specificity(pred_logits, target)
    assert 'Class_1_Precision' in pr_rec

    hd95 = compute_hd95(target[0] == 1, target[0] == 1)
    assert hd95 == 0.0

def test_xai_grad_cam():
    model = ProposedUnifiedHybridModel(in_channels=4, num_classes=4)
    target_layer = model.final_conv
    grad_cam = GradCAMPlusPlus(model=model, target_layer=target_layer)

    dummy_input = torch.randn(1, 4, 64, 64)
    heatmap = grad_cam.generate_heatmap(dummy_input, target_class=3)
    
    assert heatmap.shape == (64, 64)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0

def test_robustness_perturbations():
    raw_img = np.random.rand(4, 64, 64).astype(np.float32)

    noisy = apply_gaussian_noise(raw_img, noise_level=0.1)
    assert noisy.shape == raw_img.shape

    blurred = apply_motion_blur(raw_img, kernel_size=5)
    assert blurred.shape == raw_img.shape

    bright = apply_brightness_shift(raw_img, factor=1.2)
    assert bright.shape == raw_img.shape

    low_res = apply_low_resolution(raw_img, downscale_factor=2)
    assert low_res.shape == raw_img.shape
