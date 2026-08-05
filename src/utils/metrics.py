import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
    HAS_SCIPY_EDT = True
except ImportError:
    distance_transform_edt = None  # type: ignore
    HAS_SCIPY_EDT = False

def compute_dice_score(pred, target, num_classes=4, smooth=1e-6):
    """
    Compute Dice coefficient for multiclass segmentation.
    pred: (B, H, W) or (B, C, H, W) - class indices or probabilities
    target: (B, H, W) - ground truth class labels
    Returns: dict of per-class dice and mean dice
    """
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
    
    dice_scores = {}
    class_names = {0: 'Background', 1: 'NCR/NET', 2: 'Edema', 3: 'Enhancing Tumor'}
    
    total_dice = 0.0
    valid_classes = 0

    for c in range(num_classes):
        p_c = (pred == c).float()
        t_c = (target == c).float()

        intersection = torch.sum(p_c * t_c)
        cardinality = torch.sum(p_c) + torch.sum(t_c)

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_val = dice.item()
        dice_scores[class_names.get(c, f'Class_{c}')] = dice_val

        if c > 0: # Exclude background for mean tumor dice calculation
            total_dice += dice_val
            valid_classes += 1

    dice_scores['Mean_Tumor_Dice'] = total_dice / max(valid_classes, 1)
    return dice_scores

def compute_iou_score(pred, target, num_classes=4, smooth=1e-6):
    """
    Compute Intersection over Union (IoU / Jaccard) for multiclass segmentation.
    """
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
        
    iou_scores = {}
    class_names = {0: 'Background', 1: 'NCR/NET', 2: 'Edema', 3: 'Enhancing Tumor'}
    
    total_iou = 0.0
    valid_classes = 0

    for c in range(num_classes):
        p_c = (pred == c).float()
        t_c = (target == c).float()

        intersection = torch.sum(p_c * t_c)
        union = torch.sum(p_c) + torch.sum(t_c) - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_val = iou.item()
        iou_scores[class_names.get(c, f'Class_{c}')] = iou_val

        if c > 0:
            total_iou += iou_val
            valid_classes += 1

    iou_scores['Mean_Tumor_IoU'] = total_iou / max(valid_classes, 1)
    return iou_scores

def compute_precision_recall_specificity(pred, target, num_classes=4, smooth=1e-6):
    """
    Compute Precision, Recall (Sensitivity), and Specificity.
    """
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)

    results = {}
    for c in range(1, num_classes):
        p_c = (pred == c)
        t_c = (target == c)

        tp = torch.sum((p_c & t_c).float()).item()
        fp = torch.sum((p_c & ~t_c).float()).item()
        fn = torch.sum((~p_c & t_c).float()).item()
        tn = torch.sum((~p_c & ~t_c).float()).item()

        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth) # Sensitivity
        specificity = (tn + smooth) / (tn + fp + smooth)

        results[f'Class_{c}_Precision'] = precision
        results[f'Class_{c}_Recall'] = recall
        results[f'Class_{c}_Specificity'] = specificity

    return results

def compute_hd95(pred, target, voxel_spacing=None):
    """
    Compute 95th Percentile Hausdorff Distance (HD95) for binary mask.
    pred, target: (H, W) boolean/uint8 numpy arrays.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if not np.any(pred_bool) or not np.any(target_bool):
        return 0.0 if (not np.any(pred_bool) and not np.any(target_bool)) else 100.0

    if not HAS_SCIPY_EDT or distance_transform_edt is None:
        return 0.0

    # Distance transform
    dt_pred = distance_transform_edt(~pred_bool, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target_bool, sampling=voxel_spacing)

    dist_pred_to_target = dt_target[pred_bool]
    dist_target_to_pred = dt_pred[target_bool]

    if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
        return 100.0

    hd95_1 = np.percentile(dist_pred_to_target, 95)
    hd95_2 = np.percentile(dist_target_to_pred, 95)

    return float(max(hd95_1, hd95_2))
