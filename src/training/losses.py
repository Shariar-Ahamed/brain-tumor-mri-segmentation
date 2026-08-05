import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) long tensor
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice = 0.0
        for c in range(num_classes):
            p_c = probs[:, c]
            t_c = targets_one_hot[:, c]
            intersection = torch.sum(p_c * t_c)
            cardinality = torch.sum(p_c) + torch.sum(t_c)
            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice += (1.0 - dice_c)

        return dice / num_classes


class CombinedDiceCELoss(nn.Module):
    """
    Combined Dice Loss and Cross-Entropy Loss for brain tumor MRI segmentation.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce
