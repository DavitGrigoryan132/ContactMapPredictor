import torch
import torch.nn as nn
import torch.nn.functional as F

      
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice loss for binary segmentation.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        
        intersection = (probs * targets).sum(dim=1)
        dice_score = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss.mean()


class CompoundLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=100.0):
        """
        Compound loss combining weighted BCEWithLogitsLoss and Dice loss.
        
        Args:
            bce_weight (float): Weight for BCE loss.
            dice_weight (float): Weight for Dice loss.
            pos_weight (float): Positive weight for BCE loss.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.dice = DiceLoss(smooth=1.0)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # targets: (B, 1, L, L) with values 0 or 1 (float)
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return loss

