import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        """
        :param gamma: Focusing parameter, usually between 2-5.
        :param weight: Class weights for imbalanced classes (optional).
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # For handling class imbalance

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # Apply gamma scaling
        return focal_loss.mean()  # Return mean loss
