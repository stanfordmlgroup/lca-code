import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    # """Focal loss for binary classification.
    # Adapted from:
    #     https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
    #
    # Courtesy of Chris Chute.
    # """
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        inv_probs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (inv_probs * self.gamma).exp() * loss

        return loss.mean()


class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, input, target, p_weights, n_weights):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        probs = torch.sigmoid(input)
        mask = (target == 1).float() * torch.cuda.FloatTensor(p_weights) + (
                    target == 0).float() * torch.cuda.FloatTensor(n_weights)
        loss = F.binary_cross_entropy_with_logits(input, target, mask)
        # loss = - torch.mul(target, probs.log())
        # loss = torch.mul(loss, mask)

        return loss
