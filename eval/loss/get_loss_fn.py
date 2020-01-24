import torch
import torch.nn as nn
from torch.nn import functional as F

from .cross_entropy_with_uncertainty import CrossEntropyLossWithUncertainty
from .focal_loss import FocalLoss
from dataset import LabelMapper


def get_loss_fn(loss_fn_name,
                device,
                model_uncertainty,
                has_missing_tasks,
                mask_uncertain=True,
                class_weights=None):

        """Returns the loss function

            Args:
                loss_fn_name: Name of the loss function. Alternatives: cross_entropy, weighted_loss.
                device: Device for loss-related tensors.
                model_uncertainty: If true, uncertainty is explicitly modeled in the outputs.
                has_missing_tasks: Should be true if there is a possibility that labels for some classes for some examples will be missing.
                mask_uncertain: a bool determined whether or not to skip the loss for uncertain. NOTE: Must be set to true currently.
                class_weights: a list [negative_weights, positive_weights]

        """
        if model_uncertainty:
            if loss_fn_name == 'weighted_loss':
                return None
            loss_fn = CrossEntropyLossWithUncertainty()
        else:
            apply_masking = has_missing_tasks or mask_uncertain

            # Weighted or unweighted
            # Only reduce, if we're not gonna mask
            if loss_fn_name == 'weighted_loss':
                loss_fn = WeightedBCEWithLogitsLoss(class_weights, reduce=not apply_masking)
            elif loss_fn_name == 'focal_loss':
                loss_fn = FocalLoss()
            else:
                loss_fn = nn.BCEWithLogitsLoss(reduce=not apply_masking)

            # Apply a wrapper that masks missing labels
            # and uncertain labels.
            if apply_masking:
                loss_fn = MaskedLossWrapper(loss_fn, mask_uncertain, has_missing_tasks, device)

        return loss_fn


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, class_weights, reduce):
        """Returns a weighted binary cross entropy loss.

            Args:
                class_weights: a list of two numpy arrays


        """

        super().__init__()
        assert class_weights is not None

        self.reduce = reduce
        self.n_weights = class_weights[0]
        self.p_weights = class_weights[1]

    def _get_weights(self, targets, n_weights, p_weights):
        p_weights = torch.cuda.FloatTensor(p_weights)
        n_weights = torch.cuda.FloatTensor(n_weights)

        weights = ((targets == 1).float() * p_weights
                + (targets == 0).float() * n_weights)

        return weights

    def forward(self, logits, targets):

        weights = self._get_weights(targets,
                                    self.n_weights,
                                    self.p_weights)
        loss = F.binary_cross_entropy_with_logits(logits,
                                                  targets,
                                                  weights,
                                                  reduce=self.reduce)
        return loss

class MaskedLossWrapper(nn.Module):

    def __init__(self,
                loss_fn,
                mask_uncertain,
                has_missing_tasks,
                device,
                weights=None):

        super().__init__()
        self.loss_fn = loss_fn
        self.has_missing_tasks = has_missing_tasks
        self.mask_uncertain = mask_uncertain
        self.device = device


    def _get_mask(self, targets):
        """Returns a mask to mask uncertain
        and missing labels.

        Functions tales advantage of the following:
            Negative/Positive: 0/1
            Uncertain: -1
            Missing: -2        """

        mask = torch.ones(targets.shape)
        if self.mask_uncertain:
            mask[targets == LabelMapper.UNCERTAIN] = 0

        if self.has_missing_tasks:
            mask[targets == LabelMapper.MISSING] = 0

        mask = mask.to(self.device)

        return mask

    def forward(self, logits, targets):

        # Apply loss function
        loss = self.loss_fn(logits, targets)

        # Apply mask to skip missing labels
        # and handle uncertain labels
        if self.mask_uncertain or self.has_missing_tasks:
            mask = self._get_mask(targets)
            loss = loss * mask

        # Average the loss
        loss = loss.sum()
        loss = loss * (1 / (mask.sum()))

        return loss
