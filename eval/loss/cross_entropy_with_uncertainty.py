import torch
import torch.nn as nn


class CrossEntropyLossWithUncertainty(nn.Module):
    """Cross-entropy loss modified to also include uncertainty outputs."""
    def __init__(self, size_average=True, reduce=True):
        super(CrossEntropyLossWithUncertainty, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, logits, labels):
        """
        Args:
            logits: Un-normalized outputs of shape (batch_size, num_tasks, 3)
            labels: Labels of shape (batch_size, num_tasks) where -1 is uncertain, 0 is negative, 1 is positive.
        """
        batch_size, last_dim = logits.size()
        if last_dim % 3:
            raise ValueError('Last dim should be divisible by 3, got last dim of {}'.format(last_dim))
        num_tasks = last_dim // 3

        logits = logits.view(batch_size * num_tasks, 3)  # Fuse batch and task dimensions
        labels = (labels + 1).type(torch.int64)          # Shift labels into range [0, 2]
        labels = labels.view(-1)                         # Flatten

        loss = self.ce_loss(logits, labels)              # Output shape (batch_size * num_tasks,)
        loss = loss.view(batch_size, num_tasks)          # Reshape and take average over batch dim

        if self.size_average:
            loss = loss.mean(1)
        if self.reduce:
            loss = loss.mean(0)

        return loss
