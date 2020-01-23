import torch
import torch.nn as nn


class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict):
        super().__init__()

        self.models_dict = models_dict

    def forward(self, studies):

        """
        Args:
            inputs (list): a list of studies.
                Each study is a list of tuples. E.g:
                [('AP', torchimg), ('lateral', torchimg)]
        """
        batch_size = len(studies)
        logits = torch.zeros(batch_size)

        for i, study in enumerate(studies):

            image_logits = torch.zeros(study.size()[0])

            for j, (img_type, img) in enumerate(study):
                image_logits[i] = self.models_dict[img_type](img)

            logits[i] = torch.max(image_logits)

        return logits
