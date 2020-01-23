import torch.nn as nn


class HierarchyWrapper(nn.Module):

    def __init__(self, model, task_sequence):
        super().__init__()

        self.model = model
        self.task_sequence = task_sequence


    def forward(self, inputs):
        ls = self.task_sequence

        logits = self.model.forward(inputs)
        logits[:,ls['Edema']] += logits[:,ls['Airspace Opacity']]
        logits[:,ls['Atelectasis']] += logits[:,ls['Airspace Opacity']]
        logits[:,ls['Consolidation']] += logits[:,ls['Airspace Opacity']]
        logits[:,ls['Pneumonia']] += logits[:,ls['Airspace Opacity']]
        logits[:,ls['Lung Lesion']] += logits[:,ls['Airspace Opacity']]
        logits[:,ls['Cardiomegaly']] += logits[:,ls['Enlarged Cardiomediastinum']]


        return logits
