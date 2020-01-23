import pandas as pd
import torch

from collections import OrderedDict
from dataset.constants import COL_PATH


class CSVReaderModel(object):
    """Class that looks like a PyTorch model, but actually reads from a CSV file."""

    def __init__(self, csv_path, task_sequence):
        self.task_sequence = task_sequence
        self.study2probs = {}  # Format: {study_path -> {pathology -> prob}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Add predictions to the master dict
            self.study2probs[row[COL_PATH]] = OrderedDict((t, row[t]) for t in task_sequence)

    def forward(self, study_paths):
        probs = [list(self.study2probs[path].values()) for path in study_paths]
        probs = torch.tensor(probs, dtype=torch.float32)

        return probs

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        return self

    @property
    def module(self):
        return self
