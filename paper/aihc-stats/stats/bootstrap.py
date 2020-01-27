import numpy as np


class Bootstrapper():
    def __init__(self, N, replicates=5000, seed=42):
        np.random.seed(seed)

        self.N = N
        self.replicates = replicates

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.replicates:
            self.counter += 1
            return np.random.choice(self.N, size=self.N, replace=True)
        else:
            raise StopIteration
