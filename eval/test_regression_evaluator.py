import numpy as np

from regression_evaluator import RegressionEvaluator

class DummyDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset

    def __next__(self):
        self.dataset.next()

class DummyDataset:
    def __init__(self, nBatches, batch_shape):
        self.nBatches = nBatches
        self.batch_shape = batch_shape
        self.counter = 0
        self.split = 'test'
        self.dataset_name = 'Dummy'

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter > self.nBatches:
            raise StopIteration
        else:
            self.counter += 1
            return np.random.random_sample(self.batch_shape), np.random.random_sample(self.batch_shape[0])

    def __len__(self):
        return self.nBatches

class DummyModel:
    def __init__(self, nFeatures):
        self.nFeatures = nFeatures

    def predict(self, X):
        assert self.nFeatures == X.shape[1]
        return np.random.random_sample(X.shape[0])

    def train(self):
        return

    def eval(self):
        return

if __name__ == '__main__':
    
    dataset = DummyDataset(10, (10, 20))
    data_loader = DummyDataLoader(dataset)
    model = DummyModel(20)

    evaluator = RegressionEvaluator([data_loader], None, None)
    metrics, curves = evaluator.evaluate(model, None)
    print(metrics)
