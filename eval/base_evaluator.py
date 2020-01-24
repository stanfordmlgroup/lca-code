import numpy as np

class BaseEvaluator(object):
    """Class for evaluating a model during training and testing."""

    def __init__(self, data_loaders, logger, num_visuals=None, iters_per_eval=1, optimizer=None):
        """
        Args:
            data_loaders: List of Torch `DataLoader`s to sample from.
            logger: Logger for plotting to console and TensorBoard.
            num_visuals: Number of visuals to display from the validation set.
            max_eval: Maximum number of examples to evaluate at each evaluation.
            iters_per_eval: Number of iterations between each evaluation.
            rad_perf: Dataframe of radiologist performance.
            save_path: Path to save CSV file.
        """
        self.data_loaders = data_loaders
        self.iters_per_eval = iters_per_eval
        self.logger = logger
        self.num_visuals = num_visuals
        self.optimizer = optimizer

    def evaluate(self, model, device, iter=None, results_dir='', report_probabilities=False):
        """Evaluate a model at the end of the given iteration.

        Args:
            model: Model to evaluate.
            device: Device on which to evaluate the model.
            iter: The iteration that just finished. Determines whether to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the current model.

        Notes:
            Returned dictionary will be empty if not an evaluation iteration.
        """
        metrics, curves = {}, {}

        if iter is None or iter % self.iters_per_eval == 0:
            # Evaluate on the training and validation sets
            for data_loader in self.data_loaders:
                split_metrics, split_curves = self._eval_split(model,
                                                               data_loader,
                                                               data_loader.dataset.split,
                                                               data_loader.dataset.dataset_name,
                                                               device, results_dir, report_probabilities)
                metrics.update(split_metrics)
                curves.update(split_curves)

        return metrics, curves

    def _eval_split(self, model, data_loader, split, dataset_name, device):
        raise NotImplementedError

    @staticmethod
    def _get_summary_dicts(**kwds):
        raise NotImplementedError

    def _update_metric(self, **kwds):
        raise NotImplementedError

    def _get_eval_functions(self, task_name):
        raise NotImplementedError
