import sklearn.metrics as sk_metrics
import numpy as np
from tqdm import tqdm

from .average_meter import AverageMeter
from .base_evaluator import BaseEvaluator
from .sksurv.metrics import concordance_index_censored

class RegressionEvaluator(BaseEvaluator):
    """Class for evaluating regression models during train and test"""
    def __init__(self, data_loaders, logger, device=None):
        """
        Args: 
            data_loaders: DataLoader Class, iterable and returns (input target)
            logger: logs values
            device: Device on which to evaluate the model
        """
        #TODO: iter_per_eval is not so relevant, find a way to safely depricate"
        super().__init__(data_loaders, logger)
        self.device = device


    def _eval_split(self, model, data_loader, split, dataset_name, device):
        """Evaluate a model for a single data split
        Args:
            model: Model to evaluate
            data_loader: DataLoader to sample from
            split: Split to evaluate ("train", "val", or "test")
            dataset_name: Name of dataset to be evaluated
            device: Device on which to evaluate the model
        
        Returns:
            metrics, curves: Dictionaries with evaluation metrics and curves respectively
        """

        records = {'targets': [], 'preds': []}
        
        num_examples = len(data_loader.dataset)

       # sample from the data loader and record model outputs
        num_evaluated = 0
        with tqdm(total=num_examples, unit=' ' + split + ' ' + dataset_name) as progress_bar:
            for data in data_loader:
                if num_evaluated >= num_examples:
                    break

                inputs, targets = data

                batch_preds = model.predict(inputs)

                # record 
                records['preds'].append(batch_preds)
                records['targets'].append(targets)

                # expects list or numpy array
                progress_bar.update(len(targets))
                num_evaluated += len(targets)

        # Map to summary dictionaries
        metrics, curves = self._get_summary_dicts(self, split, dataset_name, **records)

        return metrics, curves


    @staticmethod
    def _get_summary_dicts(self, split, dataset_name,
                           preds, targets):
        """Get summary dictionaries given dictionary of records kept during evaluation
        Args:
            data_loader: DataLoader to sample from 
            split: Split being evaluated
            dataset_name: Name of dataset to be evaluated
            preds: The predicted values from model
            targets: The groundtruth as a numpy array
        
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model
        """
        # use all available metrics
        eval_metrics = self._get_eval_functions().keys()

        metrics, curves = {}, {}

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        data_subset = f'{dataset_name}-{split}'

        # assume single task

        for metric in eval_metrics:
           self._update_metric(metrics, metric, targets, preds, data_subset)

        return metrics, curves


    def _update_metric(self, dict_to_update, eval_metric, task_targets, task_preds, data_subset):
        """Calls eval fn and updates the metric dict with appropriate name and value"""
        fn_dict_preds = self._get_eval_functions()

        if eval_metric in fn_dict_preds:
            inputs = task_preds
            eval_fn = fn_dict_preds[eval_metric]

        try:
            dict_to_update.update({
                data_subset + '_' + eval_metric: eval_fn(task_targets, inputs)
            })
        except Exception as e:
            print(f'Could not update')
            print(e)

    @staticmethod
    def _get_eval_functions():
        """Return dictionary of eval functions"""

        def cindex(targets, preds):
            """Use sksurv implementation of c-index with event-indicator all ones"""
            return concordance_index_censored(np.ones(targets.shape[0], dtype=bool), targets, -preds)

        fn_dict_preds = {
            'MSE': sk_metrics.mean_squared_error,
            'MAE': sk_metrics.mean_absolute_error,
            'R2': sk_metrics.r2_score,
            'C-Index': cindex
        }

        return fn_dict_preds
