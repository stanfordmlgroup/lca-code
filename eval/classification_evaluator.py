import os
import torch
import sklearn.metrics as sk_metrics
import numpy as np
import pandas as pd

from .average_meter import AverageMeter
from .loss import get_loss_fn
from .base_evaluator import BaseEvaluator
from tqdm import tqdm
from .metric.collate_accuracy import collate_accuracy_score

#TODO: Figure out how to handle imports from projects
from .below_curve_counter import BelowCurveCounter


NEG_INF = -1e9


class ClassificationEvaluator(BaseEvaluator):
    """Class for evaluating a model during training and testing."""
    def __init__(self, data_loaders, logger, num_visuals=None,
                 iters_per_eval=1, has_missing_tasks=False, model_uncertainty=False,
                 class_weights=None, max_eval=None, device=None, optimizer=None):
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
        super().__init__(data_loaders, logger, num_visuals, iters_per_eval, optimizer=optimizer)
        self.max_eval = None if max_eval is None or max_eval < 0 else max_eval
        self.uw_loss_fn = get_loss_fn('cross_entropy', device, model_uncertainty,
                                      has_missing_tasks, mask_uncertain=False, class_weights=class_weights)
        self.w_loss_fn = get_loss_fn('weighted_loss', device, model_uncertainty,
                                     has_missing_tasks, mask_uncertain=False, class_weights=class_weights)
        self.model_uncertainty = model_uncertainty
        self.device = device



    def _eval_split(self, model, data_loader, split, dataset_name, device,
            results_dir='', report_probabilities=False):
        """Evaluate a model for a single data split.
        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            split: Split being evaluated. One of 'train', 'val', or 'test'.
            dataset_name: Name of the dataset being evaluated on
            device: Device on which to evaluate the model.

        Returns:
            metrics, curves: Dictionaries with evaluation metrics and curves (as PNG images), respectively.
        """

        # Keep track of records needed for computing overall metrics
        records = {'keys': [], 'probs': [], 'slide_ids': [], 'loss_meter': AverageMeter(),
                   'w_loss_meter': AverageMeter()}
        subjectIDs = []

        num_examples = len(data_loader.dataset)
        if self.max_eval is not None:
            num_examples = min(num_examples, self.max_eval)

        # Sample from the data loader and record model outputs
        num_evaluated = 0
        with tqdm(total=num_examples, unit=' ' + split + ' ' + dataset_name) as progress_bar:
            for data in data_loader:
                if num_evaluated >= num_examples:
                    break

                with torch.no_grad():
                    # TODO (amit): remove the 3 and fix test set run.
                    inputs, targets, info_dict = data[:3]

                    logits = model.forward(inputs.to(device))

                    unweighted_loss = self.uw_loss_fn(logits, targets.to(self.device))

                    batch_probs = torch.sigmoid(logits)

                    if self.w_loss_fn is not None:
                        weighted_loss = self.w_loss_fn(logits, targets.to(self.device))
                        records['w_loss_meter'].update(weighted_loss.item(), logits.size(0))

                    # TODO (amit): redesign this logic.
                    if self.logger is not None and split is not 'train':
                        self.logger.log_iter(inputs, logits, targets, unweighted_loss,
                            weighted_loss, self.optimizer, phase=split)


                records['probs'].append(batch_probs)
                records['keys'].append(targets)
                records['loss_meter'].update(unweighted_loss.item(), logits.size(0))
                
                if report_probabilities and 'subjID' in info_dict:
                    subjectIDs.append(info_dict['subjID'])

                if info_dict is not None and 'slide_id' in info_dict.keys():
                    for id in info_dict['slide_id']:
                        records['slide_ids'].append(id)

                progress_bar.update(targets.size(0))
                num_evaluated += targets.size(0)

        # TODO (amit): generalize it to multi-task
        if report_probabilities:
            temp_probs = np.concatenate(records['probs'])
            final_subjecIDs = np.concatenate(subjectIDs)
            final_probs = [prob[0] for i, prob in enumerate(temp_probs)]
            probs_df = pd.DataFrame(data={'subjectID': final_subjecIDs, 'prob': final_probs})
        
            if results_dir:
                results_path = os.path.join(results_dir, f'probabilities.csv')
                probs_df.to_csv(results_path)

        # Map to summary dictionaries
        metrics, curves = self._get_summary_dicts(self, data_loader, split, dataset_name, **records)

        return metrics, curves

    @staticmethod
    def _get_summary_dicts(self, data_loader, split, dataset_name,
                           keys, probs, slide_ids, loss_meter, w_loss_meter):
        """Get summary dictionaries given dictionary of records kept during evaluation.
        Args:
            data_loader:
            split: Split being evaluated. One of 'train', 'val', or 'test'.
            dataset_name: The name of the dataset as a string.
                Either 'nih' or 'stanford'.
            device: The device to do the evaluation on
            keys: The groundtruth as a numpy array.
            probs: The predicted probabilities as a list of numpy arrays (batch_size, num_classes).
            slide_ids: The slide_ids for each patch specific for the use of whole slide predictions
            loss_meter: AverageMeter keeping track of average loss during evaluation.
            w_loss_meter: AverageMeter keeping track of average weighted loss during evaluation.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model.
        """
        metrics, curves = {}, {}

        probs = np.concatenate(probs)
        keys = np.concatenate(keys)
        data_subset = f'{dataset_name}-{split}'

        # Load the dataset classes, and the classes the model outputs values for.
        original_tasks = list(data_loader.dataset.original_tasks)
        tasks = list(data_loader.dataset.target_tasks)

        #competition_tasks = list(TASK_SEQUENCES['competition'])
        for i in range(0, keys.shape[1]):

            # Don't evaluate on missing categories
            #if tasks[i] not in original_tasks:
            #    continue

            task_keys = keys[:, i]
            task_probs = probs[:, i]
            task_name = tasks[i]


            task_keys, task_probs = np.array(task_keys), np.array(task_probs)
            task_preds = (task_probs > 0.5)

            '''
            if (self.rad_perf is not None
                    and split in ['valid', 'test']
                    and dataset_name == 'stanford'):
                eval_metrics += ['rads_below_ROC', 'rads_below_PR']
            '''

            for curve in ['PRC', 'ROC']:
                self._update_metric(curves, curve, task_name, task_keys,
                                    task_probs, task_preds, data_subset, slide_ids)

            eval_metrics = ['AUPRC', 'AUROC', 'log_loss', 'accuracy', 'precision', 'recall']

            #TODO: Replace this with command line arg?
            if len(slide_ids) > 0:
                eval_metrics.append('collate_accuracy')

            for metric in eval_metrics:
                self._update_metric(metrics, metric, task_name, task_keys,
                                    task_probs, task_preds, data_subset, slide_ids=slide_ids)


        metrics.update({
            data_subset + '_' + 'loss': loss_meter.avg,
            data_subset + '_' + 'weighted_loss': w_loss_meter.avg
        })


        return metrics, curves

    def _update_metric(self, dict_to_update, eval_metric, task_name,
                       task_keys, task_probs, task_preds, data_subset, slide_ids=None):
        fn_dict_probs, fn_dict_binary = self._get_eval_functions(task_name)

        if eval_metric in fn_dict_probs:
            inputs = task_probs
            eval_fn = fn_dict_probs[eval_metric]
        elif eval_metric in fn_dict_binary:
            inputs = task_preds
            eval_fn = fn_dict_binary[eval_metric]

        try:
            if eval_metric == 'collate_accuracy':
                dict_to_update.update({
                    data_subset + '_' + task_name + eval_metric: eval_fn(task_keys, inputs, slide_ids)
                })
            else:
                dict_to_update.update({
                    data_subset + '_' + task_name + eval_metric: eval_fn(task_keys, inputs)
                })
        except Exception as e:
            print(f'Exception for Pathology: {task_name}, {eval_metric}')
            print(e)


    def _get_eval_functions(self, task_name):
        # Functions that take probs as input
        fn_dict_probs = {
            'AUPRC': sk_metrics.average_precision_score,
            'AUROC': sk_metrics.roc_auc_score,
            'log_loss': sk_metrics.log_loss,
            'PRC': sk_metrics.precision_recall_curve,
            'ROC': sk_metrics.roc_curve,
        }

        # Rads below curve counter (PR and ROC)
        '''
        if self.rad_perf is not None:
            below_curve_counter = BelowCurveCounter(self.rad_perf, task_name)
            fn_dict_probs.update({
                'rads_below_ROC': below_curve_counter.ROC,
                'rads_below_PR': below_curve_counter.PR
            })
        '''
        '''
        if self.calibrate:
            fn_dict_probs.update({
                'CC_Brier': self._calibration_curve_and_brier(Path(self.results_dir+"-predict") / "valid", task_name)
            })
        '''

        # Functions that take binary as input
        fn_dict_binary = {
            'accuracy': sk_metrics.accuracy_score,
            'precision': sk_metrics.precision_score,
            'recall': sk_metrics.recall_score,
            'collate_accuracy': collate_accuracy_score,
        }

        return fn_dict_probs, fn_dict_binary

    def _calibration_curve_and_brier(self, results_dir, task_name):

        calibrator_dir = results_dir / "calibration_models"

        def compute_curve_and_brier(y_true, y_prob):
            uncalibrated_prob_true, uncalibrated_prob_pred = sk_calibration.calibration_curve(y_true, y_prob)
            uncalibrated_brier_loss = sk_metrics.brier_score_loss(y_true, y_prob)

            isotonic_calibrator = Calibrator("isotonic", calibrator_dir, task_name, eval=True)
            isotonic_y_prob = isotonic_calibrator.predict(y_prob)
            isotonic_prob_true, isotonic_prob_pred = sk_calibration.calibration_curve(y_true, isotonic_y_prob)
            isotonic_brier_loss = sk_metrics.brier_score_loss(y_true, isotonic_y_prob)

            platt_calibrator = Calibrator("platt", calibrator_dir, task_name, eval=True)
            platt_y_prob = platt_calibrator.predict(y_prob)
            platt_prob_true, platt_prob_pred = sk_calibration.calibration_curve(y_true, platt_y_prob)
            platt_brier_loss = sk_metrics.brier_score_loss(y_true, platt_y_prob)

            return uncalibrated_prob_true, uncalibrated_prob_pred, uncalibrated_brier_loss,\
                   isotonic_prob_true, isotonic_prob_pred, isotonic_brier_loss,\
                   platt_prob_true, platt_prob_pred, platt_brier_loss

        return compute_curve_and_brier