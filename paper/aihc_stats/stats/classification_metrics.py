from .point_metrics import *
from .summary_metrics import *


# Classification metric helper function
def get_classification_metrics(eval_groundtruth,
                               eval_probabilities,
                               tune_threshold=False,
                               tune_groundtruth=None,
                               tune_probabilities=None,
                               operating_point_value=0.95,
                               ci=True,
                               logger=None):
    summary_metrics = compute_summary_metrics(eval_groundtruth,
                                              eval_probabilities,
                                              ci=ci,
                                              logger=logger)
    point_metrics = compute_point_metrics(eval_groundtruth,
                                          eval_probabilities,
                                          tune_threshold,
                                          tune_groundtruth,
                                          tune_probabilities,
                                          operating_point_value,
                                          ci=ci,
                                          logger=logger)

    return summary_metrics, point_metrics


def get_experiment_roc_curves(experiment2probabilities, eval_groundtruth):
    """Aggregate experiment trials and return one ROC curve per experiment."""
    proc = importr('pROC')
    experiment_curves = {}
    for experiment, probabilities in experiment2probabilities.items():

        experiment_curve = proc.roc(eval_groundtruth, probabilities)
        experiment_curves[experiment] = experiment_curve

    return experiment_curves
