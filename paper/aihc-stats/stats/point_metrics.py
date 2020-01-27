import numpy as np
import sklearn.metrics as skm
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri as pandas2ri


def get_ci_metric(groundtruth, predictions, metric,
                  methods="wilson", conf_level=0.95):
    binom = importr('binom')
    tn, fp, fn, tp = skm.confusion_matrix(groundtruth,
                                          predictions).ravel()
    if metric == "precision":
        value = binom.binom_confint(x=int(tp), n=int(tp+fp),
                                    methods=methods, conf_level=conf_level)
    elif metric == 'recall':
        value = binom.binom_confint(x=int(tp), n=int(tp+fn),
                                    methods=methods, conf_level=conf_level)
    elif metric == 'specificity':
        value = binom.binom_confint(x=int(tn), n=int(tn+fp),
                                    methods=methods, conf_level=conf_level)
    elif metric == 'npv':
        value = binom.binom_confint(x=int(tn), n=int(tn+fn),
                                    methods=methods, conf_level=conf_level)
    elif metric == 'accuracy':
        value = binom.binom_confint(x=int(tp+tn), n=groundtruth.shape[0],
                                    methods=methods, conf_level=conf_level)
    else:
        raise ValueError(f"ci for metric {metric} not supported.")
    
    ci_list = list(value)

    mean = ci_list[3][0]
    lower = ci_list[4][0]
    upper = ci_list[5][0]
    
    ci_dict = {"lower": lower,
               "mean": mean,
               "upper": upper}

    return ci_dict

def precision_score(groundtruth, predictions, ci=False):
    if ci:
        return get_ci_metric(groundtruth, predictions, "precision")
    else:
        return skm.precision_score(groundtruth,
                                   predictions)


def recall_score(groundtruth, predictions, ci=False):
    if ci:
        return get_ci_metric(groundtruth, predictions, "recall")
    else:
        return skm.recall_score(groundtruth,
                                predictions)


def specificity_score(groundtruth, predictions, ci=False):
    if ci:
        return get_ci_metric(groundtruth, predictions, "specificity")
    else:
        if "output_dict" in skm.classification_report.__code__.co_varnames:
            return skm.classification_report(groundtruth,
                                             predictions,
                                             output_dict=True)["0.0"]["recall"]
        else:
            tn, fp, fn, tp = skm.confusion_matrix(groundtruth,
                                                  predictions).ravel()
            return tn / (tn + fp)


def npv_score(groundtruth, predictions, ci=False):
    if ci:
        return get_ci_metric(groundtruth, predictions, "npv")
    else:
        if "output_dict" in skm.classification_report.__code__.co_varnames:
            return skm.classification_report(groundtruth,
                                             predictions,
                                             output_dict=True)["0.0"]["precision"]
        else:
            tn, fp, fn, tp = skm.confusion_matrix(groundtruth,
                                                  predictions).ravel()
            return tp / (tp + fp)


def accuracy_score(groundtruth, predictions, ci=False):
    if ci:
        return get_ci_metric(groundtruth, predictions, "accuracy")
    else:
        return skm.accuracy_score(groundtruth,
                                  predictions)

ppv_score = precision_score
sensitivity_score = recall_score
f1_score = skm.f1_score
kappa_score = skm.cohen_kappa_score


def get_simple_point_metrics(groundtruth, predictions, ci):
    point_metric_names = ["precision", "ppv", "npv", "recall", 
                          "sensitivity", "specificity", "accuracy"]

    point_metrics = {}
    global_funcs = globals()
    for point_metric_name in point_metric_names:
        point_metric_func = global_funcs[f"{point_metric_name}_score"]
        point_metrics[point_metric_name] = point_metric_func(groundtruth,
                                                             predictions,
                                                             ci=ci)

    return point_metrics


def get_optimal_youdensj(groundtruth, probabilities,
                         return_threshold=False):
    """Get threshold maximizing sensitivity + specificity."""
    fpr, tpr, threshold = skm.roc_curve(groundtruth, probabilities)

    youdens_j_values = tpr + 1 - fpr

    argmax_youdens_j = np.argmax(youdens_j_values)
    max_youdens_j = np.max(youdens_j_values)

    if return_threshold:
        return max_youdens_j, threshold[argmax_youdens_j]
    else:
        return max_youdens_j


def get_sensitivity_at_specificity(groundtruth, probabilities, value=0.95,
                                   return_threshold=False):
    """Get threshold at specificity = value."""
    fpr, tpr, threshold = skm.roc_curve(groundtruth, probabilities)

    specificity_values = 1 - fpr
    sensitivity_values = tpr

    index = np.argmin(np.absolute(specificity_values - value))
    sensitivity_at_specificity = sensitivity_values[index]

    if return_threshold:
        return sensitivity_at_specificity, threshold[index]
    else:
        return sensitivity_at_specificity


def get_specificity_at_sensitivity(groundtruth, probabilities, value=0.95,
                                   return_threshold=False):
    """Get threshold at specificity = value."""
    fpr, tpr, threshold = skm.roc_curve(groundtruth, probabilities)

    sensitivity_values = tpr
    specificity_values = 1 - fpr

    index = np.argmin(np.absolute(sensitivity_values - value))
    specificity_at_sensitivity = specificity_values[index]

    if return_threshold:
        return specificity_at_sensitivity, threshold[index]
    else:
        return specificity_at_sensitivity


def compute_point_metrics(eval_groundtruth,
                          eval_probabilities,
                          tune_threshold,
                          tune_groundtruth,
                          tune_probabilities,
                          operating_point_value,
                          ci=True,
                          logger=None):
    """Compute point metrics."""
    threshold = 0.5
    if tune_threshold:
        _, threshold = get_optimal_youdensj(tune_groundtruth,
                                            tune_probabilities,
                                            return_threshold=True)

    if logger is not None:
        logger.log(f"Using threshold={threshold}.")
    eval_predictions = (eval_probabilities > threshold).astype(int)

    point_metrics = get_simple_point_metrics(eval_groundtruth, eval_predictions, ci=ci)

    if tune_threshold:
        _, specificity_threshold = get_sensitivity_at_specificity(tune_groundtruth,
                                                                  tune_probabilities,
                                                                  value=operating_point_value,
                                                                  return_threshold=True)
        _, sensitivity_threshold = get_specificity_at_sensitivity(tune_groundtruth,
                                                                  tune_probabilities,
                                                                  value=operating_point_value,
                                                                  return_threshold=True)
        high_specificity_predictions = (eval_probabilities > specificity_threshold).astype(int)
        high_sensitivity_predictions = (eval_probabilities > sensitivity_threshold).astype(int)

        sensitivity_at_high_specificity = sensitivity_score(eval_groundtruth,
                                                            high_specificity_predictions,
                                                            ci=ci)
        specificity_at_high_sensitivity = specificity_score(eval_groundtruth,
                                                            high_sensitivity_predictions,
                                                            ci=ci)
        accuracy_at_high_specificity = accuracy_score(eval_groundtruth,
                                                      high_specificity_predictions,
                                                      ci=ci)
        accuracy_at_high_sensitivity = accuracy_score(eval_groundtruth,
                                                      high_sensitivity_predictions,
                                                      ci=ci)

        point_metrics[f"sensitivity@specificity={operating_point_value}"] = sensitivity_at_high_specificity
        point_metrics[f"specificity@sensitivity={operating_point_value}"] = specificity_at_high_sensitivity
        point_metrics[f"accuracy@specificity={operating_point_value}"] = accuracy_at_high_specificity
        point_metrics[f"accuracy@sensitivity={operating_point_value}"] = accuracy_at_high_sensitivity
        point_metrics[f"specificity={operating_point_value} threshold"] = specificity_threshold
        point_metrics[f"sensitivity={operating_point_value} threshold"] = sensitivity_threshold

    if logger is not None:
        logger.log(f"Point metrics: {point_metrics}")

    return point_metrics
