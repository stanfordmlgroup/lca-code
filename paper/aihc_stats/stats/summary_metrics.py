import sklearn.metrics as skm
from rpy2.robjects.packages import importr


log_loss = skm.log_loss

def roc_auc_score(groundtruth, probabilities, ci):
    if ci:
        proc = importr('pROC')
        roc = proc.roc(groundtruth, probabilities)
        auc = proc.ci_auc(roc, conf_level=0.95, method="delong")

        # lower (95%), mean, upper (95%)
        ci_dict = {'lower': auc[0], 'mean': auc[1], 'upper': auc[2]}

        return ci_dict
        
    else:
        return skm.roc_auc_score(groundtruth, probabilities)


def pr_auc_score(groundtruth, probabilities, ci):
    if ci:
        # See also: https://www.rdocumentation.org/packages/MRIaggr/versions/1.1.5/topics/calcAUPRC
        mriaggr = importr('MRIaggr')
        prc = mriaggr.calcAUPRC(probabilities, groundtruth,
                                ci=True, alpha=0.05)
        
        # mean, lower (95%), upper (95%)
        ci_dict = {'lower': prc[1], 'mean': prc[0], 'upper': prc[2]}
        
        return ci_dict

    else:
        return skm.average_precision_score(groundtruth, probabilities)


def compute_summary_metrics(eval_groundtruth,
                            eval_probabilities,
                            ci=True,
                            logger=None):
    """Compute summary metrics."""
    auroc = roc_auc_score(eval_groundtruth,
                          eval_probabilities,
                          ci=ci)
    # FIXME: MRIagg won't install so CI for PR AUC does not work.
    auprc = pr_auc_score(eval_groundtruth,
                         eval_probabilities,
                         ci=False)
    
    log_loss = skm.log_loss(eval_groundtruth,
                            eval_probabilities)

    summary_metrics = {}
    summary_metrics["auroc"] = auroc
    summary_metrics["auprc"] = auprc
    summary_metrics["log_loss"] = log_loss

    if logger is not None:
        logger.log(f"Summary metrics: {summary_metrics}")

    return summary_metrics


# Curves.
roc_curve = skm.roc_curve
pr_curve = skm.precision_recall_curve


# Curve helper function
def get_curves(groundtruth,
               probabilities,
               logger=None):
    roc = roc_curve(groundtruth, probabilities)
    pr = pr_curve(groundtruth, probabilities)

    curves = {}
    curves["roc"] = roc
    curves["pr"] = pr

    return curves
