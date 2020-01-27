import numpy as np
import sklearn.metrics as skm
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2 import robjects
from rpy2.robjects.packages import importr
from sklearn.utils import check_consistent_length, check_array
from collections import defaultdict

from .bootstrap import Bootstrapper


def concordance_index_censored(event_indicator, event_time, estimate):
    """Concordance index for right-censored data
    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.
    Samples are comparable if for at least one of them an event occurred.
    If the estimated risk is larger for the sample with a higher time of
    event/censoring, the predictions of that pair are said to be concordant.
    If an event occurred for one sample and the other is known to be
    event-free at least until the time of event of the first, the second
    sample is assumed to *outlive* the first.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added
    to the count of concordant pairs.
    A pair is not comparable if an event occurred for both of them at the same
    time or an event occurred for one of them but the time of censoring is
    smaller than the time of event of the first one.
    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred
    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event
    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of pairs having an event at the same time
    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = check_array(estimate, ensure_2d=False)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    n_samples = len(event_time)
    if n_samples < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    order = np.argsort(event_time)

    tied_time = 0
    comparable = {}
    for i in range(n_samples - 1):
        inext = i + 1
        j = inext
        time_i = event_time[order[i]]
        while j < n_samples and event_time[order[j]] == time_i:
            j += 1

        if event_indicator[order[i]]:
            mask = np.zeros(n_samples, dtype=bool)
            mask[inext:] = True
            if j - i > 1:
                # event times are tied, need to check for coinciding events
                event_at_same_time = event_indicator[order[inext:j]]
                mask[inext:j] = np.logical_not(event_at_same_time)
                tied_time += event_at_same_time.sum()
            comparable[i] = mask
        elif j - i > 1:
            # events at same time are comparable if at least one of them is positive
            mask = np.zeros(n_samples, dtype=bool)
            mask[inext:j] = event_indicator[order[inext:j]]
            comparable[i] = mask

    concordant = 0
    discordant = 0
    tied_risk = 0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        if event_i:
            # an event should have a higher score
            con = (est < est_i).sum()
        else:
            # a non-event should have a lower score
            con = (est > est_i).sum()
        concordant += con

        tie = (est == est_i).sum()
        tied_risk += tie

        discordant += est.size - con - tie

    cindex = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    return cindex, concordant, discordant, tied_risk, tied_time


# Regression metric helper function
def get_regression_metrics(groundtruth, predictions, num_predictors=None, ci=None,
                           replicates=5000, seed=42, logger=None):
    """Returns regression metrics from groundtruth and predictions"""
    metrics = defaultdict(list)
    fn_dict = get_regression_fn()

    groundtruth = np.array(groundtruth)
    predictions = np.array(predictions)
    
    if ci == "bootstrap":

        N = groundtruth.shape[0]

        bootstrapper = Bootstrapper(N=N, replicates=replicates, seed=seed)

        for replicate_inds in bootstrapper:
            for metric in fn_dict.keys():
                groundtruth_replicate = groundtruth[replicate_inds]
                predictions_replicate = predictions[replicate_inds]
                metrics[metric].append(fn_dict[metric](groundtruth_replicate,
                                                       predictions_replicate,
                                                       ci=False))

        for metric in fn_dict.keys():
            
            sample_metric = metrics[metric]

            mean = np.mean(sample_metric)
            lower = 2 * mean - np.quantile(sample_metric, 0.975)
            upper = 2 * mean - np.quantile(sample_metric, 0.025)

            metrics[metric] = {'lower': lower, 'mean': mean, 'upper': upper}

    elif ci == "other":
        for metric in fn_dict.keys():
            metrics[metric] = fn_dict[metric](groundtruth, predictions, num_predictors=num_predictors, ci=True)

    elif ci is None:
        for metric in fn_dict.keys():
            metrics[metric] = fn_dict[metric](groundtruth, predictions, ci=False)

    else:
        raise ValueError(f"ci={ci} not supported.")

    if logger is not None:
        logger.log(f"Regression metrics: {metrics}")

    return metrics


def get_regression_fn():
    """Returns fn dict"""
    fn_dict = {
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'R2': r2_score,
        'C_index': c_index,
        'Predictive_ratio': predictive_ratio,
        'calibration': calibration
    }

    return fn_dict


def calibration(groundtruth, predictions, ci=False, **kwargs):
        
    r_predictions = robjects.FloatVector(predictions)
    r_groundtruth = robjects.FloatVector(groundtruth)

    robjects.r('''
        calibration <- function(groundtruth, predictions){
            m <- lm(groundtruth ~ predictions)
            intercept <- unname(coef(m)[1])
            slope <- unname(coef(m)[2])
            intercept_lower <- confint(m)[1,1]
            intercept_upper <- confint(m)[1,2]
            slope_lower <- confint(m)[2,1]
            slope_upper <- confint(m)[2,2]
            return(c(intercept, intercept_lower, intercept_upper,
                     slope, slope_lower, slope_upper))
        }
    ''')
    calibration_metrics = robjects.r['calibration'](r_groundtruth, r_predictions)

    intercept = calibration_metrics[0]
    intercept_lower = calibration_metrics[1]
    intercept_upper = calibration_metrics[2]
    slope = calibration_metrics[3]
    slope_lower = calibration_metrics[4]
    slope_upper = calibration_metrics[5]
    
    if ci:
        return ({'lower': intercept_lower, 'mean': intercept, 'upper': intercept_upper},
                {'lower': slope_lower, 'mean': slope, 'upper': slope_upper})
    
    else:
        return (intercept, slope)


def predictive_ratio(groundtruth, predictions, ci=False, **kwargs):
    """
    :param groundtruth: True label
    :param predictions: Prediction from model
    :param ci:
    :return: average predicted cost / average true cost
    """
    if ci:
        
        N = groundtruth.shape[0]

        bootstrapper = Bootstrapper(N=N, replicates=5000)
        
        scores = []

        for replicate_inds in bootstrapper:
            groundtruth_replicate = groundtruth[replicate_inds]
            predictions_replicate = predictions[replicate_inds]
            scores.append(predictive_ratio(groundtruth_replicate,
                                           predictions_replicate,
                                           ci=False))

        mean = np.mean(scores)
        lower = 2 * mean - np.quantile(scores, 0.975)
        upper = 2 * mean - np.quantile(scores, 0.025)

        return {'lower': lower, 'mean': mean, 'upper': upper}

    else:
        return predictions.mean() / groundtruth.mean()


def mean_squared_error(groundtruth, predictions, ci=False, **kwargs):

    if ci:
        stats = importr('stats')
        diff = np.square(predictions - groundtruth)
        mse = stats.t_test(robjects.FloatVector(np.array(diff)))
        return {"lower": mse[mse.names.index('conf.int')][0],
                "mean": mse[mse.names.index('estimate')][0],
                "upper": mse[mse.names.index('conf.int')][1]} 

    else:
        return skm.mean_squared_error(groundtruth, predictions)


def mean_absolute_error(groundtruth, predictions, ci=False, **kwargs):

    if ci:
        stats = importr('stats')
        diff = np.abs(predictions - groundtruth)
        mae = stats.t_test(robjects.FloatVector(np.array(diff)))
        return {"lower": mae[mae.names.index('conf.int')][0],
                "mean": mae[mae.names.index('estimate')][0],
                "upper": mae[mae.names.index('conf.int')][1]}

    else:
        return skm.mean_absolute_error(groundtruth, predictions)


def r2_score(groundtruth, predictions, num_predictors=None, ci=False, **kwargs):

    if ci:

        r_predictions = robjects.FloatVector(predictions)
        r_groundtruth = robjects.FloatVector(groundtruth)

        robjects.r('''
            library(psychometric)
            cohen_r2 <- function(groundtruth, predictions, k){
                
                n <- length(groundtruth)
                d <- data.frame(groundtruth, predictions)
                rsq <- summary(lm(groundtruth ~ predictions, d))$r.squared
                mat <- CI.Rsq(rsq, n, k, level=.95)
                mean <-mat$Rsq
                lower <- mat$LCL
                upper <- mat$UCL
                return(c(mean, lower, upper))
            }
        ''')

        bootstrap = robjects.r['cohen_r2'](r_groundtruth, r_predictions, num_predictors)
        
        mean = bootstrap[0]
        lower = bootstrap[1]
        upper = bootstrap[2]

        return {'lower': lower, 'mean': mean, 'upper': upper}

    else:
        return skm.r2_score(groundtruth, predictions)


def c_index(groundtruth, predictions, ci=False, **kwargs):
    if ci:
        # # R implementation
        # sc = importr('survcomp')

        # # TODO: Figure out right params to pass here
        # c_index = sc.concordance_index(robjects.IntVector(groundtruth),
        #                                robjects.IntVector(predictions))
        hmisc = importr('Hmisc')

        # See https://www.rdocumentation.org/packages/Hmisc/versions/4.2-0/topics/rcorr.cens
        cens_stats = hmisc.rcorr_cens(robjects.FloatVector(predictions),
                                      robjects.FloatVector(groundtruth))

        # C Index, Dxy, S.D., n, missing, uncensored, Relevant Pairs,
        # Concordant, and Uncertain

        # Compute 95% CI
        c_index = cens_stats[0]
        sd = cens_stats[2] / 2
        lower = c_index - 1.96*sd
        upper = c_index + 1.96*sd

        return {"lower": lower, "mean": c_index, "upper": upper}

    else:
        # Use sksurv implementation of c-index with event-indicator all ones
        return concordance_index_censored(np.ones(groundtruth.shape[0], dtype=bool),
                                          groundtruth,
                                          -predictions)
