import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.packages import importr
from scipy import stats

from .bootstrap import Bootstrapper
from .regression_metrics import get_regression_fn


def test_auc_difference(roc1, roc2):
    proc = importr('pROC')

    list_vector = proc.roc_test(roc1, roc2, method="delong",
                                alternative="greater")

    p_value = np.array(list_vector.rx('p.value')).flatten()

    return p_value[0]


def test_bootstrap_difference(groundtruth, predictions1, predictions2,
                              metric="R2", replicates=5000, seed=42):
    regression_fn_dict = get_regression_fn()

    regression_fn =  regression_fn_dict[metric]

    groundtruth = np.array(groundtruth)
    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)

    N = groundtruth.shape[0]

    bootstrapper = Bootstrapper(N=N, replicates=replicates, seed=seed)

    differences = []
    for replicate_inds in bootstrapper:
        groundtruth_replicate = groundtruth[replicate_inds]
        predictions1_replicate = predictions1[replicate_inds]
        predictions2_replicate = predictions2[replicate_inds]

        predictions1_r2 = regression_fn(groundtruth_replicate,
                                        predictions1_replicate)
        predictions2_r2 = regression_fn(groundtruth_replicate,
                                        predictions2_replicate)

        differences_r2 = predictions2_r2 - predictions1_r2

        differences.append(differences_r2)

    mean = np.mean(differences)
    lower = 2 * mean - np.quantile(differences, 0.975)
    upper = 2 * mean - np.quantile(differences, 0.025)

    return {'lower': lower, 'mean': mean, 'upper': upper}


def test_r2_score_difference(groundtruth, predictions, predictions2):

    raise ValueError("test_r2_score_difference is deprecated. " +
                     "Use test_bootstrap_differences instead.")

    r_groundtruth = robjects.FloatVector(groundtruth)
    r_predictions = robjects.FloatVector(predictions)
    r_predictions2 = robjects.FloatVector(predictions2)

    robjects.r('''
        library(boot)
        bootstrapped_r2_diff <- function(groundtruth, predictions, predictions2){
            d <- data.frame(groundtruth, predictions, predictions2)
            b <- boot(d, function(data, indices)
                summary(lm(groundtruth ~ predictions2, data[indices,]))$r.squared -
                summary(lm(groundtruth ~ predictions, data[indices,]))$r.squared, R=10000)
            mean <- b$t0
            lower <- unname(quantile(b$t, 0.025))
            upper <- unname(quantile(b$t, 0.975))
            return(c(mean, lower, upper))
        }
    ''')

    bootstrap = robjects.r['bootstrapped_r2_diff'](r_groundtruth, r_predictions, r_predictions2)

    mean = bootstrap[0]
    lower = bootstrap[1]
    upper = bootstrap[2]

    return {'lower': lower, 'mean': mean, 'upper': upper}


def test_cindex_difference(groundtruth, predictions, predictions2):
    """
    Code adapted from 'survcomp' R package
    https://github.com/bhklab/survcomp/blob/master/R/cindex.comp.R
    """

    r_groundtruth = robjects.FloatVector(groundtruth)
    r_predictions = robjects.FloatVector(predictions)
    r_predictions2 = robjects.FloatVector(predictions2)

    robjects.r('''
        library(Hmisc)
        cindex_diff <- function(groundtruth, predictions, predictions2){
            c1 <- rcorr.cens(predictions, groundtruth)
            c2 <- rcorr.cens(predictions2, groundtruth)

            c1.cindex <- unname(c1[1])
            c2.cindex <- unname(c2[1])
            c1.sd <- unname(c1[3])
            c2.sd <- unname(c2[3])
            n <- unname(c1[4])

            c1.se <- c1.sd / sqrt(n)
            c2.se <- c2.sd / sqrt(n)

            r <- cor(predictions, predictions2, use = "complete.obs", method = "spearman")

            mean <- c2.cindex - c1.cindex
            se <- sqrt(c1.se^2 + c2.se^2 - 2 * r * c1.se * c2.se)
            t.stat <- mean / se
            diff.ci.p <- pt(q = t.stat, df = n - 1, lower.tail = FALSE)

            lower <- mean - 1.96 * se
            upper <- mean + 1.96 * se

            return(c(lower, mean, upper, diff.ci.p))
        }
    ''')

    diff_test = robjects.r['cindex_diff'](r_groundtruth, r_predictions, r_predictions2)

    lower = diff_test[0]
    mean = diff_test[1]
    upper = diff_test[2]
    pvalue = diff_test[3]

    return {'lower': lower, 'mean': mean, 'upper': upper, 'pvalue': pvalue}


def test_ND(groundtruth, predictions, G=10, greenwood=False):
    """Perform the Nam-D'Agostino Test (in the uncensored setting)

    Originally proposed here:
    https://www.sciencedirect.com/science/article/pii/S0169716103230017

    And clearly formulated here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4555993/

    Args:
        groundtruth: continuous ground truth outcomes
        predictions: continuous predicted outcomes
        G: number of groups
        greenwood: perform GND test instead of ND. not necessary in the
                   uncensored setting.
    """
    # Divide predictions into G evenly spaced buckets
    min_prediction = predictions.min()
    max_prediction = predictions.max()
    groups = np.linspace(min_prediction, max_prediction, num=G+1)

    # Initialize chi squared statistic.
    chi2_nd = 0.

    for g in range(G):
        group_low = groups[g]
        group_high = groups[g+1]

        if g < G - 1:
            # Get indices of predictions within bucket ranges.
            predictions_in_group_inds = ((predictions >= group_low) &
                                         (predictions < group_high))
        else:
            predictions_in_group_inds = ((predictions >= group_low) &
                                         (predictions <= group_high))

        # Get the number of items in the bucket.
        size_of_group = predictions_in_group_inds.sum()

        if size_of_group  < 2:
            raise ValueError("Stopped because at least one of the groups " +
                             "contains <2 events. Consider collapsing some " +
                             "groups.")

        else:

            # Get the predictions and ground truths in the bucket.
            predictions_in_group = predictions[predictions_in_group_inds]
            groundtruth_in_group = groundtruth[predictions_in_group_inds]

            # Compute the mean prediction and ground truth in the bucket.
            mean_predictions_in_group = predictions_in_group.mean()
            mean_groundtruth_in_group = groundtruth_in_group.mean()

            # Compute the (weighted) squared difference between the means.
            numerator = np.square(mean_groundtruth_in_group -
                                  mean_predictions_in_group) * size_of_group


            if greenwood:
                # For GND test, Normalize by the variance of the ground truth
                # in the bucket.
                denominator = np.var(groundtruth_in_group)

            else:
                # Normalize by the variance of the predictions in the bucket.
                denominator = np.var(predictions_in_group)

            summand = numerator / denominator
        
        chi2_nd += summand

    p_value = stats.distributions.chi2.sf(chi2_nd, G-1)

    return p_value
