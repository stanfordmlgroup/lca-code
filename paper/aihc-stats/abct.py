import numpy as np

from args import ArgParser
from config import load_config
from load import load_data, load_r
from stats import tests, classification_metrics as c_metrics
from figure import FigurePlotter
from table import ABCTTableWriter
from logger import Logger


def test_auc_differences(experiment2probabilities,
                         compare_groundtruth, logger):

    # Get dictionary from experiment name -> R ROC curve object.
    experiment_curves = c_metrics.get_experiment_roc_curves(experiment2probabilities,
                                                            compare_groundtruth)

    # Test differences in AUC between default setting and other experiments.
    default_experiment_curve = experiment_curves["default"]

    for experiment in experiment_curves:
        if experiment != "default":
            p_value = tests.test_auc_difference(default_experiment_curve,
                                                experiment_curves[experiment])

            logger.log(f"p-value for experiment {experiment}: {p_value}.")


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()

    logger = Logger(args.log_dir)
    config = load_config(args.config_path, logger)

    # Load R functions.
    load_r()

    compare_groundtruth = load_data(config["compare_groundtruth_path"])
    experiments = config["compare_probabilities_paths"].keys()
    experiment2probabilities = {}
    experiment2metrics = {}
    for experiment in experiments:

        print(experiment)

        compare_trials = config["compare_probabilities_paths"][experiment]
        # Ensemble the trials.
        compare_probabilities = np.mean([load_data(compare_trial)
                                         for compare_trial in compare_trials],
                                        axis=0)

        summary_metrics = c_metrics.compute_summary_metrics(compare_groundtruth,
                                                            compare_probabilities,
                                                            ci=True,
                                                            logger=logger)

        experiment2probabilities[experiment] = compare_probabilities
        experiment2metrics[experiment] = summary_metrics

    table_generator = ABCTTableWriter(logger)
    table_generator.write_tables(experiment2metrics,
                                 config["operating_point_value"])

    test_auc_differences(experiment2probabilities,
                         compare_groundtruth, logger)

    # Get results of best model on the test set
    eval_groundtruth = load_data(config["eval_groundtruth_path"])

    tune_threshold = config["tune_threshold"]
    if tune_threshold:
        tune_groundtruth = load_data(config["tune_groundtruth_path"])
    else:
        tune_groundtruth = None

    eval_experiments = config["eval_probabilities_paths"].keys()
    for eval_experiment in eval_experiments:

        print(eval_experiment)
        eval_trials = config["eval_probabilities_paths"][eval_experiment]
        # Ensemble the trials.
        eval_probabilities = np.mean([load_data(eval_trial)
                                      for eval_trial in eval_trials],
                                     axis=0)

        if tune_threshold:
            tune_trials = config["tune_probabilities_paths"][eval_experiment]
            tune_probabilities = np.mean([load_data(tune_trial)
                                          for tune_trial in tune_trials],
                                         axis=0)
        else:
            tune_probabilities = None

        summary_metrics, point_metrics =\
            c_metrics.get_classification_metrics(eval_groundtruth,
                                                 eval_probabilities,
                                                 config["tune_threshold"],
                                                 tune_groundtruth,
                                                 tune_probabilities,
                                                 config["operating_point_value"],
                                                 ci=True,
                                                 logger=logger)
