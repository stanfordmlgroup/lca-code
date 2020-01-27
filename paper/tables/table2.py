import numpy as np
import pandas as pd

from aihc_stats.stats.point_metrics import get_simple_point_metrics

"""
This script generates the .tsv file for Table 1
in TumorAssistant paper.
"""

# set this value given the path to the results csv
PATH_TO_RESULTS = "../data/experiment_results.csv"
results = pd.read_csv(PATH_TO_RESULTS)

# remove training slides
results = results[~results["slide"].str.contains("slide0")]

table = {}

# generate point metrics per individual
for name in results["name"].agg('unique'):
    participant_results = results[results['name'] == name]
    participant_type = participant_results['type'].values[0]
    ground_truth = participant_results['gt'].values
    asst_pred = participant_results['asst_pred'].values
    unasst_pred = participant_results['unasst_pred'].values
    model_pred = np.asarray(participant_results['confidence'].values > .5, dtype=np.int32)

    # use CC as positive (reverse of original data)
    ground_truth = 1 - ground_truth
    asst_pred = 1 - asst_pred
    unasst_pred = 1 - unasst_pred
    model_pred = 1 - model_pred

    table[name] = {}
    table[name]['type'] = participant_type
    table[name]['assisted'] = get_simple_point_metrics(groundtruth=ground_truth,
                                                       predictions=asst_pred, ci=True)
    table[name]['unassisted'] = get_simple_point_metrics(groundtruth=ground_truth,
                                                         predictions=unasst_pred, ci=True)
    table[name]['model'] = get_simple_point_metrics(groundtruth=ground_truth,
                                                    predictions=model_pred, ci=True)


def get_ci(dictionary):
    """
    Helper function to generate formatted CI string
    :param dictionary: dict containing the values
    :return: formatted string
    """
    return "{0:.2f} ({1:.2f}, {2:.2f})".format(dictionary["mean"],
                                               dictionary["lower"], dictionary["upper"])


# metrics to print to TSV
metrics = ['accuracy', 'specificity', 'sensitivity']
modes = ['model', 'unassisted', 'assisted']

with open("table1.tsv", "w") as file:
    # write header
    header = list()
    header.append("participant")
    for metric in metrics:
        for mode in modes:
            header.append("%s %s" % (mode, metric))
    file.write("\t".join(header) + "\n")

    for name in table.keys():
        participant = table[name]
        output = list()
        output.append(name)
        for metric in metrics:
            for mode in modes:
                output.append(get_ci(participant[mode][metric]))
        file.write("\t".join(output) + "\n")
