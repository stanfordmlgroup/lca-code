import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from tabulate import tabulate

import util
import re
import json
import argparse

import pandas as pd

MEAN_COL = "_Mean"
LOWER_COL = "_Lower"
UPPER_COL = "_Upper"
PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


class TableParser(object):
    """argument parser for creating final tables"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CXR')
        
        # Args
        self.parser.add_argument('experiments', nargs='+')
        self.parser.add_argument('--results_dir', type=str, default='results/',
                                 help='Where results of evaluated models are stored')
        self.parser.add_argument('--split', type=str, default='valid', choices=('valid', 'test'),
                                 help='Where results of evaluated models are stored')

        self.parser.add_argument('--table_output_dir', type=str, default='./tables', help='Where to output')

        self.parser.add_argument('--write_uncertainty_table', type=util.str_to_bool, default=False, help='If true, write uncertainty table')
        self.parser.add_argument('--write_spreadsheets', type=util.str_to_bool, default=False, help='If true, write prediction spreadsheets')

    def parse_args(self):
        args = self.parser.parse_args()
        return args


def create_tables(args):
    results = load_results(args)
    if args.write_uncertainty_table:
        print('Writing Uncertainty Table')
        write_uncertainty_table(args, results)  
    elif args.write_spreadsheets:
        write_spreadsheets(args, results)


def load_results(args):

    results = []
    for experiment in args.experiments:
        exp_results_path = Path(args.results_dir) / experiment / args.split / "scores.csv"
        exp_results = pd.read_csv(exp_results_path)

        results.append(exp_results)

    return pd.concat(results)


def get_bolder(reported_metric, max_val, min_val):
    
    def bolder(row):
        mean = row[reported_metric + MEAN_COL]
        lower = row[reported_metric + LOWER_COL]
        upper = row[reported_metric + UPPER_COL]
        if mean == max_val:
            return "textbf{" + f"{float(mean):.3f}" + "} (textbf{" f"{float(lower):.3f}" + "},textbf{" + f"{float(upper):.3f}" + "})" 
        elif mean == min_val:
            return "textit{" + f"{float(mean):.3f}" + "} (textit{" f"{float(lower):.3f}" + "},textit{" + f"{float(upper):.3f}" + "})" 
        else:
            return f"{float(mean):.3f} ({float(lower):.3f}, {float(upper):.3f})"
    
    return bolder


def write_uncertainty_table(args, results):

    metric = 'AUROC'

    for pathology in PATHOLOGIES:
        reported_metric = pathology + metric
        assert results.columns.contains(reported_metric), "Results CSV did not measure needed metric - {}".format(reported_metrics[-1])

        lower_col = reported_metric + LOWER_COL
        mean_col = reported_metric + MEAN_COL
        upper_col = reported_metric + UPPER_COL

        results[lower_col], results[mean_col], results[upper_col] = zip(*results[reported_metric].str.split('|'))

        max_val = results[mean_col].max()
        min_val = results[mean_col].min()
        bolder = get_bolder(reported_metric, max_val, min_val)
        results[pathology] = results.apply(bolder, axis=1)

    results["Approach"] = args.experiments

    columns = ["Approach"] + PATHOLOGIES
    latex_table = tabulate(results[columns], headers='keys', showindex=False, tablefmt="latex_booktabs")

    table_output_dir = Path(args.table_output_dir)
    table_output_dir.mkdir(exist_ok=True)
    table_output_file = table_output_dir / f'{args.split}_table.txt'
    with open(table_output_file, 'w') as f:
        print(latex_table.replace("lllllll", "l|ccccc|c").replace("tex", "\\tex").replace("\{", "{").replace("\}", "}"), file=f)

from sklearn.metrics import roc_auc_score
from models import CSVReaderModel
from dataset import TASK_SEQUENCES
import numpy as np

def write_spreadsheets(args, results):

    master = pd.read_csv("/data3/xray4all/master_unpostprocessed.csv")
    valid = master[master["DataSplit"] == "valid"][["SimpleValidImageID", "Path"]]
    valid["Path"] = valid["Path"].apply(lambda x: "/data3/xray4all/" + x)
    valid["Path"] = valid["Path"].apply(lambda x: "/".join(x.split("/")[:-1]))
    valid = valid.drop_duplicates()
    valid = valid.rename(columns={"SimpleValidImageID": "Study #"})
    
    gt = pd.read_csv("/data3/xray4all/valid_rad_majority.csv").merge(valid, on="Study #")

    experiment_predictions = []
    for experiment in args.experiments:

        predictions_path = Path(args.results_dir) / (experiment + "-predict") / args.split / "all_combined.csv"

        task_sequence = TASK_SEQUENCES["stanford"]
        csv_model = CSVReaderModel(predictions_path, task_sequence)
        probs = []
        for path in gt["Path"]:
            probs.append(csv_model.forward([path]).cpu().numpy().squeeze())

        probs = np.array(probs)

        predictions = pd.DataFrame({path: probs[:, i] for path, i in task_sequence.items()})

        experiment_predictions.append((experiment, predictions))

    for pathology in PATHOLOGIES:

        spreadsheet = pd.DataFrame({"Groundtruth": gt[pathology]})

        for experiment, predictions in experiment_predictions:
            spreadsheet[experiment] = predictions[pathology]

        spreadsheet.to_csv(f"spreadsheets/{pathology}.csv")




if __name__ == '__main__':
    parser = TableParser()
    create_tables(parser.parse_args())
