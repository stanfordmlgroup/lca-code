"""Average multiple model outputs in CSV format.

This is used to produce the ensemble labels for uncertainty experiments.
"""

import argparse
import pandas as pd
import util

from dataset.constants import COL_PATH


def main(args):

    # Load all DataFrames and index by study ID.
    dfs = [pd.read_csv(csv_path) for csv_path in args.input_paths]

    # Get mean prediction for each study ID
    concat_df = pd.concat(dfs, sort=False)
    avg_df = concat_df.groupby(COL_PATH, as_index=False).mean()

    # Write the averaged DataFrame to disk
    print('Writing averaged CSV to {}'.format(args.output_path))
    avg_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_paths', nargs='+',
                        help='List of input CSV files to average together.')
    parser.add_argument('--output_path', required=True, type=str)

    main(parser.parse_args())
