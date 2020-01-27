import argparse
from scipy import stats
import numpy as np
import pandas as pd

def random_effects_ci(df):
    means = df.groupby('name').accuracy.mean().values
    variances = df.groupby('name').accuracy.var().values
    alpha_FE = 1 / variances
    theta_FE = np.inner(alpha_FE, means) / np.sum(alpha_FE)
    S = len(means)
    num = np.inner(alpha_FE, (means - theta_FE)**2) - S + 1
    denom = np.sum(alpha_FE) - np.sum(alpha_FE / np.sum(alpha_FE))
    var_RE = max(0, num / denom)

    tot_var = S*var_RE + np.sum(variances)
    return np.mean(means), stats.norm.interval(0.95, np.mean(means), tot_var / S**2) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=str)

    args = parser.parse_args()
    df = pd.read_csv(args.df)

    mean, interval = random_effects_ci(df)
    print("mean: {}".format(mean))
    print("CI = {}".format(interval))
