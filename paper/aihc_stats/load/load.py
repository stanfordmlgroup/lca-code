import pandas as pd
import numpy as np


def load_data(path, csv_header=None):
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".txt"):
        data = np.loadtxt(path)
    elif path.endswith(".csv"):
        if csv_header is None:
            # Assume one column csv with no header.
            data = pd.read_csv(path,
                               header=None,
                               names="Col")["Col"].as_matrix()
        else:
            data = pd.read_csv(path)[csv_header].as_matrix()  
    else:
        raise ValueError(f"Path prefix {path[-4:]} not supported.")

    return data


def load_r():
    def _install_r_packages(packnames):
        # import rpy2's package module
        import rpy2.robjects.packages as rpackages

        # import R's utility package
        utils = rpackages.importr('utils')

        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list


        # R vector of strings
        from rpy2.robjects.vectors import StrVector

        # Selectively install what needs to be install.
        # We are fancy, just because we can.
        names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))


    # R package names
    packnames = ('ggplot2', 'hexbin', 'irr', 'pROC', 'binom', 'ROCR', 'MRIaggr')
    _install_r_packages(packnames)

    # for using numpy arrays
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    # for using pandas
    import rpy2.robjects.pandas2ri
    rpy2.robjects.pandas2ri.activate()