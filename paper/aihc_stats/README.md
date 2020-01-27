# aihc-stats
Module for statistical evaluation and plotting. Work in progress.

## Prerequisites
1. Clone the repo
`git clone git@github.com:stanfordmlgroup/aihc-stats.git`
2. If on deep, run
`source activate stats`.

Otherwise, create the environment
`conda env create -f environment.yml`

You may need to install some R packages:
`from rpy2.robjects.packages import importr`
`utils = importr('utils')`
`utils.install_packages('pROC')`
`utils.install_packages('binom')`

## Usage from the repo
Make a configuration file (see [config.json](https://github.com/stanfordmlgroup/aihc-stats/blob/master/abct-config.json) for an example).

Then run
`python abct.py -c {path to config file} -l {path to log file}`

Arguments default to `config.json` and `logs/` respectively.

## Files
- `args/arg_parser.py` defines the argument parser for the CLI through `abct.py`.
- `config/config.py` contains functions to load and validate the configuration file.
- `logger/logger.py` defines the logger class for printing results to stdout and file.
- `load/load.py` contains functions to load data and load R packages.
- `stats/metrics.py` contains functions for computing metrics on binary groundtruth data.
- `stats/tests.py` contains functions for statistical tests.
- `figure/figure.py` defines the plotter class for plotting figures.
- `table/table.py` defines the generator class for generating tables.
- `util/util.py` contains various utility functions.