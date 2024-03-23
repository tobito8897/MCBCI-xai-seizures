#!/usr/bin/python3.7
"""
Usage:
    summarize_ml_metrics.py --file=<f>

Options:
    --file=<p>       Full file name
"""
import sys
import logging
import numpy as np
import pandas as pd
from docopt import docopt
sys.path.append("../")

from utils import settings

OPTS = docopt(__doc__)
DBCONF = settings["chb-mit"]


def main():
    metrics_dir = DBCONF["metrics_dir"]
    metrics_file = f"{metrics_dir}{OPTS['--file']}"
    performances = pd.read_csv(metrics_file)
    grouped_perfs = performances.groupby(["patient"])
    mean_perfs = grouped_perfs[["f1_score", "sensitivity", "specificity", "accuracy", "auc_roc"]].mean()

    output_file = metrics_file.replace(".csv", "_summary.csv")
    print(mean_perfs)
    mean_perfs.to_csv(output_file)


if __name__ == "__main__":
    main()
