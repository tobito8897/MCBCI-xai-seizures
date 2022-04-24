#!/usr/bin/python3.7
"""
Usage:
    visualization_tools_wang.py.py --db=<d> --model=<m>

Options:
    --db=<p>          Database to use for training
    --model=<m>       Wang 1d or 2d
"""
import sys
import numpy as np
from docopt import docopt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
sys.path.append("../")

from utils import settings
from utils.visualization import *
from utils.files import *

OPTS = docopt(__doc__)
patients = settings[OPTS['--db']]["patients"]
settings = settings[OPTS["--model"]]


def main():
    input_dir = settings["stats"]
    files = list_files(input_dir, f"stats_{OPTS['--db']}_*")
    filename = "{}/epochsVSperformance_{}".format(settings["images"],
                                                  OPTS['--db'])
    epochs, f1_scores, sensitivities, specificities = [], [], [], []

    for patient in patients:
        ictal_epochs = 0
        stats_files = [x for x in files if str(patient) in x]
        for file in stats_files:
            predictions = read_pickle(file)
            ictal_epochs += np.sum(predictions["real"])
            cmatrix = confusion_matrix(predictions["real"], predictions["predicted"],
                                       labels=[0, 1])
            cmatrix = cmatrix.astype("float") / cmatrix.sum(axis=1)[:, np.newaxis]
            f1s = f1_score(predictions["real"], predictions["predicted"])
            sen = float(cmatrix[1][1]/np.sum(cmatrix[1]))
            spec = float(cmatrix[0][0]/np.sum(cmatrix[0]))
        if ictal_epochs == 0:
            continue
        epochs.append(ictal_epochs)
        f1_scores.append(f1s)
        sensitivities.append(sen)
        specificities.append(spec)

    plot_epochs_vs_performance(epochs, f1_scores, sensitivities, specificities,
                               filename)


if __name__ == "__main__":
    main()
