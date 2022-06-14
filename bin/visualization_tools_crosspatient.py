#!/usr/bin/python3.7
"""
Usage:
    visualization_tools_crosspatient.py --patient=<p> --db=<d> --model=<m>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database to use for training
    --model=<m>       Hossain, Wang 1d or Wang 2d
"""
import sys
import numpy as np
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.visualization import *
from utils.files import *

OPTS = docopt(__doc__)
PATIENT = OPTS["--patient"]
settings = settings[OPTS["--model"]]


def main():
    input_dir = settings["stats"]
    files = list_files(input_dir, f"history_{OPTS['--db']}_*")
    history_files = [x for x in files if str(PATIENT) in x]
    files = list_files(input_dir, f"stats_{OPTS['--db']}_*")
    stats_files = [x for x in files if str(PATIENT) in x]

    for file in history_files:
        history = read_pickle(file)
        modifier = "_" if "full" not in file else "_full_"
        filename = "{}/accuracy_curve{}_{}{}".format(settings["images"],
                                                     OPTS['--db'],
                                                     modifier, PATIENT)
        plot_validation_curve(history["acc"], history["val_acc"], filename,
                              "Accuracy")

        filename = "{}/loss_curve{}_{}{}".format(settings["images"],
                                                 OPTS['--db'],
                                                 modifier, PATIENT)
        plot_validation_curve(history["loss"], history["val_loss"], filename,
                              "Loss")

    for file in stats_files:
        predictions = read_pickle(file)
        modifier = "_" if "full" not in file else "_full_"
        filename = "{}/confusion_matrix_{}_{}{}".format(settings["images"],
                                                        OPTS['--db'],
                                                        modifier, PATIENT)
        plot_confusion_matrix(predictions["real"], predictions["predicted"],
                              filename)


if __name__ == "__main__":
    main()
