#!/usr/bin/python3.7
"""
Usage:
    visualization_tools_hossain.py --patient=<p>

Options:
    --patient=<p>     Patient used for evaluation
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
settings = settings["wang_1d"]


def main():
    input_dir = settings["stats"]
    files = list_files(input_dir, "history_*")
    history_files = [x for x in files if str(PATIENT) in x]
    files = list_files(input_dir, "stats_*")
    stats_files = [x for x in files if str(PATIENT) in x]

    for file in history_files:
        history = read_pickle(file)
        modifier = "_" if "full" not in file else "_full_"
        filename = "{}/validation_curve{}{}".format(settings["images"],
                                                    modifier, PATIENT)
        plot_validation_curve(history["acc"], history["val_acc"], filename)

    for file in stats_files:
        predictions = read_pickle(file)
        modifier = "_" if "full" not in file else "_full_"
        filename = "{}/confusion_matrix{}{}".format(settings["images"],
                                                    modifier, PATIENT)
        plot_confusion_matrix(predictions["real"], predictions["predicted"],
                              filename)


if __name__ == "__main__":
    main()
