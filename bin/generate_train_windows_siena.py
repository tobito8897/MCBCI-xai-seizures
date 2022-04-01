#!/usr/bin/python3.7
"""
Usage:
    generate_train_windows_siena.py --patient=<p>

Options:
    --patient=<p>     Patient used for evaluation
"""
import os
import sys
import logging
import numpy as np
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.files import *
from utils.parsers import *
from utils.windowing import *


OPTS = docopt(__doc__)
settings = settings["siena"]
os.environ["FSAMPLING"] = str(settings["fs"])
os.environ["NCHANNELS"] = str(settings["num_channels"])
os.environ["LWINDOW"] = str(settings["length"])


def main():
    expected_channels = read_txt_file(settings["channels"])
    metadata_file = list_files(settings["metadata"] + "/" +
                               OPTS["--patient"], "*.txt")[0]
    edf_files = [x for x in sorted(list_files(settings["database"] + "/" +
                                              OPTS["--patient"]))
                 if OPTS["--patient"] in x]

    seizure_ranges = {}

    lines = read_txt_file(metadata_file)
    seizure_ranges = get_seizure_ranges_siena(lines)

    for file in edf_files:
        filename = file.split("/")[-1]

        eeg, channels = read_mne_file(file)
        channels = [x.split()[1] if len(x.split()) == 2
                    else x for x in channels]
        eeg = convert_to_bipolar(expected_channels, channels, eeg)

        try:
            noictal_windows = get_noseizure_windows(settings["length"],
                                                    settings["noictal_sample_step"],
                                                    seizure_ranges[filename],
                                                    eeg)
        except Exception as exc:
            logging.error(exc)
            continue

        try:
            ictal_windows = get_seizure_windows(settings["length"],
                                                settings["ictal_sample_step"],
                                                seizure_ranges[filename],
                                                eeg)
        except Exception as exc:
            logging.error(exc)
            continue

        if noictal_windows.shape[0] > ictal_windows.shape[0]*2:
            noictal_windows = random_selection(noictal_windows,
                                               ictal_windows.shape[0]*2)
        else:
            ictal_windows = random_selection(ictal_windows,
                                             noictal_windows.shape[0])
        if noictal_windows.shape[0] > 5000:
            noictal_windows = random_selection(noictal_windows,
                                               5000)
        if ictal_windows.shape[0] > 5000:
            ictal_windows = random_selection(ictal_windows,
                                             5000)
        logging.info("File=%s, Ictal window=%s, No Ictal windows=%s", filename,
                     ictal_windows.shape[0], noictal_windows.shape[0])

        output_dir = settings["windows"]
        filename = filename.split(".")[0].replace("-", "_")

        noictal_windows = [np.round(x*settings["gain"]/settings["units"], 3)
                           for x in noictal_windows]
        ictal_windows = [np.round(x*settings["gain"]/settings["units"], 3)
                         for x in ictal_windows]

        noictal_windows = [x.ravel() for x in noictal_windows]
        ictal_windows = [x.ravel() for x in ictal_windows]

        save_numpy_as_csv(noictal_windows, f"{output_dir}/train/{filename}_noictal.csv")
        save_numpy_as_csv(ictal_windows, f"{output_dir}/train/{filename}_ictal.csv")


if __name__ == "__main__":
    main()
