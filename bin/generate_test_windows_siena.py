#!/usr/bin/python3.7
"""
Usage:
    generate_test_windows_siena.py --patient=<p> --db=<d>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database
"""
import os
import sys
import logging
import numpy as np
from scipy import signal
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.files import *
from utils.parsers import *
from utils.windowing import *


OPTS = docopt(__doc__)
settings = settings[OPTS["--db"]]
os.environ["FSAMPLING"] = str(settings["fs"])
os.environ["NCHANNELS"] = str(settings["num_channels"])
os.environ["LWINDOW"] = str(settings["length"])


def chunks(lst, n):
    for i in range(0, lst.shape[1], n):
        yield lst[:, i:i + n]


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
        for idx, segment in enumerate(chunks(eeg, settings["fs"]*60*60)):
            try:
                noictal_windows = get_noseizure_windows(settings["length"],
                                                        settings["length"],
                                                        seizure_ranges[filename],
                                                        segment,
                                                        init_time=idx*60*60)
            except Exception as exc:
                logging.error(exc)
                continue

            try:
                ictal_windows = get_seizure_windows(settings["length"],
                                                    settings["length"],
                                                    seizure_ranges[filename],
                                                    segment,
                                                    init_time=idx*60*60)
            except Exception as exc:
                logging.error(exc)
                continue

            if noictal_windows.shape[0] > ictal_windows.shape[0]*2:
                noictal_windows = random_selection(noictal_windows,
                                                   ictal_windows.shape[0]*10)
            else:
                ictal_windows = random_selection(ictal_windows,
                                                 noictal_windows.shape[0])

            noictal_windows = signal.decimate(noictal_windows, 2, axis=-1)
            ictal_windows = signal.decimate(ictal_windows, 2, axis=-1)

            logging.info("File=%s, Ictal window=%s, No Ictal windows=%s",
                         filename, ictal_windows.shape[0],
                         noictal_windows.shape[0])
            if len(ictal_windows) == 0:
                continue

            output_dir = settings["windows"]
            new_filename = filename.split(".")[0].replace("-", "_") + str(idx)

            noictal_windows = [np.round(x*settings["gain"]/settings["units"], 3)
                               for x in noictal_windows]
            ictal_windows = [np.round(x*settings["gain"]/settings["units"], 3)
                             for x in ictal_windows]

            noictal_windows = [x.ravel() for x in noictal_windows]
            ictal_windows = [x.ravel() for x in ictal_windows]

            save_numpy_as_csv(noictal_windows, f"{output_dir}/test/{new_filename}_noictal.csv")
            save_numpy_as_csv(ictal_windows, f"{output_dir}/test/{new_filename}_ictal.csv")


if __name__ == "__main__":
    main()
