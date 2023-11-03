#!/usr/bin/python3.7
"""
Usage:
    generate_windows_chb.py --class=<test,train> --overlap=<o> --proportion=<p>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database
"""
import os
import sys
import random
import logging
import numpy as np
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.signals import EegProcessor, IctalDetector, NoIctalDetector
from utils.files import Patients, MetadataList, save_numpy_as_csv


OPTS = docopt(__doc__)
settings = settings["chb-mit"]
CLASS = OPTS["--class"]
OVERLAP = OPTS["--overlap"]
PROPORTION = int(OPTS["--proportion"])
np.random.seed(0)


def main():
    logging.info("Starting the process")
    patients = Patients(settings["root_dir"])
    metadata = MetadataList(settings["meta_dir"])

    for patient in iter(patients):
        for file in iter(patient):
            processor = EegProcessor(settings["bipolar_channels"],
                                     settings["f_samp"],
                                     file)
            processor.clean()
            processor.scale(settings["gain"], settings["units"])
            try:
                processor.select_channels()
            except Exception as e:
                logging.error("Patient: %s, File: %s, Error details: %s",
                              patient.patient_name, file, str(e))
                continue
            current_metadata = metadata.get(patient.patient_name, file)
            if not len(current_metadata) > 1:
                logging.error("File does not contain seizures")
                continue

            ############################################################
            seizures = IctalDetector(current_metadata,
                                     settings["epoch_size"],
                                     float(OVERLAP),
                                     int(settings["f_samp"]),
                                     processor.data)
            ictal_windows = []
            for seizure in iter(seizures):
                for ictal_epoch in seizure:
                    ictal_windows.append(ictal_epoch)

            #############################################################
            ictal_windows = [x.ravel() for x in ictal_windows]

            if not len(ictal_windows):
                logging.info("No ictal windows were found")
                continue

            ############################################################
            no_seizures = NoIctalDetector(current_metadata,
                                          settings["epoch_size"],
                                          int(settings["f_samp"]),
                                          processor.data)
            noictal_windows = []
            for no_seizure in iter(no_seizures):
                for noictal_epoch in no_seizure:
                    noictal_windows.append(noictal_epoch)

            #############################################################
            noictal_windows = [x.ravel() for x in noictal_windows]

            if PROPORTION:
                subsample = int(len(ictal_windows)*PROPORTION)
                subsample = subsample if subsample < len(noictal_windows) else len(noictal_windows)
                indexes = list(range(len(noictal_windows)))
                random.shuffle(indexes)
                noictal_windows = [noictal_windows[x] for x in indexes[:subsample]]

            if len(ictal_windows) > 200:
                ictal_windows = ictal_windows[:200]

            if len(ictal_windows) > len(noictal_windows):
                ictal_windows = ictal_windows[:len(noictal_windows)+1]

            filename = file.split("/")[-1]
            filename = filename.split(".")[0]
            filename = f"{settings['windows_dir']}/{CLASS}/{filename}_" \
                       f"{settings['seizure_type']}_ictal.csv"
            save_numpy_as_csv(ictal_windows, filename)
            logging.info("Number of ictal epochs: %s", len(ictal_windows))

            filename = file.split("/")[-1]
            filename = filename.split(".")[0]
            filename = f"{settings['windows_dir']}/{CLASS}/{filename}_" \
                       f"{settings['seizure_type']}_noictal.csv"
            save_numpy_as_csv(noictal_windows, filename)

            logging.info("Number of no ictal epochs: %s", len(noictal_windows))



if __name__ == "__main__":
    main()
