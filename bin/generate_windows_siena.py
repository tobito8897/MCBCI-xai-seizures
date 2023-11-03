#!/usr/bin/python3.7
"""
Usage:
    generate_windows_siena.py --class=<test,train> --overlap=<o> --proportion=<p>

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
from utils.signals import EegProcessorSiena, IctalDetector, NoIctalDetector
from utils.files import Patients, MetadataListSiena, save_numpy_as_csv


OPTS = docopt(__doc__)
settings = settings["siena"]
CLASS = OPTS["--class"]
OVERLAP = OPTS["--overlap"]
PROPORTION = int(OPTS["--proportion"])
np.random.seed(0)


def main():
    logging.info("Starting the process")
    patients = Patients(settings["root_dir"])
    metadata = MetadataListSiena(settings["meta_dir"])

    for patient in iter(patients):
        if len(patient) < 2:
            logging.warning("Not enough files")
            continue
        for file in iter(patient):

            processor = EegProcessorSiena(settings["bipolar_channels"],
                                          settings["f_samp"],
                                          file)
            processor.clean()
            processor.scale(settings["gain"], settings["units"])
            #processor.downsample(int(settings["f_samp"]/256))
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
                    if ictal_epoch.shape[1] != int(settings["f_samp"])*settings["epoch_size"]:
                        continue
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
                    if noictal_epoch.shape[1] != int(settings["f_samp"])*settings["epoch_size"]:
                        continue
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
            filename = filename.replace("-", "_")
            filename = f"{settings['windows_dir']}/{CLASS}/{filename}_" \
                       f"{settings['seizure_type'][filename.split('_')[0]]}_ictal.csv"
            save_numpy_as_csv(ictal_windows, filename)
            logging.info("Number of ictal epochs: %s", len(ictal_windows))

            filename = file.split("/")[-1]
            filename = filename.split(".")[0]
            filename = filename.replace("-", "_")
            filename = f"{settings['windows_dir']}/{CLASS}/{filename}_" \
                       f"{settings['seizure_type'][filename.split('_')[0]]}_noictal.csv"
            save_numpy_as_csv(noictal_windows, filename)

            logging.info("Number of no ictal epochs: %s", len(noictal_windows))


if __name__ == "__main__":
    main()
