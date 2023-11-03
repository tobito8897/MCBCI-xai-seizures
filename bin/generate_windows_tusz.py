#!/usr/bin/python3.7
"""
Usage:
    generate_windows_tusz.py --class=<test,train> --overlap=<o> --proportion=<p>

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
from utils.signals import EegProcessorTusz, IctalDetector, NoIctalDetector
from utils.files import PatientsTusz, MetadataListTusz, save_numpy_as_csv


OPTS = docopt(__doc__)
settings = settings["tusz"]
CLASS = OPTS["--class"]
OVERLAP = OPTS["--overlap"]
PROPORTION = int(OPTS["--proportion"])
np.random.seed(0)


AVOID = ['8476', '10547', '10158', '8174', '10587', '2297', '8615', '6986',
         '9540', '8479', '9623', '6507', '1052', '10106', '8606', '9842',
         '5034', '9578', '8616', '7623', '8608', '8460', '630', '9570',
         '8453', '6083', '3281', '10088']


def chunks(lst, n):
    for i in range(0, lst.shape[1], n):
        yield lst[:, i:i + n]


def main():
    logging.info("Starting the process")
    patients = PatientsTusz(settings["root_dir"])
    metadata = MetadataListTusz(settings["meta_dir"])
    overall_patients = 0

    for patient in iter(patients):

        if len(patient) < 2:
            logging.warning("Not enough files")
            continue

        initial_f_samp = None

        for file in iter(patient):

            seizure_type = None

            processor = EegProcessorTusz(settings["bipolar_channels"],
                                         0, file)
            if not initial_f_samp:
                initial_f_samp = processor.f_samp
            if initial_f_samp != processor.f_samp:
                logging.info("Recording has a different FS: %s",
                             processor.f_samp)
                continue

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
                                     int(initial_f_samp),
                                     processor.data)
            ictal_windows = []
            for seizure in iter(seizures):
                for ictal_epoch in seizure:
                    if ictal_epoch.shape[1] != int(initial_f_samp)*settings["epoch_size"]:
                        continue
                    ictal_windows.append(ictal_epoch)

            #############################################################
            ictal_windows = [x.ravel() for x in ictal_windows]

            if not len(ictal_windows):
                logging.info("No ictal windows were found")
                continue

            if not seizure_type:
                seizure_type = current_metadata[1][2]

            ############################################################
            no_seizures = NoIctalDetector(current_metadata,
                                          settings["epoch_size"],
                                          int(initial_f_samp),
                                          processor.data)
            noictal_windows = []
            for no_seizure in iter(no_seizures):
                for noictal_epoch in no_seizure:
                    if noictal_epoch.shape[1] != int(initial_f_samp)*settings["epoch_size"]:
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

            logging.info("Number of ictal epochs: %s", len(ictal_windows))
            logging.info("Number of no ictal epochs: %s", len(noictal_windows))

            if len(ictal_windows) and len(noictal_windows):
                filename = file.split("/")[-1]
                filename = filename.split(".")[0]
                filename = f"{settings['windows_dir']}/{CLASS}/{filename.strip('0')}_" \
                        f"{seizure_type}_ictal.csv"
                save_numpy_as_csv(ictal_windows, filename)

                filename = file.split("/")[-1]
                filename = filename.split(".")[0]
                filename = f"{settings['windows_dir']}/{CLASS}/{filename.strip('0')}_" \
                        f"{seizure_type}_noictal.csv"
                save_numpy_as_csv(noictal_windows, filename)

        overall_patients = overall_patients + 1
        if overall_patients > 120:
            break


if __name__ == "__main__":
    main()
