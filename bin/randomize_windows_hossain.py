#!/usr/bin/python3.7
"""
Usage:
    randomize_windows_hossain.py --patient=<p> --db=<d> --dataset=<d>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database
    --dataset=<d>     test or train
"""
import sys
import random
import logging
import numpy as np
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.files import *
from utils.parsers import *
from utils.windowing import *


np.random.seed(1)
OPTS = docopt(__doc__)
database = settings[OPTS["--db"]]
patient = OPTS["--patient"]
dataset = OPTS["--dataset"]


def shuffle_array(array: list) -> list:
    shuffled_array = []
    indexes = list(range(len(array)))
    random.shuffle(indexes)
    for i in indexes:
        shuffled_array.append(array[i])
    return shuffled_array


def split_array(array: list, batch_size: int) -> list:
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


def save_batch(array: list, batch_number: int, basename: int) -> list:
    filename = basename % f"batch_{batch_number}"
    with open(filename, "w") as fp:
        for line in array:
            fp.write(line + "\n")
    logging.info("File saved %s", filename)


def main():
    logging.info("Patient selected: %s", patient)

    ictal_files = [x for x in list_files(f"{database['windows']}/{dataset}",
                                         "*.csv")
                   if patient in x and "_ictal" in x and "batch" not in x]
    ictal_data = []
    for file in ictal_files:
        ictal_data += read_txt_file(file)

    filename = f"{database['windows']}/{dataset}/{patient}_%s_ictal.csv"
    ictal_data = shuffle_array(ictal_data)
    for i, array in enumerate(split_array(ictal_data,
                                          batch_size=database["minibatch"])):
        save_batch(array,
                   batch_number=i,
                   basename=filename)
    del ictal_data

    noictal_files = [x for x in list_files(f"{database['windows']}/{dataset}",
                                           "*.csv")
                     if patient in x and "_ictal" not in x and "batch" not in x]
    noictal_data = []
    for file in noictal_files:
        noictal_data += read_txt_file(file)

    filename = f"{database['windows']}/{dataset}/{patient}_%s_noictal.csv"
    noictal_data = shuffle_array(noictal_data)
    for i, array in enumerate(split_array(noictal_data,
                                          batch_size=database["minibatch"])):
        save_batch(array,
                   batch_number=i,
                   basename=filename)


if __name__ == "__main__":
    main()
