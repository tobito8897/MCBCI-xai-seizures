#!/usr/bin/python3.7
"""
1.- Download all EDF files form physionet CHB-MIT
"""
import os
import sys
import logging
from os import listdir
from os.path import isfile, join
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from utils import settings
from utils.http import download_file


record_seizure_file = os.path.join(current_dir, settings["chb-mit_wang"]["seizure_records"])
dst_directory = os.path.join(current_dir, settings["chb-mit_wang"]["database"])
chb_url = "https://physionet.org/files/chbmit/1.0.0/"


def main():
    downloaded_files = [f for f in listdir(dst_directory)
                        if isfile(join(dst_directory, f))]

    with open(record_seizure_file, "r") as f:
        files = [a.replace("\n", "") for a in f.readlines()]

    for a in files:
        if not a:
            continue
        if a.split("/")[-1] in downloaded_files:
            logging.info("Skipping file: %s" % a)
        else:
            size = download_file(source=chb_url + a, destine=dst_directory)
            logging.info("File dowloaded: %s, size: %s" % (a, size))


if __name__ == "__main__":
    main()