#!/usr/bin/python3.7
import os
import csv
import glob
import pickle
from typing import Any
import mne
import gzip
import numpy as np
import pandas as pd


__all__ = ["list_files", "read_mne_file", "read_txt_file",
           "write_pickle", "read_pickle", "save_numpy_as_csv"]


def list_files(directory: str, glob_pattern: str = "*.edf") -> list:
    return glob.glob(directory + "/" + glob_pattern)


def read_mne_file(filename: str) -> np.array:
    data = mne.io.read_raw_edf(filename)
    return data.get_data(), data.ch_names


def read_txt_file(filename: str) -> list:
    with open(filename, "r") as stream:
        lines = stream.readlines()
    lines = [x.strip("\n") for x in lines]
    return lines


def load_csv_numpy(filename: str, num_channels: int):
    array = []
    with open(filename, "r") as stream:
        reader = csv.reader(stream)
        for row in reader:
            length = int(len(row)/num_channels)
            array.append(np.array(row).reshape(num_channels, length))
    array = np.stack(array, axis=0)
    return array.astype(np.float)


def write_pickle(filename: str, data: dict):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content


def save_numpy_as_csv(data: list, filename: str):
    with open(filename, "w") as stream:
        output = csv.writer(stream)
        output.writerows(data)
