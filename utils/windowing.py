#!/usr/bin/python3.7
import os
import re
import logging
import numpy as np


__all__ = ["select_channels", "get_seizure_windows", "get_noseizure_windows",
           "random_selection", "convert_to_bipolar"]


def select_channels(expected_ch: list, current_channels,
                    data: np.array) -> np.array:
    count = 0
    filtered_data = np.zeros([len(expected_ch), data.shape[1]])
    for ch in expected_ch:
        for idx, _ in enumerate(current_channels):
            if ch in _:
                try:
                    filtered_data[count, :] = data[idx, :]
                    count += 1
                except Exception as e:
                    logging.warning(e)
                break
    return filtered_data


def convert_to_bipolar(bipolar_montage: list, current_channels: dict,
                       data: np.array) -> np.array:
    current_channels = {v.lower(): k for k, v in enumerate(current_channels)}
    bipolar_montage = [x.split("-") for x in bipolar_montage]
    output_data = np.zeros((len(bipolar_montage), data.shape[1]))

    for index, (ch1, ch2) in enumerate(bipolar_montage):
        ch1 = current_channels[ch1.lower()]
        ch2 = current_channels[ch2.lower()]
        output_data[index, :] = data[ch1, :] - data[ch2, :]
    return output_data


def get_windows(length: int, step: int, seizure_intervals: tuple,
                data: np.array, ictal: bool = True) -> list:
    fs = int(os.environ.get("FSAMPLING"))
    start = 0
    end = data.shape[1]
    windows = []

    while True:
        ictal_flag = 0
        for interval in seizure_intervals:
            if start/fs >= interval[0] and (start + step)/fs <= interval[1]:
                ictal_flag = 1
                break
            elif start/fs <= interval[1] or (start + step)/fs >= interval[0]:
                ictal_flag = -1
                break

        if ictal and ictal_flag == 1:
            window = data[:, start: start+length]
            windows.append(window)
        elif not ictal and ictal_flag == -1:
            window = data[:, start: start+length]
            windows.append(window)
        start += step
        if start >= end:
            break
    windows = [x for x in windows if x.shape[1] == length]
    if len(windows) > 10000:
        windows = windows[:10000]
    return np.stack(windows, axis=0)


def get_seizure_windows(*args):
    return get_windows(*args, ictal=True)


def get_noseizure_windows(*args):
    return get_windows(*args, ictal=False)


def random_selection(windows: np.array, selected: int) -> np.array:
    np.random.seed(1)
    permutated_index = np.random.permutation(windows.shape[0]).tolist()

    if windows.shape[0] > selected:
        permutated_index = permutated_index[:selected]
        windows = windows[permutated_index, :, :]

    return windows
