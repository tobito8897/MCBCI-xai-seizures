#!/usr/bin/python3.7
import re
from collections import defaultdict
from datetime import datetime, timedelta


__all__ = ["get_seizure_ranges", "get_seizure_ranges_siena"]


def get_seizure_single_file(text_lines: list) -> tuple:
    seizures = []
    filename = re.search("(?<=:)\s.+", text_lines[0]).group(0).strip()
    number_seizures = int(re.findall("(?<=:\s).+", text_lines[1])[0])

    if number_seizures == 0:
        return filename, set()

    for idx in range(number_seizures):
        text = text_lines[(idx+1)*2]
        start = int(re.findall("(?<=:\s)[0-9]+", text)[0])
        text = text_lines[(idx+1)*2 + 1]
        end = int(re.findall("(?<=:\s)[0-9]+", text)[0])
        seizures.append((start, end))
    return filename, tuple(seizures)


def get_seizure_ranges(text_lines: list) -> tuple:
    clean_lines, single_file = [], []
    text_lines = [re.sub("\s+", " ", x.lower()) for x in text_lines]

    skip = True
    clean_lines = []
    for line in text_lines:
        if "file name" in line:
            skip = False
        if "changed" in line:
            skip = True
        if not skip:
            clean_lines.append(line)

    clean_lines, text_lines = [], clean_lines

    for line in text_lines:
        if line == "":
            clean_lines.append(single_file)
            single_file = []
        elif not any(x in line for x in ["file start", "file end"]):
            single_file.append(line)
    if len(single_file):
        clean_lines.append(single_file)

    for file in clean_lines:
        filename, seizures = get_seizure_single_file(file)
        yield filename, seizures


def get_seizure_siena(lines: list) -> tuple:
    assert "Registration start" in lines[0]
    recording_start_time = re.search("(?<=:\s).*", lines[0]).group(0)
    seizure_start_time = re.search("(?<=:\s).*", lines[2]).group(0)
    seizure_end_time = re.search("(?<=:\s).*", lines[3]).group(0)
    recording_start_time = datetime.strptime(recording_start_time.strip(), "%H.%M.%S")
    seizure_start_time = datetime.strptime(seizure_start_time.strip(), "%H.%M.%S")
    seizure_end_time = datetime.strptime(seizure_end_time.strip(), "%H.%M.%S")

    if recording_start_time > seizure_start_time:
        recording_start_time = recording_start_time - timedelta(days=1)

    start_seconds = (seizure_start_time - recording_start_time).seconds
    end_seconds = (seizure_end_time - recording_start_time).seconds
    return (start_seconds, end_seconds)


def get_seizure_ranges_siena(text_lines: list) -> dict:
    seizures = defaultdict(list)
    for index, line in enumerate(text_lines):
        if "File name" in line:
            filename = line.split(" ")[-1]
            seizure = get_seizure_siena(text_lines[index+1: index+5])
            seizures[filename].append(seizure)
    return seizures
