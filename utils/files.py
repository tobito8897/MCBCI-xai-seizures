#!/usr/bin/python3.7
import os
import re
import csv
import glob
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta


class Patient():

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.file_list = root_dir
        self.patient_name = root_dir
        self.counter = -1

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, root_dir: str):
        self._file_list = glob.glob(os.path.join(root_dir, "*.edf"))

    @property
    def patient_name(self):
        return self._patient_name

    @patient_name.setter
    def patient_name(self, root_dir: str):
        self._patient_name = os.path.split(root_dir)[-1]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.file_list)

    def __next__(self):
        if self.counter + 1 == len(self.file_list):
            raise StopIteration
        self.counter += 1
        return self.file_list[self.counter]


class Patients():

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.patient_list = root_dir
        self.counter = -1

    @property
    def patient_list(self):
        return self._patient_list

    @patient_list.setter
    def patient_list(self, root_dir: str):
        self._patient_list = []
        for x in list(os.walk(root_dir))[1:]:
            self._patient_list.append(Patient(x[0]))

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter + 1 == len(self.patient_list):
            raise StopIteration
        self.counter += 1
        return self.patient_list[self.counter]


class PatientTusz(Patient):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, root_dir: str):
        self._file_list = glob.glob(os.path.join(root_dir, "**/*.edf"))

    @property
    def patient_name(self):
        return self._patient_name

    @patient_name.setter
    def patient_name(self, root_dir: str):
        self._patient_name = os.path.split(root_dir)[-1]


class PatientsTusz(Patients):

    def __init__(self, root_dir: str):
        super().__init__(root_dir)

    @property
    def patient_list(self):
        return self._patient_list

    @patient_list.setter
    def patient_list(self, root_dir: str):
        self._patient_list = []
        patients = [x.split("/")[-1]
                    for x in sorted(glob.glob(os.path.join(root_dir, "**/**/**/**")))]
        for x in patients:
            directory = f"{root_dir}**/**/**/{x}"
            self._patient_list.append(PatientTusz(directory))


class Metadata():

    def __init__(self, file: str):
        self.seizure_ranges = file

    @property
    def seizure_ranges(self):
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, file: str):
        self._seizure_ranges = {}
        clean_lines, single_file = [], []

        with open(file) as fp:
            lines = fp.readlines()
        lines = [re.sub("\s+", " ", x.lower()) for x in lines]

        skip = True
        clean_lines = []
        for line in lines:
            if "file name" in line:
                skip = False
            if "changed" in line:
                skip = True
            if not skip:
                clean_lines.append(line)

        clean_lines, text_lines = [], clean_lines
        for line in text_lines:
            if line == " ":
                clean_lines.append(single_file)
                single_file = []
            elif not any(x in line for x in ["file start", "file end"]):
                single_file.append(line)
        if len(single_file):
            clean_lines.append(single_file)
        for file in clean_lines:
            filename, seizures = self.get_seizure_single_file(file)
            self._seizure_ranges.update({filename: seizures})

    def get_seizure_single_file(self, text_lines: list) -> tuple:
        seizures = [(0.0, 0.0)]
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


class MetadataList():

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.patient_metadata = root_dir

    def get(self, patient: str, file: str):
        file = file.split("/")[-1]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self):
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str):
        self._patient_metadata = {}
        for x in glob.glob(os.path.join(root_dir, "*.txt")):
            patient = x.split("/")[-1].split("-")[0]
            self._patient_metadata.update({patient: Metadata(x)})


class MetadataSiena():

    def __init__(self, file: str):
        self.seizure_ranges = file

    @property
    def seizure_ranges(self):
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, file: str):
        self._seizure_ranges = {}

        with open(file) as fp:
            lines = fp.readlines()

        for index, line in enumerate(lines):
            if "File name" in line:
                filename = line.split(" ")[-1][:-1]
                seizure = self.get_seizure_single_file(lines[index+1:
                                                             index+5])
                if filename not in self._seizure_ranges:
                    self._seizure_ranges[filename] = [(0.0, 0.0)]
                self._seizure_ranges[filename].append(seizure)

    def get_seizure_single_file(self, text_lines: list) -> tuple:
        assert "Registration start" in text_lines[0]
        recording_start_time = re.search("(?<=:\s).*", text_lines[0]).group(0)
        seizure_start_time = re.search("(?<=:\s).*", text_lines[2]).group(0)
        seizure_end_time = re.search("(?<=:\s).*", text_lines[3]).group(0)
        recording_start_time = datetime.strptime(recording_start_time.strip(),
                                                 "%H.%M.%S")
        seizure_start_time = datetime.strptime(seizure_start_time.strip(),
                                               "%H.%M.%S")
        seizure_end_time = datetime.strptime(seizure_end_time.strip(),
                                             "%H.%M.%S")

        if recording_start_time > seizure_start_time:
            recording_start_time = recording_start_time - timedelta(days=1)

        start_seconds = (seizure_start_time - recording_start_time).seconds
        end_seconds = (seizure_end_time - recording_start_time).seconds
        return (start_seconds, end_seconds)


class MetadataTusz():

    def __init__(self, files: str):
        self.seizure_ranges = files

    @property
    def seizure_ranges(self):
        return self._seizure_ranges

    @seizure_ranges.setter
    def seizure_ranges(self, files: list):
        self._seizure_ranges = {}

        for file in files:
            with open(file) as fp:
                lines = fp.readlines()
            filename = file.split("/")[-1].rstrip(".tse")

            for _, line in enumerate(lines):
                if filename not in self._seizure_ranges:
                    self._seizure_ranges[filename] = [(0.0, 0.0, "bckg")]
                if "z" in line:
                    seizure = self.get_seizure_single_file(line)
                    self._seizure_ranges[filename].append(seizure)

    def get_seizure_single_file(self, text_line: str) -> tuple:
        parts = text_line.split(" ")
        return (float(parts[0]), float(parts[1]), parts[2])


class MetadataListTusz():

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.patient_metadata = root_dir

    def get(self, patient: str, file: str):
        file = file.split("/")[-1].split(".")[0]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self):
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str):
        _grouped_files = defaultdict(list)
        self._patient_metadata = {}
        for x in glob.glob(os.path.join(root_dir, "**/**/**/**/**/*.tse")):
            patient = x.split("/")[-3]
            _grouped_files[patient].append(x)

        for patient, files in _grouped_files.items():
            self._patient_metadata.update({patient: MetadataTusz(files)})


class MetadataListSiena():

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.patient_metadata = root_dir

    def get(self, patient: str, file: str):
        file = file.split("/")[-1]
        patient_files = self._patient_metadata.get(patient)
        single_metadata = patient_files.seizure_ranges.get(file)
        return single_metadata

    @property
    def patient_metadata(self):
        return self._patient_metadata

    @patient_metadata.setter
    def patient_metadata(self, root_dir: str):
        self._patient_metadata = {}
        for x in glob.glob(os.path.join(root_dir, "**/*.txt")):
            patient = x.split("/")[-1].split("-")[-1].rstrip(".txt")
            self._patient_metadata.update({patient: MetadataSiena(x)})


def save_numpy_as_csv(data: list, filename: str):
    with open(filename, "w") as stream:
        output = csv.writer(stream)
        output.writerows(data)


def load_csv_numpy(filename: str, num_channels: int):
    logging.info(filename)
    array = []
    with open(filename, "r") as stream:
        reader = csv.reader(stream)
        for row in reader:
            length = int(len(row)/num_channels)
            row = np.array(row).reshape(num_channels, length)
            if length % 2 == 0:
                row = np.append(row, row[:, -2:-1], axis=1)
            array.append(row)
    array = np.stack(array, axis=0)
    return array.astype(np.float32, copy=False)
