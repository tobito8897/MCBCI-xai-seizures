import os
import glob
import numpy as np
from .files import load_csv_numpy


class Patient():

    def __init__(self, patient: str, root_dir: str):
        self.patient_name = patient
        self.root_dir = root_dir
        self.file_list = root_dir
        self.counter = -1

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, root_dir: str):
        self._file_list = [x for x in glob.glob(os.path.join(root_dir, "train/*.csv"))
                           if x.split("/")[-1].startswith(f"{self.patient_name}_")]

    @property
    def patient_name(self):
        return self._patient_name

    @patient_name.setter
    def patient_name(self, root_dir: str):
        self._patient_name = os.path.split(root_dir)[-1]

    def compute_kfolds(self):
        sessions = set()
        for x in self._file_list:
            session_id = "_".join(x.split("_")[1:-1])
            sessions.add(session_id)
        sessions = list(sessions)
        sessions.sort()

        if len(sessions) < 2:
            return

        labels = [(0, 1) if "_ictal" in x else (1, 0) for x in self._file_list]
        training_files = dict(zip(self._file_list, labels))

        data_files = [x.replace("train", "test") for x in self._file_list]
        testing_files = dict(zip(data_files, labels))

        self.kfolds = []
        for x in sessions:
            _train = {k: v for k, v in training_files.items()
                      if f"_{x}_" not in k}
            _test = {k: v for k, v in testing_files.items()
                     if f"_{x}_" in k}
            self.kfolds.append((_train, _test))

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.kfolds)

    def __next__(self):
        if self.counter + 1 == len(self.kfolds):
            raise StopIteration
        self.counter += 1
        return self.kfolds[self.counter][0], self.kfolds[self.counter][1]


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
        files = glob.glob(f"{root_dir}train/*.csv")

        patients = set()
        for file in files:
            basename = os.path.basename(file)
            patient = basename.split("_")[0]
            patients.add(patient)

        self._patient_list = []
        for x in patients:
            self._patient_list.append(Patient(x, root_dir))

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter + 1 == len(self.patient_list):
            raise StopIteration
        self.counter += 1
        return self.patient_list[self.counter]


class StandardScaler():

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.array):
        self.mean = np.mean(np.mean(data, axis=0), axis=1)
        self.std = np.std(np.std(data, axis=0), axis=1)

    def transform(self, data: np.array):
        for idx in range(data.shape[1]):
            data[:, idx, :, :] = (data[:, idx, :, :] - self.mean[idx, 0])/self.std[idx, 0]
        return data

    def fit_transform(self, data: np.array) -> np.array:
        self.fit(data)
        data = self.transform(data)
        return data


def shuffle_dataset(instances: int, labels: int) -> tuple:
    new_order = np.random.permutation(len(instances))
    instances = instances[new_order, :, :, :]
    labels = labels[new_order, :]
    return instances, labels


def prepare_dataset(files_map: dict, num_channels: int,
                    shuffle: bool=True) -> tuple:
    x_train = []
    y_train = []
    for file, label in files_map.items():
        train_set = load_csv_numpy(file, num_channels)
        labels = create_labels(label, train_set.shape[0])
        x_train.append(train_set)
        y_train.append(labels)
    x_train = np.concatenate(x_train, axis=0)
    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.concatenate(y_train, axis=0)
    x_train, y_train = shuffle_dataset(x_train, y_train)
    return x_train, y_train


def create_labels(label: int, length: int) -> np.array:
    return np.array([label]*length, dtype=np.int32)
