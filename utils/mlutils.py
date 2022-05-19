#!/usr/bin/python3.7
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from .files import load_csv_numpy
import time

__all__ = ["prepare_dataset", "CustomDataGen"]


def create_labels(label: int, length: int) -> np.array:
    return np.array([label]*length, dtype=np.int32)


def shuffle_dataset(instances: int, labels: int) -> tuple:
    new_order = np.random.permutation(len(instances))
    instances = instances[new_order, :, :, :]
    labels = labels[new_order, :]
    return instances, labels


def prepare_dataset(files_map: dict, num_channels: int,
                    shuffle: bool = True) -> tuple:
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
    if shuffle:
        x_train, y_train = shuffle_dataset(x_train, y_train)
    return x_train, y_train


def prepare_dataset_from_list(intances: list, labels: list,
                              num_channels: int) -> tuple:
    x_train = []
    y_train = []
    for x, y in zip(intances, labels):
        length = int(len(x)/num_channels)
        train_set = np.array(x).reshape(num_channels, length)
        label = create_labels(y, 1)
        x_train.append(train_set)
        y_train.append(label)
    x_train = np.stack(x_train, axis=0)
    x_train = np.expand_dims(x_train, axis=3)
    y_train = np.concatenate(y_train, axis=0)
    return x_train.astype(np.float32, copy=False), y_train


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, ictal_files_map: dict, no_ictal_files_map: dict,
                 num_channels: int, patients: list, current_patient: str,
                 batch_size: int):

        self.ictal_files_map = ictal_files_map
        self.no_ictal_files_map = no_ictal_files_map
        self.files_length = {}
        self.instances_number = 0
        self.patients = patients
        self.current_patient = current_patient
        self.batch_size = batch_size
        self.num_channels = num_channels
        for _, files in self.ictal_files_map.items():
            for file in files:
                with open(file[1]) as fp:
                    length = [x for x, _ in enumerate(fp)]
                self.files_length[file[1]] = length
                self.instances_number += len(length)
        for _, files in self.no_ictal_files_map.items():
            for file in files:
                with open(file[1]) as fp:
                    length = [x for x, _ in enumerate(fp)]
                self.files_length[file[1]] = length
                self.instances_number += len(length)

    def on_epoch_end(self):
        pass

    def __get_data(self, selected_files):
        instances, labels = [], []
        for _ in range(self.batch_size):
            file, label = random.choice(selected_files)
            index = random.choice(self.files_length[file])
            with open(file) as fp:
                for i, line in enumerate(fp):
                    if i == index:
                        instances.append(line.replace("\n", "").split(","))
                        labels.append(label)
        x, y = prepare_dataset_from_list(instances, labels,
                                         self.num_channels)
        return x, y

    def __getitem__(self, index):
        #print(time.time())
        dataset = []
        for x in self.patients:
            if x == self.current_patient:
                continue
            y = random.choice(self.no_ictal_files_map[x])
            dataset.append([y[1], y[0]])
            y = random.choice(self.ictal_files_map[x])
            dataset.append([y[1], y[0]])
        X, y = self.__get_data(dataset)
        #print(time.time())
        return X, y

    def __len__(self):
        return self.instances_number//self.batch_size
