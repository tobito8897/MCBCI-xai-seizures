#!/usr/bin/python3.7
import numpy as np
import tensorflow as tf
from .files import load_csv_numpy

__all__ = ["prepare_dataset"]


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
