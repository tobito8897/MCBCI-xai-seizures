#!/usr/bin/python3.7
"""
Usage:
    train_ml_model_full.py
"""
import sys
import random
import logging
import keras
import numpy as np
import tensorflow as tf
from docopt import docopt
sys.path.append("../")

from src import settings
from src.mlutils import *
from src.files import *
from src.models import *


np.random.seed(1)
tf.random.set_seed(1)
settings = settings["wang_1d"]
database = settings["chb-mit"]


def main():
    data_files = list_files(database["windows"] + "/train", "*.csv")
    noictal_files = [(x, (1, 0)) for x in data_files if "_noictal" in x]
    ictal_files = [(x, (0, 1)) for x in data_files if "_ictal" in x]

    history_file = "{}/history_full.pickle".format(settings["stats"])
    model_file = "{}/model_patient_full.h5".format(settings["models"])

    try:
        x = read_pickle(history_file)
        training_acc, validation_acc = x["acc"], x["val_acc"]
        logging.info("Model history retrieved")
    except Exception as e:
        logging.error(e)
        training_acc, validation_acc = [], []

    try:
        model = keras.models.load_model(model_file)
        logging.info("Model loaded from file")
    except Exception as e:
        logging.error(e)
        model = Net_Hossain((database["num_channels"], settings["length"], 1))

    for _ in range(settings["epochs"]):
        logging.info("Epoch number: %s", _)

        training_dataset = {}
        for _ in range(3):
            x = random.choice(ictal_files)
            y = random.choice(noictal_files)
            training_dataset.update({x[0]: x[1], y[0]: y[1]})
        x_train, y_train = prepare_dataset(training_dataset,
                                           database["num_channels"])

        history = model.fit(x_train, y_train, epochs=1, verbose=1,
                            validation_split=0.2, steps_per_epoch=100)
        training_acc += history.history["accuracy"]
        validation_acc += history.history["val_accuracy"]

    model.save(model_file)

    history_data = {"acc": training_acc,
                    "val_acc": validation_acc}
    write_pickle(history_file, history_data)

    tf.keras.backend.clear_session()
    del model


if __name__ == "__main__":
    main()
