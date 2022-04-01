#!/usr/bin/python3.7
"""
Usage:
    train_ml_model_crosspatient.py --patient=<p>

Options:
    --patient=<p>     Patient used for evaluation
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

OPTS = docopt(__doc__)


def main():
    patient = OPTS["--patient"]
    logging.info("Patient selected for evaluation: %s", patient)
    data_files = [x for x in list_files(settings["windows_dir"] + "/train", "*.csv")
                  if patient not in x]
    noictal_files = [(x, (1, 0)) for x in data_files if "_noictal" in x]
    ictal_files = [(x, (0, 1)) for x in data_files if "_ictal" in x]

    data_files = [x for x in list_files(settings["windows_dir"] + "/test", "*.csv")
                  if patient in x]
    labels = [(0, 1) if "_ictal" in x else (1, 0) for x in data_files]
    testing_files = dict(zip(data_files, labels))

    history_file = "{}/history_{}.pickle".format(settings["stats_dir"],
                                                 patient)
    stats_file = "{}/stats_{}.pickle".format(settings["stats_dir"],
                                             patient)
    model_file = "{}/model_patient_{}.h5".format(settings["models_dir"],
                                                 patient)

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
        model = Net_Hossain((settings["num_channels"], settings["length"], 1))

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

    logging.info("Starting predictions")
    x_test, y_test = prepare_dataset(testing_files,
                                     database["num_channels"])
    results = model.predict(x_test)
    y_real = y_test.argmax(axis=1).tolist()
    y_predicted = results.argmax(axis=1).tolist()

    stats_data = {"real": y_real,
                  "predicted": y_predicted}
    history_data = {"acc": training_acc,
                    "val_acc": validation_acc}
    write_pickle(stats_file, stats_data)
    write_pickle(history_file, history_data)


if __name__ == "__main__":
    main()
