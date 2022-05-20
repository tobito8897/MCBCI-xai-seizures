#!/usr/bin/python3.7
"""
Usage:
    train_ml_model_crosspatient.py --patient=<p> --db=<d>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database to use for training
"""
import sys
import logging
import numpy as np
import tensorflow as tf
from collections import defaultdict
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.mlutils import *
from utils.files import *
from utils.models import *

np.random.seed(1)
tf.random.set_seed(1)

OPTS = docopt(__doc__)
database = settings[OPTS["--db"]]
patient = OPTS["--patient"]
settings = settings["hossain"]


def main():
    logging.info("Patient selected for evaluation: %s", patient)

    ictal_data, noictal_data = defaultdict(list), defaultdict(list)
    data_files = [x for x in list_files(database["windows"] + "/train", "*.csv")
                  if patient not in x and "_ictal" in x]
    labels = [(0, 1) for _ in data_files]
    for x, y in zip(labels, data_files):
        for z in database["patients"]:
            if z in y:
                ictal_data[z].append((x, y))
                break

    data_files = [x for x in list_files(database["windows"] + "/train", "*.csv")
                  if patient not in x and "_ictal" not in x]
    labels = [(1, 0) for _ in data_files]
    for x, y in zip(labels, data_files):
        for z in database["patients"]:
            if z in y:
                noictal_data[z].append((x, y))
                break

    data_files = [x for x in list_files(database["windows"] + "/test", "*.csv")
                  if patient in x]
    labels = [(0, 1) if "_ictal" in x else (1, 0) for x in data_files]
    testing_files = dict(zip(data_files, labels))

    history_file = "{}/history_{}_{}.pickle".format(settings["stats"],
                                                    OPTS["--db"], patient)
    stats_file = "{}/stats_{}_{}.pickle".format(settings["stats"],
                                                OPTS["--db"], patient)
    model_file = "{}/model_patient_{}_{}.h5".format(settings["models"],
                                                    OPTS["--db"], patient)

    model = Net_Hossain((database["num_channels"],
                         database["downsampled_length"], 1),
                        database["num_channels"])

    history_data = {"acc": [],
                    "val_acc": [],
                    "loss": [],
                    "val_loss": []}

    train_gen = CustomDataGen(ictal_data, noictal_data,
                              database["num_channels"],
                              database["patients"], patient,
                              100)
    val_gen = CustomDataGen(ictal_data, noictal_data,
                            database["num_channels"],
                            database["patients"], patient,
                            20)

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=settings["epochs"], verbose=1,
                        shuffle=True, use_multiprocessing=True,
                        workers=3)
    history_data["acc"] = history.history["accuracy"]
    history_data["val_acc"] = history.history["val_accuracy"]
    history_data["loss"] = history.history["loss"]
    history_data["val_loss"] = history.history["val_loss"]
    model.save(model_file)

    logging.info("Starting predictions")
    x_test, y_test = prepare_dataset(testing_files,
                                     database["num_channels"], shuffle=False)
    results = model.predict(x_test)
    y_real = y_test.argmax(axis=1).tolist()
    y_predicted = results.argmax(axis=1).tolist()

    write_pickle(history_file, history_data)

    stats_data = {"real": y_real,
                  "predicted": y_predicted}
    write_pickle(stats_file, stats_data)


if __name__ == "__main__":
    main()
