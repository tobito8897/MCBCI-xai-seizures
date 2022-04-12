#!/usr/bin/python3.7
"""
Usage:
    train_ml_model.py --patient=<p> --db=<d> --model=<m>

Options:
    --patient=<p>     Patient used for evaluation
    --db=<p>          Database to use for training
    --model=<m>       Wang 1d or 2d
"""
import sys
import logging
import numpy as np
import tensorflow as tf
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
settings = settings[OPTS["--model"]]


def main():
    patient = OPTS["--patient"]
    logging.info("Patient selected for evaluation: %s", patient)
    data_files = [x for x in list_files(database["windows"] + "/train", "*.csv")
                  if patient in x]

    sessions = {x.split("_")[2] for x in data_files}
    labels = [(0, 1) if "_ictal" in x else (1, 0) for x in data_files]
    training_files = dict(zip(data_files, labels))

    data_files = [x.replace("train", "test") for x in data_files]
    testing_files = dict(zip(data_files, labels))

    k_folds = []
    for x in sessions:
        _train = {k: v for k, v in training_files.items()
                  if f"_{x}_" not in k}
        _test = {k: v for k, v in testing_files.items()
                 if f"_{x}_" in k}
        k_folds.append((_train, _test))

    stats_file = "{}/stats_{}_{}.pickle".format(settings["stats"],
                                                OPTS["--db"],
                                                patient)
    global_y_real, global_y_predicted = [], []

    for fold, (training_files, testing_files) in enumerate(k_folds):
        logging.info("Fold number: %s", fold + 1)

        history_file = "{}/history_{}_{}_{}.pickle".format(settings["stats"],
                                                           OPTS["--db"],
                                                           patient, fold)
        model_file = ("{}/model_{}_{}_{}.h5".format(settings["models"],
                                                    OPTS["--db"],
                                                    patient, fold))
        x_train, y_train = prepare_dataset(training_files,
                                           database["num_channels"])

        if OPTS["--model"] == "wang_1d":
            model = Net_Wang_1d((database["num_channels"], database["length"]))
        elif OPTS["--model"] == "wang_2d":
            model = Net_Wang_2d((database["num_channels"], database["length"], 1))
            x_train = x_train[..., np.newaxis]

        history = model.fit(x_train, y_train, epochs=settings["epochs"],
                            verbose=1, validation_split=0.2,
                            batch_size=100, callbacks=[early_stop_wang()])
        model.save(model_file)

        logging.info("Starting predictions")
        x_test, y_test = prepare_dataset(testing_files,
                                         database["num_channels"])
        if OPTS["--model"] == "wang_2d":
            x_test = x_test[..., np.newaxis]

        results = model.predict(x_test)
        global_y_real += y_test.argmax(axis=1).tolist()
        global_y_predicted += results.argmax(axis=1).tolist()

        history_data = {"acc": history.history["accuracy"],
                        "val_acc": history.history["val_accuracy"],
                        "loss": history.history["loss"],
                        "val_loss": history.history["val_loss"]}
        write_pickle(history_file, history_data)

        tf.keras.backend.clear_session()
        del model

    stats_data = {"real": global_y_real,
                  "predicted": global_y_predicted}
    write_pickle(stats_file, stats_data)


if __name__ == "__main__":
    main()
