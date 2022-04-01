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
    data_files = list_files(database["windows"] + "/train", "*.csv")
    training_files = {}

    for file in data_files:
        if patient not in file:
            continue
        label = (1, 0) if "_noictal" in file else (0, 1)
        training_files.update({file: label})

    history_file = "{}/history_full_{}_{}.pickle".format(settings["stats"],
                                                         OPTS["--db"],
                                                         patient)
    model_file = "{}/model_full_{}_{}.h5".format(settings["models"],
                                                 OPTS["--db"],
                                                 patient)

    model = Net_Wang_1d((settings["num_channels"], settings["length"]))

    x_train, y_train = prepare_dataset(training_files,
                                       database["num_channels"])

    history = model.fit(x_train, y_train, epochs=settings["epochs"], verbose=1,
                        validation_split=0.2, batch_size=100,
                        callbacks=[early_stop_wang()])
    model.save(model_file)

    history_data = {"acc": history.history["accuracy"],
                    "val_acc": history.history["val_accuracy"],
                    "loss": history.history["loss"],
                    "val_loss": history.history["val_loss"]}
    write_pickle(history_file, history_data)


if __name__ == "__main__":
    main()
