#!/usr/bin/python3.7
"""
Usage:
    reevaluate_model_crossseizure.py --model=<p> --db=<d> --overlap=<o>

Options:
    --model=<p>       Model to be used: wang_1d, wang_2d
    --db=<d>          Database to use for training
    --overlap=<m>     Overlap applied to the training dataset, 0.7, 0.8, 0.9
"""
import sys
import logging
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.metrics import performance_metrics
from utils.mlfiles import StandardScaler, Patients, prepare_dataset

np.random.seed(1)
tf.random.set_seed(1)

OPTS = docopt(__doc__)
DBCONF = settings[OPTS["--db"]]
MLCONF = settings[OPTS["--model"]]


def main():
    metrics_file = f'{DBCONF["metrics_dir"]}perf_{OPTS["--db"]}_' \
                   f'{OPTS["--model"]}_{OPTS["--overlap"]}_reevaluated.csv'
    metrics = []

    patients = Patients(DBCONF["windows_dir"])
    for patient in patients:
        logging.info("Selected patient: %s", patient.patient_name)
        patient.compute_kfolds()

        try:
            if not len(patient):
                logging.warning("Not enough sessions")
                continue
        except:
            logging.warning("Not enough sessions")
            continue

        logging.info("Number of folds: %s", len(patient))
        num_channels = len(DBCONF["bipolar_channels"])

        for (training_files, testing_files) in iter(patient):
            file_list = list(testing_files.keys())
            seizure_type = file_list[0].split("_")[-2]
            model_name = "_".join(file_list[0].split("/")[-1].split("_")[:-1])
            model_file = f'{DBCONF["models_dir"]}{OPTS["--model"]}/{model_name}_' \
                         f'{OPTS["--overlap"]}.h5'

            x_train, y_train = prepare_dataset(training_files,
                                             num_channels)
            x_test, y_test = prepare_dataset(testing_files,
                                             num_channels)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            logging.info("Number of instances for training: %s", x_train.shape[0])
            logging.info("Number of instances for testing: %s", x_test.shape[0])

            file_list = list(testing_files.keys())
            model_name = "_".join(file_list[0].split("/")[-1].split("_")[:-1])
            model_file = f'{DBCONF["models_dir"]}{OPTS["--model"]}/{model_name}_' \
                         f'{OPTS["--overlap"]}.h5'
            model = keras.models.load_model(model_file)

            y_predicted = model.predict(x_test)
            f1_score, sen, spec, acc = performance_metrics(y_test, y_predicted)
            logging.info("Metrics, f1=%s, sen=%s, spec=%s", f1_score, sen, spec)
            metrics.append({"patient": patient.patient_name,
                            "seizure_type": seizure_type,
                            "f1_score": f1_score,
                            "sensitivity": sen,
                            "specificity": spec,
                            "accuracy": acc})

        dataframe = pd.DataFrame.from_dict(metrics)
        dataframe.to_csv(metrics_file, index=False)


if __name__ == "__main__":
    main()
