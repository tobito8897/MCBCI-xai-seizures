#!/usr/bin/python3.7
"""
Usage:
    generate_explanations_shap.py --db=<d> --model=<m> --overlap=<o>

Options:
    --model=<p>       Model to be used: wang_1d, wang_2d
    --db=<d>          Database to use for training
    --overlap=<m>     Overlap applied to the training dataset, 0.7, 0.8, 0.9
"""
import sys
import logging
import keras
import shap
import numpy as np
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.mlfiles import Patients
from utils.mlfiles import StandardScaler
from utils.mlutils import prepare_dataset
from utils.eeg_masker import EegMask


OPTS = docopt(__doc__)
DBCONF = settings[OPTS["--db"]]


class Prediction():

    def __init__(self, model):
        self.model = model
    
    def __call__(self, input):
        print(input.shape)
        return self.model.predict(input[:, :, :, 0:1])


def main():
    patients = Patients(DBCONF["windows_dir"])
    for patient in patients:
        logging.info("Selected patient: %s", patient.patient_name)
        patient.compute_kfolds()

        if not hasattr(patient, "kfolds"):
            logging.warning("Not enough sessions")
            continue

        if not len(patient):
            logging.warning("Not enough sessions")
            continue

        logging.info("Number of folds: %s", len(patient))
        num_channels = len(DBCONF["bipolar_channels"])

        for (training_files, testing_files) in iter(patient):
            testing_files = {k: v for k, v in testing_files.items()
                             if "_ictal" in k}
            x_train, _ = prepare_dataset(training_files,
                                         num_channels,
                                         shuffle=False)
            x_test, y_test = prepare_dataset(testing_files,
                                             num_channels,
                                             shuffle=False)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            logging.info("Initial number of instances: %s", len(y_test))

            file_list = list(testing_files.keys())
            model_name = "_".join(file_list[0].split("/")[-1].split("_")[:-1])
            model_file = f'{DBCONF["models_dir"]}{OPTS["--model"]}/{model_name}_' \
                         f'{OPTS["--overlap"]}.h5'
            model = keras.models.load_model(model_file)

            masker_blur = EegMask("gaussian_noise", x_test[0, :, :, :].shape)
            logging.info("Number of instances to evaluate: %s", y_test.shape[0])

            y_real = y_test.argmax(axis=1)
            results = model.predict(x_test)
            y_predicted = results.argmax(axis=1)
            mask = y_real == y_predicted
            x_test = x_test[mask, :, :, :]
            y_test = y_test[mask, :]
            logging.info("Number of instances after removing the wrong predictions: %s", len(y_test))
            if len(y_test) == 0:
                continue

            ictal_importances, ictal_eeg = [], []

            for x in range(y_test.shape[0]):
                explainer_blur = shap.Explainer(model.predict, masker_blur, algorithm="partition")
                # max_evals = 2   + 2   + 8   +  32     + 128   + 512
                # tile_size = 100 + 50% + 25% +  12.5%  + 6.25% + 3.125%
                # batch_size should be equal/largest than 512
                shap_values = explainer_blur(x_test[x:x+1, :, :, :],
                                             max_evals=684, batch_size=600,
                                             outputs=shap.Explanation.argsort.flip[:4])

                if y_test[x, :].tolist() == [0, 1]:
                    ictal_importances.append(shap_values.values[0, :, :, 0, 0])
                    ictal_eeg.append(x_test[x, :, :, 0])
                else:
                    raise Exception
                logging.info("Instance explained, progress %s/%s",
                             x, y_test.shape[0])

            ictal_importances = np.stack(ictal_importances)
            filename = f"{DBCONF['exp_dir']}shap/" \
                       f"{model_name}_shap.npy"
            with open(filename, 'wb') as f:
                np.save(f, ictal_importances)

            ictal_eeg = np.stack(ictal_eeg)
            filename = f"{DBCONF['exp_dir']}shap/" \
                       f"{model_name}_eeg.npy"
            with open(filename, 'wb') as f:
                np.save(f, ictal_eeg)


if __name__ == "__main__":
    main()
