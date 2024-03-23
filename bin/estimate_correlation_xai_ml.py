#!/usr/bin/python3.7
"""
Usage:
    estimate_correlation_xai_ml.py --db=<d> --model=<m> --xai=<x>

Options:
    --model=<p>       Model to be used: wang_1d, wang_2d
    --db=<d>          Database to use for training
    --xai=<x>         XAI algorithm: lime, shap
"""
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from scipy import stats
from docopt import docopt
sys.path.append("../")

from utils import settings
from utils.mlfiles import Patients
from utils.signal_processors import SignalFeatures, filter_butterworth_2


OPTS = docopt(__doc__)
DBCONF = settings[OPTS["--db"]]
FEATURES = ["complexity", "mobility", "interquartile_range", "absolute_median_deviation",
            "peak_frequency", "median_frequency", "rms", "skewness", "kurtosis",
            "zerocrossing", "sampleentropy", "range_val", "mean", "sdeviation"]
np.random.seed(0)


filt_bank = {"low": {"f": 12, "order": 2, "fs": 256, "type": "lowpass"},               # Delta, theta, alpha band
             "beta": {"f": (12, 25), "order": 2, "fs": 256, "type": "bandpass"},       # Beta band
             "gamma": {"f": 25, "order": 2, "fs": 256, "type": "highpass"}}      # Gamma band


def main():
    patients = Patients(DBCONF["windows_dir"])
    for patient in patients:
        logging.info("Selected patient: %s", patient.patient_name)
        if patient.patient_name not in ["6904", "6563"]: # ----------- 
            continue

        root_dir = f"{DBCONF['exp_dir']}{OPTS['--xai']}"
        name_pattern = f"*{patient.patient_name}*eeg*"
        eeg_files = sorted(glob.glob(os.path.join(root_dir, name_pattern)))
        name_pattern = f"*{patient.patient_name}*{OPTS['--xai']}*"
        explanation_files = sorted(glob.glob(os.path.join(root_dir, name_pattern)))

        if not len(eeg_files):
            logging.warning("Not enough sessions")
            continue

        logging.info("Number of files: %s", len(eeg_files))
        ictal_eeg_timeseries = []
        ictal_eeg_importances = []

        for (eeg_file, explanation_file) in zip(eeg_files, explanation_files):
            logging.info("Opening file: %s", eeg_file)
            eeg_data = np.load(eeg_file)
            logging.info("Opening file: %s", explanation_file)
            explanation_data = np.load(explanation_file)
    
            for x in range(eeg_data.shape[0]):
                for y in range(eeg_data.shape[1]):
                    importance_sum = np.sum(explanation_data[x, y, :])
                    ictal_eeg_importances.append(importance_sum)
                    ictal_eeg_timeseries.append(eeg_data[x, y, :])

        logging.info("Number of instances: %s", len(ictal_eeg_timeseries))
        ictal_eeg_timeseries = np.vstack(ictal_eeg_timeseries)
        ictal_eeg_importances = np.mean(np.vstack(ictal_eeg_importances),
                                        axis=1)

        features_stimator = SignalFeatures(int(ictal_eeg_timeseries.shape[1]/DBCONF["epoch_size"]))

        try:
            metrics = pd.read_csv(f"{DBCONF['metrics_dir']}corr_" \
                                  f"{OPTS['--db']}_{OPTS['--model']}_{OPTS['--xai']}_second_round.csv")
            metrics = metrics.to_dict(orient="records")
        except:
            metrics = []

        # #################SIGNAL SEPARATION BY BAND####################
        band_separated_ictal_eeg_timeseries = {"full": ictal_eeg_timeseries}
        for band, details in filt_bank.items():
            logging.info(f"Filtering by band, name={band}")
            _ictal_eeg_timeseries = filter_butterworth_2(details["f"],
                                                         details["order"],
                                                         details["type"],
                                                         details["fs"],
                                                         ictal_eeg_timeseries)
            band_separated_ictal_eeg_timeseries[band] = _ictal_eeg_timeseries

        # #################CHARTS####################
        for band, ictal_eeg_timeseries in band_separated_ictal_eeg_timeseries.items():
            for feature in FEATURES:
                logging.info(f"Processing the class=ictal, feature={feature}, band={band}")
                eeg_timeseries_param = features_stimator(feature, ictal_eeg_timeseries)

                try:
                    importances = ictal_eeg_importances.tolist()
                    features = eeg_timeseries_param.tolist()
                    importance_rank = [sorted(importances, reverse=True).index(x) + 1 for x in importances]
                    features_rank = [sorted(features, reverse=True).index(x) + 1 for x in features]
                    spearman_result = stats.spearmanr(features_rank, importance_rank)
                    logging.info(f"Ictal spearman correlation: {spearman_result}")
                except:
                    spearman_result = [None]
                metrics.append({"subject": patient.patient_name,
                                "class": "ictal",
                                "band": band,
                                "feature": feature,
                                "coefficient": spearman_result[0],
                                "pvalue": str(spearman_result[1])})
        metrics = pd.DataFrame(metrics)
        print(metrics)
        metrics.to_csv(f"{DBCONF['metrics_dir']}corr_" \
                       f"{OPTS['--db']}_{OPTS['--model']}_{OPTS['--xai']}_second_round.csv",
                       index=False)


if __name__ == "__main__":
    main()
