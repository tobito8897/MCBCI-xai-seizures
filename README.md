# epilepsy_xai

### Pre-requisite
* python -m pip install -r requirements
* Run create_tree_dir.sh
* Run download_siena.sh
* Run download_edf_chb-mit.py

### Train WANG model
* Run generate_test_windows_chb.py (for each patient)
* Run generate_train_windows_chb.py (for each patient)
* Run generate_test_windows_siena.py (for each patient)
* Run generate_train_windows_siena.py (for each patient)
* Run train_ml_model_crossseizure_wang.py (for each patient, db and model (1d or 2d))


### Train HOSSAIN model
* Run generate_test_windows_chb.py (for each patient)
* Run generate_train_windows_chb.py (for each patient)
* Run generate_test_windows_siena.py (for each patient)
* Run generate_train_windows_siena.py (for each patient)
* Run randomize_windows_hossain (for each patient and db)
* Run train_ml_model_crosspatient_hossain.py (for each patient and db)