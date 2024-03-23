#!/bin/bash
python ../bin/estimate_correlation_xai_ml.py --db=chb-mit --model=wang_1d --xai=shap
python ../bin/estimate_correlation_xai_ml.py --db=chb-mit --model=wang_1d --xai=lime

python ../bin/estimate_correlation_xai_ml.py --db=siena --model=wang_1d --xai=shap
python ../bin/estimate_correlation_xai_ml.py --db=siena --model=wang_1d --xai=lime

python ../bin/estimate_correlation_xai_ml.py --db=tusz --model=wang_1d --xai=shap
python ../bin/estimate_correlation_xai_ml.py --db=tusz --model=wang_1d --xai=lime