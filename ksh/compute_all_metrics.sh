#!/bin/bash
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=chb-mit --overlap=0.5
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=chb-mit --overlap=0.7
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=chb-mit --overlap=0.8

python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=siena --overlap=0.5
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=siena --overlap=0.7
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=siena --overlap=0.8

python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=tusz --overlap=0.5
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=tusz --overlap=0.7
python ../bin/reevaluate_model_crossseizure.py --model=wang_1d --db=tusz --overlap=0.8