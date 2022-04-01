#!/bin/bash
source venv/bin/activate
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

/lustre/home/ssanchez/python-core_375/bin/python3 train_ml_model_crossseizure_wang.py --patient=chb01 --db=chb-mit --model=wang_1d
deactivate