#!/bin/bash
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

python3 generate_train_windows_chb.py --patient=chb01 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb02 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb03 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb04 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb05 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb06 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb07 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb08 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb09 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb10 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb11 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb12 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb13 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb14 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb15 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb16 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb17 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb18 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb19 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb20 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb21 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb22 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb23 --db=chb-mit_wang
python3 generate_train_windows_chb.py --patient=chb24 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb01 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb02 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb03 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb04 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb05 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb06 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb07 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb08 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb09 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb10 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb11 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb12 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb13 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb14 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb15 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb16 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb17 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb18 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb19 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb20 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb21 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb22 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb23 --db=chb-mit_wang
python3 generate_test_windows_chb.py --patient=chb24 --db=chb-mit_wang