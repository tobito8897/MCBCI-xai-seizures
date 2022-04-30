#!/bin/bash
source ../venv/bin/activate
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb01 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb02 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb03 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb04 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb05 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb06 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb07 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb08 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb09 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb10 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb11 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb12 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb13 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb14 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb15 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb16 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb17 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb18 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb19 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb20 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb21 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb22 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb23 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_chb.py --patient=chb24 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb01 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb02 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb03 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb04 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb05 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb06 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb07 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb08 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb09 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb10 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb11 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb12 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb13 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb14 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb15 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb16 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb17 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb18 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb19 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb20 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb21 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb22 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb23 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb24 --db=siena_wang
deactivate