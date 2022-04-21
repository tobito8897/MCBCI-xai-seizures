#!/bin/bash
source ../venv/bin/activate
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN00 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN01 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN02 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN03 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN04 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN05 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN06 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN07 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN08 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN09 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN10 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN11 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN12 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN13 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN14 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN15 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN16 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_train_windows_siena.py --patient=PN17 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN00 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN01 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN02 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN03 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN04 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN05 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN06 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN07 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN08 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN09 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN10 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN11 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN12 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN13 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN14 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN15 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN16 --db=siena_wang
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_siena.py --patient=PN17 --db=siena_wang
deactivate