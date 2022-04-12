#!/bin/bash
source ../venv/bin/activate
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb01
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb02
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb03
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb04
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb05
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb06
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb07
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb08
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb09
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb10
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb11
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb12
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb13
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb14
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb15
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb16
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb17
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb18
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb19
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb20
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb21
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb22
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb23
/lustre/home/ssanchez/python-core_375/bin/python3 generate_test_windows_chb.py --patient=chb24
deactivate