#!/bin/bash
set -e
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../..`
cd $SCRIPTPATH$DATAPATH

mkdir -p "data/chbmit"
mkdir -p "data/siena"

mkdir -p "processeddata/data_chb/test/"
mkdir -p "processeddata/data_chb/train/"
mkdir -p "processeddata/data_siena/test/"
mkdir -p "processeddata/data_siena/train/"

mkdir -p "models/hossain/"
mkdir -p "models/wang_1d/"
mkdir -p "models/wang_2d/"

mkdir -p "mldata/hossain/"
mkdir -p "mldata/wang_1d/"
mkdir -p "mldata/wang_2d/"