#!/bin/bash
set -e
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../..`
cd $SCRIPTPATH$DATAPATH

mkdir -p "data/chbmit"
mkdir -p "data/siena"

mkdir -p "processeddata/wangdata_chb/test/"
mkdir -p "processeddata/wangdata_chb/train/"
mkdir -p "processeddata/wangdata_siena/test/"
mkdir -p "processeddata/wangdata_siena/train/"

mkdir -p "processeddata/hossaindata_chb/test/"
mkdir -p "processeddata/hossaindata_chb/train/"
mkdir -p "processeddata/hossaindata_siena/test/"
mkdir -p "processeddata/hossaindata_siena/train/"

mkdir -p "models/hossain/"
mkdir -p "models/wang_1d/"
mkdir -p "models/wang_2d/"

mkdir -p "mldata/hossain/"
mkdir -p "mldata/wang_1d/"
mkdir -p "mldata/wang_2d/"