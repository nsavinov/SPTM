#!/usr/bin/env bash
conda env create -f conda_env.txt
source activate sptm
pip install vizdoom==1.1.4
pip install omgifol==0.3.0
