#!/bin/bash

#It can be beneficial to set the numba thread number to the number of physical cores of the machine used
export NUMBA_NUM_THREADS=12
export MKL_NUM_THREADS=12

#nice -n 20 /usr/bin/time python -u runQuickMCMC.py
python -u QuickCW/runQuickMCMC.py
