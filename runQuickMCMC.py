import numpy as np
import matplotlib.pyplot as plt
import corner

import pickle

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

from enterprise_extensions import deterministic

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP

import glob
import json

import QuickCW
import CWFastLikelihoodNumba

#with open('data/fast_like_test_psrs_A2e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad.pkl', 'rb') as psr_pkl:
with open('data/fast_like_test_psrs_A2e-15_M5e9_f2e-8_evolve_no_gwb.pkl', 'rb') as psr_pkl:
#with open('data/fast_like_test_psrs.pkl', 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)

print(len(psrs))

#N = 30_000
#N = 500_000
N = 2_000_000
T_max = 5
n_chain = 8

noisefile = 'data/channelized_12p5yr_v3_full_noisedict_gp_ecorr.json'

#savefile = 'results/quickCW_test_A2e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.npz'
#savefile = 'results/quickCW_test_A1e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.npz'
savefile = None#'results/quickCW_test_A1e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.h5'

pta = QuickCW.QuickCW(N, T_max, n_chain, psrs,
                      n_status_update=200, n_int_block=1000, save_every_n=10000,
                      noise_json=noisefile,
                      savefile=savefile)


