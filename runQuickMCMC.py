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

import glob
import json

import QuickCW
import CWFastLikelihoodNumba

#make sure this points to the pickled pulsars you want to analyze
data_pkl = 'data/quickCW_test.pkl'

with open(data_pkl, 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)

print(len(psrs))

#number of iterations
N = 1_000_000

#Parallel tempering prameters
T_max = 3.0
n_chain = 5

#make sure this points to your white noise dictionary
noisefile = 'data/quickCW_noisedict_gp_ecorr.json'

#this is where results will be saved
savefile = 'results/quickCW_test.h5'

#Setup and start MCMC
pta = QuickCW.QuickCW(N, T_max, n_chain, psrs,
                      n_int_block=10_000, #number of iterations in a block (which has one shape update and the rest are projection updates)
                      n_status_update=100, #number of status update printouts (N/n_status_update needs to be an intiger multiple of n_int_block)
                      save_every_n=100_000, #number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block)
                      noise_json=noisefile,
                      savefile=savefile)


