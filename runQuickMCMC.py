#!/usr/bin/env python
"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""

import numpy as np
np.seterr(all='raise')
import matplotlib.pyplot as plt
#import corner

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

#import glob
#import json

import QuickCW.QuickCW
from QuickCW.QuickMCMCUtils import ChainParams
#import QuickCW.FastLikelihoodNumba as FastLikelihoodNumba

#make sure this points to the pickled pulsars you want to analyze
data_pkl = 'data/nanograv_11yr_psrs.pkl'

with open(data_pkl, 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)

print(len(psrs))

#number of iterations (increase to 100 million - 1 billion for actual analysis)
N = 5_000_000

n_int_block = 10_000 #number of iterations in a block (which has one shape update and the rest are projection updates)
save_every_n = 100_000 #number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block)
N_blocks = np.int64(N//n_int_block) #number of blocks to do
fisher_eig_downsample = 2000 #multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements

n_status_update = 100 #number of status update printouts (N/n_status_update needs to be an intiger multiple of n_int_block)
n_block_status_update = np.int64(N_blocks//n_status_update) #number of bllocks between status updates

assert N_blocks%n_status_update ==0 #or we won't print status updates
assert N%save_every_n == 0 #or we won't save a complete block
assert N%n_int_block == 0 #or we won't execute the right number of blocks

#Parallel tempering prameters
T_max = 3.
n_chain = 4

#make sure this points to your white noise dictionary
noisefile = 'data/quickCW_noisedict_kernel_ecorr.json'

#make sure this points to the RN empirical distribution file you plan to use (or set to None to not use empirical distributions)
rn_emp_dist_file = 'data/emp_dist.pkl'
#rn_emp_dist_file = None

#file containing information about pulsar distances - None means use pulsar distances present in psr objects
#if not None psr objects must have zero distance and unit variance
psr_dist_file = None

#this is where results will be saved
savefile = 'results/quickCW_test16.h5'
#savefile = None

#Setup and start MCMC
#object containing common parameters for the mcmc chain
chain_params = ChainParams(T_max,n_chain, n_block_status_update,
                           freq_bounds=np.array([np.nan, 3e-7]), #prior bounds used on the GW frequency (a lower bound of np.nan is interpreted as 1/T_obs)
                           n_int_block=n_int_block, #number of iterations in a block (which has one shape update and the rest are projection updates)
                           save_every_n=save_every_n, #number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block)
                           fisher_eig_downsample=fisher_eig_downsample, #multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements
                           rn_emp_dist_file=rn_emp_dist_file, #RN empirical distribution file to use (no empirical distribution jumps attempted if set to None)
                           savefile = savefile,#hdf5 file to save to, will not save at all if None
                           thin=100,  #thinning, i.e. save every `thin`th sample to file (increase to higher than one to keep file sizes small)
                           prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3, #probability of different jump types
                           dist_jump_weight=0.2, rn_jump_weight=0.3, gwb_jump_weight=0.1, common_jump_weight=0.2, all_jump_weight=0.2, #probability of updating different groups of parameters
                           fix_rn=False, zero_rn=False, fix_gwb=False, zero_gwb=False) #switches to turn off GWB or RN jumps and keep them fixed and to set them to practically zero (gamma=0.0, log10_A=-20)

pta,mcc = QuickCW.QuickCW.QuickCW(chain_params, psrs,
                                  amplitude_prior='detection', #specify amplitude prior to use - 'detection':uniform in log-amplitude, 'UL': uniform in amplitude
                                  psr_distance_file=psr_dist_file, #file to specify advanced (parallax+DM) pulsar distance priors, if None use regular Gaussian priors based on pulsar distances in pulsar objects
                                  noise_json=noisefile)

#Some parameters in chain_params can be updated later if needed
mcc.chain_params.thin = 10

#Do the main MCMC iteration
mcc.advance_N_blocks(N_blocks)
