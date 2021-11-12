"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""
import pickle

import numpy as np
import numba as nb
#make sure to use the right threading layer
from numba import config
config.THREADING_LAYER = 'omp'

from numba import jit,njit,prange
from numba.experimental import jitclass
from numba.typed import List
#from numba_stats import uniform as uniform_numba
from numpy.random import uniform

import corner
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
import time
import re

import h5py

import CWFastLikelihoodNumba
import CWFastPrior

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
def QuickCW(N, T_max, n_chain, psrs, noise_json=None, n_status_update=100, n_int_block = 100, n_extrinsic_step=10, save_every_n=100, thin=10, savefile=None):
    #freq = 1e-8

    #use PTA used in current CW search
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # define white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    #ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection)

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # define powerlaw PSD and red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    cos_gwtheta = parameter.Uniform(-1,1)('0_cos_gwtheta')
    gwphi = parameter.Uniform(0,2*np.pi)('0_gwphi')

    #log_f = np.log10(freq)
    #log10_fgw = parameter.Constant(log_f)('0_log10_fgw')
    log10_fgw = parameter.Uniform(np.log10(3.5e-9), -7.0)('0_log10_fgw')

    #if freq>=191.3e-9:
    #    m = (1./(6**(3./2)*np.pi*freq*u.Hz))*(1./4)**(3./5)*(c.c**3/c.G)
    #    m_max = np.log10(m.to(u.Msun).value)
    #else:
    m_max = 10

    log10_mc = parameter.Uniform(7,m_max)('0_log10_mc')

    phase0 = parameter.Uniform(0, 2*np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

    p_phase = parameter.Uniform(0, 2*np.pi)
    p_dist = parameter.Normal(0, 1)

    log10_h = parameter.Uniform(-18, -11)('0_log10_h')
    #log10_h = parameter.LinearExp(-18, -11)('0_log10_h')

    tref = 53000*86400

    cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                                   log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0, psrTerm=True,
                                   p_phase=p_phase, p_dist=p_dist, evolve=True,
                                   psi=psi, cos_inc=cos_inc, tref=53000*86400)
    cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')

    log10_Agw = parameter.Constant(-16.27)('gwb_log10_A')
    gamma_gw = parameter.Constant(6.6)('gwb_gamma')
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=5, Tspan=Tspan,
                                            name='gw')

    tm = gp_signals.TimingModel()

    #s = ef + eq + ec + rn + crn + cw + tm
    #s = ef + eq + ec + rn + cw + tm
    s = ef + eq + ec + cw + tm

    models = [s(psr) for psr in psrs]

    pta = signal_base.PTA(models)

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)

    #print(noisedict)
    pta.set_default_params(noisedict)

    #print(pta.summary())
    print(pta.params)

    FastPrior = CWFastPrior.FastPrior(pta)
    print(FastPrior.uniform_par_ids)
    FPI = FastPriorInfo(FastPrior.uniform_par_ids, FastPrior.uniform_lows, FastPrior.uniform_highs,
                        FastPrior.normal_par_ids, FastPrior.normal_mus, FastPrior.normal_sigs)

    par_names = List(pta.param_names)
    par_names_cw = List(['0_cos_gwtheta', '0_cos_inc', '0_gwphi', '0_log10_fgw', '0_log10_h',
                         '0_log10_mc', '0_phase0', '0_psi'])
    par_names_cw_ext = List(['0_cos_inc', '0_log10_h', '0_phase0', '0_psi'])
    par_names_cw_int = List(['0_cos_gwtheta', '0_gwphi', '0_log10_fgw', '0_log10_mc'])
    for psr in pta.pulsars:
        par_names_cw.append(psr + "_cw0_p_dist")
        par_names_cw.append(psr + "_cw0_p_phase")
        par_names_cw_ext.append(psr + "_cw0_p_phase")
        par_names_cw_int.append(psr + "_cw0_p_dist")


    #using geometric spacing
    c = T_max**(1.0/(n_chain-1))
    Ts = c**np.arange(n_chain)
    print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\nTemperature ladder is:\n".format(n_chain,c),Ts)

    #set up samples array
    samples = np.zeros((n_chain, n_int_block*save_every_n+1, len(par_names)))

    #set up log_likelihood array
    log_likelihood = np.zeros((n_chain,n_int_block*save_every_n+1))

    print("Setting up first sample")
    for j in range(n_chain):
        samples[j,0,:] = np.array([par.sample() for par in pta.params])

        #set non-external parameters to injected for testing
        samples[j,0,par_names.index('0_cos_gwtheta')] = np.cos(np.pi/3.0)
        samples[j,0,par_names.index('0_gwphi')] = 4.5
        samples[j,0,par_names.index('0_log10_fgw')] = np.log10(2e-8)
        samples[j,0,par_names.index('0_log10_mc')] = np.log10(5e9)
        for psr in pta.pulsars:
            samples[j,0,par_names.index(psr + "_cw0_p_dist")] = 0.0
            #samples[j,0,par_names.index(psr + "_red_noise_gamma")] = noisedict[psr + "_red_noise_gamma"]
            #samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = noisedict[psr + "_red_noise_log10_A"]

        #also set external parameters for further testing
        #samples[j,0,par_names.index("0_cos_inc")] = np.cos(1.0)
        #samples[j,0,par_names.index("0_log10_h")] = np.log10(2e-15)
        #samples[j,0,par_names.index("0_phase0")] = 1.0
        #samples[j,0,par_names.index("0_psi")] = 1.0
        #p_phases = [2.6438308, 3.227938129, 2.95118811, 5.358659229, 1.06395234]
        #for ii, psr in enumerate(pta.pulsars):
        #    samples[j,0,par_names.index(psr + "_cw0_p_phase")] = p_phases[ii]

    #set up fast likelihoods
    FastLs = []
    x0s = List([])
    FLIs = List([])
    for j in range(n_chain):
        x0s.append( CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),
                                                 np.array([samples[j,0,par_names.index(par)] for par in par_names if "_cw0_p_phase" in par]),
                                                 np.array([samples[j,0,par_names.index(par)] for par in par_names if "_cw0_p_dist" in par]),
                                                 samples[j,0,par_names.index("0_cos_gwtheta")],
                                                 samples[j,0,par_names.index("0_cos_inc")],
                                                 samples[j,0,par_names.index("0_gwphi")],
                                                 samples[j,0,par_names.index("0_log10_fgw")],
                                                 samples[j,0,par_names.index("0_log10_h")],
                                                 samples[j,0,par_names.index("0_log10_mc")],
                                                 samples[j,0,par_names.index("0_phase0")],
                                                 samples[j,0,par_names.index("0_psi")]) )
        FastLs.append(CWFastLikelihoodNumba.CWFastLikelihood(psrs, pta, {pname:x for pname,x in zip(par_names,samples[j,0,:])}, x0s[j]))
        FLIs.append(FastLikeInfo(FastLs[j].resres,FastLs[j].logdet,FastLs[j].pos,FastLs[j].pdist,FastLs[j].NN,FastLs[j].MM_chol))

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((2,n_chain)) #columns: chain number; rows: proposal type (PT, internal fisher)
    a_no=np.zeros((2,n_chain))
    acc_fraction = a_yes/(a_no+a_yes)

    #printing info about initial parameters
    for j in range(n_chain):
        print("j="+str(j))
        print(samples[j,0,:])
        #log_likelihood[j,0] = pta.get_lnlikelihood(samples[j,0,:])
        log_likelihood[j,0] = FastLs[j].get_lnlikelihood(x0s[j])
        print("log_likelihood="+str(log_likelihood[j,0]))
        #print("log_prior="+str(pta.get_lnprior(samples[j,0,:])))
        print("log_prior="+str(FastPrior.get_lnprior(samples[j,0,:])))

    stop_iter = N

    ext_update = List([False,]*n_chain)
    sample_updates = List([np.copy(samples[j,0,:]) for j in range(n_chain)])

    ##############################################################################
    #
    # Main MCMC iteration
    #
    ##############################################################################
    for i in range(int(N/n_int_block)):
        #print(i*n_int_block)
        if savefile is not None and i%save_every_n==0 and i!=0:
            acc_fraction = a_yes/(a_no+a_yes)
            #np.savez(savefile, samples=samples[0,:i*n_int_block,:], par_names=par_names, acc_fraction=acc_fraction, log_likelihood=log_likelihood[:,:i*n_int_block])
            if i>save_every_n:
                print("Append to HDF5 file...")
                with h5py.File(savefile, 'a') as f:
                    f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
                    f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
                    f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                    f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
                    f['acc_fraction'][...] = np.copy(acc_fraction)
            else:
                print("Create HDF5 file...")
                with h5py.File(savefile, 'w') as f:
                    f.create_dataset('samples_cold', data=samples[0,:-1:thin,:], compression="gzip", chunks=True, maxshape=(int(N/thin),samples.shape[2]))
                    f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape=(samples.shape[0],int(N/thin)))
                    f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
                    f.create_dataset('acc_fraction', data=acc_fraction)
            #clear out log_likelihood and samples arrays
            samples_now = samples[:,-1,:]
            log_likelihood_now = log_likelihood[:,-1]
            samples = np.zeros((n_chain, n_int_block*save_every_n+1, len(par_names)))
            log_likelihood = np.zeros((n_chain,n_int_block*save_every_n+1))
            samples[:,0,:] = np.copy(samples_now)
            log_likelihood[:,0] = np.copy(log_likelihood_now)


        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            print('Progress: {0:2.2f}% '.format(i*n_int_block/N*100) +
                      'Acceptance fraction #columns: chain number; rows: proposal type (PT, fisher):')
            print(acc_fraction)
        #update extrinsic parameters sometimes
        if i%n_extrinsic_step==0 and i!=0:
            sample_updates = List(do_extrinsic_update(n_chain, psrs, pta, samples, i*n_int_block, Ts, a_yes, a_no, x0s, FastLs, FLIs, FastPrior, par_names, par_names_cw, par_names_cw_ext, par_names_cw_int, log_likelihood, n_int_block, save_every_n))
            ext_update = List([True,]*n_chain)
            #print("ext_update")
        #do actual MCMC step
        #do_fisher_jump(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FastLs, FastPrior, par_names, par_names_cw, par_names_cw_ext, log_likelihood, sample_updates)
        do_intrinsic_block(n_chain, samples, i*n_int_block, Ts, a_yes, a_no, x0s, FLIs, FPI, par_names, par_names_cw, par_names_cw_ext, log_likelihood, sample_updates, ext_update, n_int_block, save_every_n)
        #do a PT step at the end of each intrinsic block (only do it if all intrinsic update was accepted, and don;t do at the last step)
        #if i != int(N/n_int_block)-1:
        #    if not any(ext_update):
        #        do_pt_swap(n_chain, pta, samples, (i+1)*n_int_block-1, Ts, a_yes, a_no, x0s, FastLs, FLIs, FastPrior, par_names, par_names_cw, par_names_cw_ext, log_likelihood)
        #    else:
        #        print("No PT step, since not all extrinsic update was accepted")
        #        for j in range(n_chain):
        #            samples[j,(i+1)*n_int_block,:] = np.copy(samples[j,(i+1)*n_int_block-1,:])
        #            log_likelihood[j,(i+1)*n_int_block] = log_likelihood[j,(i+1)*n_int_block-1]

    acc_fraction = a_yes/(a_no+a_yes)
    print("Append to HDF5 file...")
    with h5py.File(savefile, 'a') as f:
        f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
        f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
        f['acc_fraction'][...] = np.copy(acc_fraction)
    #return samples, par_names, acc_fraction, pta, log_likelihood
    return pta

################################################################################
#
#UPDATE EXTRINSIC PARAMETERS AND RECALCULATE FILTERS
#
################################################################################
def do_extrinsic_update(n_chain, psrs, pta, samples, i, Ts, a_yes, a_no, x0s, FastLs, FLIs, FastPrior, par_names, par_names_cw, par_names_cw_ext, par_names_cw_int, log_likelihood, n_int_block, save_every_n):
    sample_updates = []
    for j in range(n_chain):
        jump_select = np.random.randint(len(par_names_cw_int))
        jump = np.zeros(len(par_names))
        jump[par_names.index(par_names_cw_int[jump_select])] = 0.03
        
        sample_update = np.copy(samples[j,i%(n_int_block*save_every_n),:]) + jump*np.random.normal()
        #if j==0:
        #    print(samples[j,i,:])
        #    print(sample_update)
        #    print(sample_update-samples[j,i,:])
        sample_updates.append(sample_update)

        x0s[j] = CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),
                                              np.array([sample_update[par_names.index(par)] for par in par_names if "_cw0_p_phase" in par]),
                                              np.array([sample_update[par_names.index(par)] for par in par_names if "_cw0_p_dist" in par]),
                                              sample_update[par_names.index("0_cos_gwtheta")],
                                              sample_update[par_names.index("0_cos_inc")],
                                              sample_update[par_names.index("0_gwphi")],
                                              sample_update[par_names.index("0_log10_fgw")],
                                              sample_update[par_names.index("0_log10_h")],
                                              sample_update[par_names.index("0_log10_mc")],
                                              sample_update[par_names.index("0_phase0")],
                                              sample_update[par_names.index("0_psi")])
        
        if "_cw0_p_dist" in par_names_cw_int[jump_select]:
            #print("psr diatance update for " + par_names_cw_int[jump_select][:-11])
            FastLs[j].update_pulsar_distance(x0s[j], pta.pulsars.index(par_names_cw_int[jump_select][:-11]))
        else:
            #print("sky location, frequency, or chirp mass update")
            #FastLs[j] = CWFastLikelihoodNumba.CWFastLikelihood(psrs, pta, {pname:x for pname,x in zip(par_names,sample_update)}, x0s[j])
            FastLs[j].update_intrinsic_params(x0s[j])

        #update FLIs too
        FLIs[j] = FastLikeInfo(FastLs[j].resres,FastLs[j].logdet,FastLs[j].pos,FastLs[j].pdist,FastLs[j].NN,FastLs[j].MM_chol)
        
    return sample_updates

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################
#def do_fisher_jump(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FastLs, FastPrior, par_names, par_names_cw, par_names_cw_ext, log_likelihood, sample_updates):
@njit(parallel=False)
def do_intrinsic_block(n_chain, samples, i, Ts, a_yes, a_no, x0s, FLIs, FPI, par_names, par_names_cw, par_names_cw_ext, log_likelihood, sample_updates, ext_update, n_int_block, save_every_n):
    #print("FISHER")
    #print("-"*100)
    for j in prange(0,n_chain):
        for k in range(n_int_block-1):
            #print(j,i+k)
            samples_current = np.copy(samples[j,(i+k)%(n_int_block*save_every_n),:])

            jump_select = np.random.randint(len(par_names_cw_ext))
            jump = np.zeros(len(par_names))
            jump[par_names.index(par_names_cw_ext[jump_select])] = 0.5

            if ext_update[j]:
                new_point = sample_updates[j] + jump*np.random.normal()
            else:
                new_point = samples_current + jump*np.random.normal()

            #if j==0: print(samples_current)

            #new_point = samples_current + jump*np.random.normal()

            #if j==0: print(new_point)

            x0s[j].cw_p_phases = np.array([new_point[par_names.index(par)] for par in par_names if "_cw0_p_phase" in par])
            x0s[j].cos_inc = new_point[par_names.index("0_cos_inc")]
            x0s[j].log10_h = new_point[par_names.index("0_log10_h")]
            x0s[j].phase0 = new_point[par_names.index("0_phase0")]
            x0s[j].psi = new_point[par_names.index("0_psi")]

            #print(log_L)
            #log_L = FastLs[j].get_lnlikelihood(x0s[j])
            log_L = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0s[j],FLIs[j].resres,FLIs[j].logdet,FLIs[j].pos,FLIs[j].pdist,FLIs[j].NN,FLIs[j].MM_chol)
            #if j==0 and i%1_000==0:
            #    print("New log_L="+str(log_L))
            #    print("Old log_L="+str(pta.get_lnlikelihood(new_point)))
            #if j==0: print(log_L)
            log_acc_ratio = log_L/Ts[j]
            #log_acc_ratio += FastPrior.get_lnprior(new_point)
            log_acc_ratio += CWFastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                       FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            #if j==0: print(log_acc_ratio)
            log_acc_ratio += -log_likelihood[j,(i+k)%(n_int_block*save_every_n)]/Ts[j]
            #log_acc_ratio += -FastPrior.get_lnprior(samples_current)
            log_acc_ratio += -CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)

            #print(pta.get_lnprior(new_point))
            #print(FastPrior.get_lnprior(new_point))

            acc_ratio = np.exp(log_acc_ratio)
            #if j==0: print(acc_ratio)
            #if np.random.uniform()<=acc_ratio:
            acc_decide = uniform(0.0, 1.0, 1)
            if acc_decide<=acc_ratio:
                #if j==0: print("yeah")
                samples[j,(i+k)%(n_int_block*save_every_n)+1,:] = np.copy(new_point)
                log_likelihood[j,(i+k)%(n_int_block*save_every_n)+1] = log_L
                a_yes[1,j]+=1
                ext_update[j] = False
            else:
                #if j==0: print("ohh")
                samples[j,(i+k)%(n_int_block*save_every_n)+1,:] = np.copy(samples[j,(i+k)%(n_int_block*save_every_n),:])
                x0s[j].cw_p_phases = np.array([samples_current[par_names.index(par)] for par in par_names if "_cw0_p_phase" in par])
                x0s[j].cos_inc = samples_current[par_names.index("0_cos_inc")]
                x0s[j].log10_h = samples_current[par_names.index("0_log10_h")]
                x0s[j].phase0 = samples_current[par_names.index("0_phase0")]
                x0s[j].psi = samples_current[par_names.index("0_psi")]
                log_likelihood[j,(i+k)%(n_int_block*save_every_n)+1] = log_likelihood[j,(i+k)%(n_int_block*save_every_n)]
                a_no[1,j]+=1

    iii = (i+n_int_block-1)%(n_int_block*save_every_n)

    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))
  
    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,iii])

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    #for swap_chain in reversed(range(n_chain-1)):
    for swap_chain in range(n_chain-2, -1, -1): #same as reversed(range(n_chain-1)) but supported in numba
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain+1]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain+1]

        acc_ratio = np.exp(log_acc_ratio)
        acc_decide = uniform(0.0, 1.0, 1)
        if acc_decide<=acc_ratio:# and do_PT:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[0,swap_chain]+=1
        else:
            a_no[0,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    FLIs_new = []
    for j in range(n_chain):
        samples[j,iii+1,:] = samples[swap_map[j],iii,:]
        log_likelihood[j,iii+1] = log_likelihood[swap_map[j],iii]
        FLIs_new.append(FLIs[swap_map[j]])

    FLIs = FLIs_new

################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FastLs, FLIs, FastPrior, par_names, par_names_cw, par_names_cw_ext, log_likelihood):
    #print("PT")
    #print(i)
    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,i])

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    for swap_chain in reversed(range(n_chain-1)):
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain+1]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain+1]

        acc_ratio = np.exp(log_acc_ratio)
        if np.random.uniform()<=acc_ratio:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[0,swap_chain]+=1
        else:
            a_no[0,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    FastLs_new = []
    FLIs_new = []
    for j in range(n_chain):
        samples[j,i+1,:] = samples[swap_map[j],i,:]
        log_likelihood[j,i+1] = log_likelihood[swap_map[j],i]
        FastLs_new.append(FastLs[swap_map[j]])
        FLIs_new.append(FLIs[swap_map[j]])

    FastLs = FastLs_new
    FLIs = FLIs_new


################################################################################
#
#DRAW FROM PRIOR JUMP
#
################################################################################
def do_draw_from_prior(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FastLs, FastPrior, par_names, par_names_cw, par_names_cw_ext, log_likelihood):
    #print("FISHER")
    for j in range(n_chain):
        samples_current = np.copy(samples[j,i,:])
        new_point = np.copy(samples[j,i,:])

        jump_select = np.random.randint(len(par_names_cw_ext))
        new_point[par_names.index(par_names_cw_ext[jump_select])] = pta.params[par_names.index(par_names_cw_ext[jump_select])].sample()

        x0s[j].cw_p_phases = np.array([new_point[par_names.index(par)] for par in par_names if "_cw0_p_phase" in par])
        x0s[j].cos_inc = new_point[par_names.index("0_cos_inc")]
        x0s[j].log10_h = new_point[par_names.index("0_log10_h")]
        x0s[j].phase0 = new_point[par_names.index("0_phase0")]
        x0s[j].psi = new_point[par_names.index("0_psi")]

        #print(log_L)
        log_L = FastLs[j].get_lnlikelihood(x0s[j])
        #print(log_L)
        log_acc_ratio = log_L/Ts[j]
        log_acc_ratio += FastPrior.get_lnprior(new_point)
        log_acc_ratio += -log_likelihood[j,i]/Ts[j]
        log_acc_ratio += -FastPrior.get_lnprior(samples_current)

        #print(pta.get_lnprior(new_point))
        #print(FastPrior.get_lnprior(new_point))

        acc_ratio = np.exp(log_acc_ratio)
        #print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,:] = np.copy(new_point)
            log_likelihood[j,i+1] = log_L
            a_yes[1,j]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            x0s[j].cw_p_phases = np.array([samples_current[par_names.index(par)] for par in par_names if "_cw0_p_phase" in par])
            x0s[j].cos_inc = samples_current[par_names.index("0_cos_inc")]
            x0s[j].log10_h = samples_current[par_names.index("0_log10_h")]
            x0s[j].phase0 = samples_current[par_names.index("0_phase0")]
            x0s[j].psi = samples_current[par_names.index("0_psi")]
            log_likelihood[j,i+1] = log_likelihood[j,i]
            a_no[1,j]+=1

@jitclass([('resres',nb.float64),('logdet',nb.float64),('pos',nb.float64[:,:]),\
           ('pdist',nb.float64[:,:]),('NN',nb.float64[:,:]),('MM_chol',nb.float64[:,:,:])])
class FastLikeInfo:
    """simple jitclass to store the various elements of fast likelihood calculation in a way that can be accessed quickly from a numba environment"""
    def __init__(self,resres,logdet,pos,pdist,NN,MM_chol):
        self.resres = resres
        self.logdet = logdet
        self.pos = pos
        self.pdist = pdist
        self.NN = NN
        self.MM_chol = MM_chol


@jitclass([('uniform_par_ids',nb.int64[:]),('uniform_lows',nb.float64[:]),('uniform_highs',nb.float64[:]),\
           ('normal_par_ids',nb.int64[:]),('normal_mus',nb.float64[:]),('normal_sigs',nb.float64[:])])
class FastPriorInfo:
    """simple jitclass to store the various elements of fast prior calculation in a way that can be accessed quickly from a numba environment"""
    def __init__(self, uniform_par_ids, uniform_lows, uniform_highs, normal_par_ids, normal_mus, normal_sigs):
        self.uniform_par_ids = uniform_par_ids
        self.uniform_lows = uniform_lows
        self.uniform_highs = uniform_highs
        self.normal_par_ids = normal_par_ids
        self.normal_mus = normal_mus
        self.normal_sigs = normal_sigs
