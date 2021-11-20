"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""
import pickle

from time import perf_counter
import glob
import json
import time

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

#import re

import h5py

import CWFastLikelihoodNumba
import CWFastPrior
import const_mcmc as cm

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
def QuickCW(N, T_max, n_chain, psrs, noise_json=None, n_status_update=100, n_int_block=100, n_extrinsic_step=1000, save_every_n=10_000, thin=10, savefile=None, n_update_fisher=100_000):
    #freq = 1e-8
    ti = perf_counter()

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

    cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                                   log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0, psrTerm=True,
                                   p_phase=p_phase, p_dist=p_dist, evolve=True,
                                   psi=psi, cos_inc=cos_inc, tref=cm.tref)
    cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')

    log10_Agw = parameter.Constant(-16.27)('gwb_log10_A')
    gamma_gw = parameter.Constant(6.6)('gwb_gamma')
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=5, Tspan=Tspan,
                                            name='gw')

    tm = gp_signals.TimingModel()

    #s = ef + eq + ec + rn + crn + cw + tm
    #s = ef + eq + ec + rn + cw + tm
    #s = ef + eq + ec + cw + tm
    s = ef + cw + tm

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

    par_inds_cw_p_phase_ext = np.zeros(len(pta.pulsars),dtype=np.int64)

    for i,psr in enumerate(pta.pulsars):
        par_names_cw.append(psr + "_cw0_p_dist")
        par_inds_cw_p_phase_ext[i] = len(par_names_cw)
        par_names_cw.append(psr + "_cw0_p_phase")
        par_names_cw_ext.append(psr + "_cw0_p_phase")
        par_names_cw_int.append(psr + "_cw0_p_dist")

    #using geometric spacing
    c = T_max**(1.0/(n_chain-1))
    Ts = c**np.arange(n_chain)
    print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\nTemperature ladder is:\n".format(n_chain,c),Ts)

    #set up samples array
    samples = np.zeros((n_chain, save_every_n+1, len(par_names)))

    #set up log_likelihood array
    log_likelihood = np.zeros((n_chain,save_every_n+1))

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
        samples[j,0,par_names.index("0_cos_inc")] = np.cos(1.0)
        samples[j,0,par_names.index("0_log10_h")] = np.log10(1e-15)
        samples[j,0,par_names.index("0_phase0")] = 1.0
        samples[j,0,par_names.index("0_psi")] = 1.0
        p_phases = [2.6438308,3.2279381,2.9511881,5.3586592,1.0639523,
                    2.1564047,1.1287014,5.9545189,4.3189053,1.3181107,
                    0.1205947,1.1594364,3.5189818,5.8613215,3.6653746,
                    0.0653161,4.3756635,2.2423111,4.5429403,5.1370920,
                    1.9794586,1.0159356,2.9529407,1.7553771,1.5110336,
                    4.0558141,1.6855663,3.5614665,4.1527070,5.2239841,
                    4.4504891,4.8126553,3.6622998,4.4647441,2.8561429,
                    0.6874573,0.3762146,1.6691351,0.8147172,0.3051969,
                    1.6177042,2.8609930,5.0392969,0.3359030,1.0489710]
        for ii, psr in enumerate(pta.pulsars):
            samples[j,0,par_names.index(psr + "_cw0_p_phase")] = p_phases[ii]

    #set up fast likelihoods
    x0s = List([])
    FLIs  = List([])
    for j in range(n_chain):
        x0s.append( CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[j,0],par_names,par_names_cw_ext))
        FLIs.append(CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0s[j]))

    #calculate the diagonal elements of the fisher matrix
    fisher_diag = np.ones((n_chain, len(par_names)))
    for j in range(n_chain):
        fisher_diag[j,:] = get_fisher_diagonal(Ts[j], samples[j,0,:], par_names, par_names_cw_ext, x0s[j], FLIs[j])
        print(fisher_diag[j,:])


    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((11,n_chain),dtype=np.int64) #columns: chain number; rows: proposal type (PT, cos_gwtheta, cos_inc, gwphi, fgw, h, mc, phase0, psi, p_phases, p_dists)
    a_no=np.zeros((11,n_chain),dtype=np.int64)
    acc_fraction = a_yes/(a_no+a_yes)

    #printing info about initial parameters
    for j in range(n_chain):
        print("j="+str(j))
        print(samples[j,0,:])
        #log_likelihood[j,0] = pta.get_lnlikelihood(samples[j,0,:])
        log_likelihood[j,0] = FLIs[j].get_lnlikelihood(x0s[j])
        print("log_likelihood="+str(log_likelihood[j,0]))
        #print("log_prior="+str(pta.get_lnprior(samples[j,0,:])))
        print("log_prior="+str(FastPrior.get_lnprior(samples[j,0,:])))

    stop_iter = N

    #ext_update = List([False,]*n_chain)
    #sample_updates = List([np.copy(samples[j,0,:]) for j in range(n_chain)])

    tf_init = perf_counter()
    print('finished initialization steps in '+str(tf_init-ti)+'s')
    ti_loop = perf_counter()

    ##############################################################################
    #
    # Main MCMC iteration
    #
    ##############################################################################
    for i in range(int(N/n_int_block)):
        if (i*n_int_block)%save_every_n==0 and i!=0:
            acc_fraction = a_yes/(a_no+a_yes)
            #np.savez(savefile, samples=samples[0,:i*n_int_block,:], par_names=par_names, acc_fraction=acc_fraction, log_likelihood=log_likelihood[:,:i*n_int_block])
            if savefile is not None:
                if i*n_int_block>save_every_n:
                    print("Append to HDF5 file...")
                    with h5py.File(savefile, 'a') as f:
                        f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
                        f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
                        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
                        f['acc_fraction'][...] = np.copy(acc_fraction)
                        f['fisher_diag'][...] = np.copy(fisher_diag)
                else:
                    print("Create HDF5 file...")
                    with h5py.File(savefile, 'w') as f:
                        f.create_dataset('samples_cold', data=samples[0,:-1:thin,:], compression="gzip", chunks=True, maxshape=(int(N/thin),samples.shape[2]))
                        f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape=(samples.shape[0],int(N/thin)))
                        f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
                        f.create_dataset('acc_fraction', data=acc_fraction)
                        f.create_dataset('fisher_diag', data=fisher_diag)
            #clear out log_likelihood and samples arrays
            samples_now = samples[:,-1,:]
            log_likelihood_now = log_likelihood[:,-1]
            samples = np.zeros((n_chain, save_every_n+1, len(par_names)))
            log_likelihood = np.zeros((n_chain,save_every_n+1))
            samples[:,0,:] = np.copy(samples_now)
            log_likelihood[:,0] = np.copy(log_likelihood_now)
        if (i*n_int_block)%n_update_fisher==0 and i!=0:
            print("Updating Fisher diagonals")
            for j in range(n_chain):
                fisher_diag[j,:] = get_fisher_diagonal(Ts[j], samples[j,(i*n_int_block)%save_every_n,:], par_names, par_names_cw_ext, x0s[j], FLIs[j])
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            print('Progress: {0:2.2f}% '.format(i*n_int_block/N*100) +
                      'Acceptance fraction #columns: chain number; rows: proposal type (PT, cos_gwtheta, cos_inc, gwphi, fgw, h, mc, phase0, psi, p_phases, p_dists):')
            t_itr = perf_counter()
            print('at t= '+str(t_itr-ti_loop)+'s')
            print(acc_fraction)
            print("New log_L=", str(FLIs[0].get_lnlikelihood(x0s[0])))#,FLIs[0].resres,FLIs[0].logdet,FLIs[0].pos,FLIs[0].pdist,FLIs[0].NN,FLIs[0].MMs)))
            #print("Old log_L=", str(pta.get_lnlikelihood(samples[0,(i*n_int_block)%save_every_n,:])))
        #update extrinsic parameters sometimes
        if i%n_extrinsic_step==0 and i!=0:
            do_intrinsic_update(n_chain, pta, samples, i*n_int_block, Ts, a_yes, a_no, x0s, FLIs, FastPrior, par_names, par_names_cw_int, log_likelihood, save_every_n, fisher_diag)
            a_yes_counts_loc,a_no_counts_loc = do_extrinsic_block(n_chain, samples, i*n_int_block+1, Ts, x0s, FLIs, FPI, len(par_names), len(par_names_cw_ext), log_likelihood, n_int_block-1, save_every_n, fisher_diag)
            a_yes += summarize_a_ext(a_yes_counts_loc,par_inds_cw_p_phase_ext)
            a_no += summarize_a_ext(a_no_counts_loc,par_inds_cw_p_phase_ext)
            do_pt_swap(n_chain, samples, (i+1)*n_int_block-1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n)
        else:
            a_yes_counts_loc,a_no_counts_loc = do_extrinsic_block(n_chain, samples, i*n_int_block, Ts, x0s, FLIs, FPI, len(par_names), len(par_names_cw_ext), log_likelihood, n_int_block, save_every_n, fisher_diag)
            a_yes += summarize_a_ext(a_yes_counts_loc,par_inds_cw_p_phase_ext)
            a_no += summarize_a_ext(a_no_counts_loc,par_inds_cw_p_phase_ext)
            do_pt_swap(n_chain, samples, (i+1)*n_int_block-1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n)

    acc_fraction = a_yes/(a_no+a_yes)
    print("Append to HDF5 file...")
    if savefile is not None:
        with h5py.File(savefile, 'a') as f:
            f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
            f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
            f['acc_fraction'][...] = np.copy(acc_fraction)
            f['fisher_diag'][...] = np.copy(fisher_diag)
        #return samples, par_names, acc_fraction, pta, log_likelihood
    tf = perf_counter()
    print('whole function time ='+str(tf-ti)+'s')
    print('loop time ='+str(tf-ti_loop)+'s')
    return pta

################################################################################
#
#UPDATE INTRINSIC PARAMETERS AND RECALCULATE FILTERS
#
################################################################################
def do_intrinsic_update(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FLIs, FPI, par_names, par_names_cw_int, log_likelihood, save_every_n, fisher_diag):
    #print("EXT")
    for j in range(n_chain):
        #save MMs and NN so we can revert them if the chain is rejected
        MMs_save = FLIs[j].MMs.copy()
        NN_save = FLIs[j].NN.copy()
        jump_select = np.random.randint(len(par_names_cw_int))
        jump = np.zeros(len(par_names))
        jump_idx = par_names.index(par_names_cw_int[jump_select])
        jump[jump_idx] = fisher_diag[j, jump_idx] #0.1

        samples_current = np.copy(samples[j,i%save_every_n,:])

        #print('jump select',jump_select,par_names_cw_int[jump_select])
        new_point = np.copy(samples[j,i%save_every_n,:]) + jump*np.random.normal()
        #TODO check wrapping is working right
        if jump_idx == x0s[j].idx_cos_gwtheta or jump_idx == x0s[j].idx_gwphi:
            new_point[x0s[j].idx_cos_gwtheta],new_point[x0s[j].idx_gwphi] = reflect_cosines(new_point[x0s[j].idx_cos_gwtheta],new_point[x0s[j].idx_gwphi],np.pi,2*np.pi)

        x0s[j].update_params(new_point)

        if "_cw0_p_dist" in par_names_cw_int[jump_select]:
            FLIs[j].update_pulsar_distance(x0s[j], pta.pulsars.index(par_names_cw_int[jump_select][:-11]))
            acc_idx = 10
        else:
            #print("sky location, frequency, or chirp mass update")
            FLIs[j].update_intrinsic_params(x0s[j])
            if par_names_cw_int[jump_select]=="0_cos_gwtheta":
                acc_idx = 1
            elif par_names_cw_int[jump_select]=="0_gwphi":
                acc_idx = 3
            elif par_names_cw_int[jump_select]=="0_log10_fgw":
                acc_idx = 4
            elif par_names_cw_int[jump_select]=="0_log10_mc":
                acc_idx = 6

        #check the maximum toa is not such that the source has already merged, and if so automatically reject the proposal
        w0 = np.pi * 10.0**x0s[j].log10_fgw
        mc = 10.0**x0s[j].log10_mc * const.Tsun

        if (1. - 256./5. * mc * const.Tsun**(5./3.) * w0**(8./3.) * (FLIs[j].max_toa - cm.tref)) < 0:
            log_acc_ratio = -1.
            acc_decide = 0.
        else:

            #update FLIs too
            log_L = FLIs[j].get_lnlikelihood(x0s[j])
            log_acc_ratio = log_L/Ts[j]
            log_acc_ratio += CWFastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                       FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_acc_ratio += -log_likelihood[j,i%save_every_n]/Ts[j]
            log_acc_ratio += -CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                        FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)

            acc_decide = np.log(uniform(0.0, 1.0, 1))
            #print(pta.get_lnprior(new_point))
            #print(FastPrior.get_lnprior(new_point))

        if acc_decide<=log_acc_ratio:
            x0s[j].update_params(new_point)

            samples[j,i%save_every_n+1,:] = np.copy(new_point)

            log_likelihood[j,i%save_every_n+1] = log_L
            a_yes[acc_idx,j]+=1
        else:
            samples[j,(i+1)%save_every_n,:] = np.copy(samples_current)#np.copy(samples[j,i%save_every_n,:])

            log_likelihood[j,(i+1)%save_every_n] = log_likelihood[j,i%save_every_n]

            a_no[acc_idx,j]+=1
            x0s[j].update_params(samples_current)

            if "_cw0_p_dist" in par_names_cw_int[jump_select]:
                #revert the changes to FastLs
                FLIs[j].MMs = MMs_save
                FLIs[j].NN = NN_save
            else:
                #revert the changes to FastLs
                FLIs[j].MMs = MMs_save
                FLIs[j].NN = NN_save
                #revert the safety tracking parameters that were altered by update_intrinsic
                FLIs[j].cos_gwtheta = x0s[j].cos_gwtheta
                FLIs[j].gwphi = x0s[j].gwphi
                FLIs[j].log10_fgw = x0s[j].log10_fgw
                FLIs[j].log10_mc = x0s[j].log10_mc#
                #print("sky location, frequency, or chirp mass update")

def summarize_a_ext(a_counts,par_inds_cw_p_phase_ext):
    """helper to sumarize the acceptance rate counts of different jumps"""
    n_chain = a_counts.shape[1]
    a_res = np.zeros((11,n_chain),dtype=np.int64)
    for j in range(0,n_chain):
        for idx in par_inds_cw_p_phase_ext:
            a_res[9,j] += a_counts[idx,j]
        a_res[2,j] += a_counts[1,j]
        a_res[5,j] += a_counts[4,j]
        a_res[7,j] += a_counts[6,j]
        a_res[8,j] += a_counts[7,j]
    return a_res


################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################
@njit(parallel=True)
def do_extrinsic_block(n_chain, samples, i, Ts, x0s, FLIs, FPI, n_par_tot, n_par_ext, log_likelihood, n_int_block, save_every_n, fisher_diag):
    #print("FISHER")
    #print("-"*100)
    a_yes_counts = np.zeros((n_par_tot,n_chain),dtype=np.int64)
    a_no_counts = np.zeros((n_par_tot,n_chain),dtype=np.int64)

    for k in range(n_int_block-1):
        for j in prange(0,n_chain):
            samples_current = np.copy(samples[j,(i+k)%save_every_n,:])

            jump_select = np.random.randint(n_par_ext)
            jump = np.zeros(n_par_tot)
            jump_idx = x0s[j].idx_cw_ext[jump_select]
            jump[jump_idx] = fisher_diag[j, jump_idx]*np.random.normal() #0.5

            new_point = samples_current + jump#*np.random.normal()

            if jump_idx == x0s[j].idx_cos_inc or jump_idx == x0s[j].idx_psi:
                #TODO check these are the right parameters to be shifting
                new_point[x0s[j].idx_cos_inc],new_point[x0s[j].idx_psi] = reflect_cosines(new_point[x0s[j].idx_cos_inc],new_point[x0s[j].idx_psi],np.pi/2,np.pi)

            x0s[j].update_params(new_point)

            log_L = FLIs[j].get_lnlikelihood(x0s[j])#FLIs[j].resres,FLIs[j].logdet,FLIs[j].pos,FLIs[j].pdist,FLIs[j].NN,FLIs[j].MMs)
            log_acc_ratio = log_L/Ts[j]
            log_acc_ratio += CWFastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                       FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_acc_ratio += -log_likelihood[j,(i+k)%save_every_n]/Ts[j]
            log_acc_ratio += -CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)

            acc_decide = np.log(uniform(0.0, 1.0, 1))
            if acc_decide<=log_acc_ratio:
                #if j==0: print("yeah")
                samples[j,(i+k)%save_every_n+1,:] = new_point
                log_likelihood[j,(i+k)%save_every_n+1] = log_L
                a_yes_counts[jump_idx,j] += 1

            else:
                #if j==0: print("ohh")
                samples[j,(i+k)%save_every_n+1,:] = samples[j,(i+k)%save_every_n,:]

                x0s[j].update_params(samples_current)
                log_likelihood[j,(i+k)%save_every_n+1] = log_likelihood[j,(i+k)%save_every_n]

                a_no_counts[jump_idx,j] += 1
    return a_yes_counts,a_no_counts


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
@njit()
def do_pt_swap(n_chain, samples, i, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n):
    #print("PT")

    iii = i%save_every_n
    #print("PT")

    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,iii])

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    #for swap_chain in reversed(range(n_chain-1)):
    for swap_chain in range(n_chain-2, -1, -1): #same as reversed(range(n_chain-1)) but supported in numba
        assert swap_map[swap_chain] == swap_chain
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain+1]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain+1]

        acc_decide = np.log(uniform(0.0, 1.0, 1))
        if acc_decide<=log_acc_ratio:# and do_PT:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[0,swap_chain]+=1
        else:
            a_no[0,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    FLIs_new = []
    x0s_new = []
    for j in range(n_chain):
        samples[j,iii+1,:] = np.copy(samples[swap_map[j],iii,:])
        log_likelihood[j,iii+1] = log_likelihood[swap_map[j],iii]
        FLIs_new.append(FLIs[swap_map[j]])
        x0s_new.append(x0s[swap_map[j]])

    FLIs[:] = List(FLIs_new)
    x0s[:] = List(x0s_new)

def do_pt_swap_alt(n_chain, samples, i, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n):
    """modification to swap routine that is easier to adapt to arbitrary swap proposals"""
    #print("PT")

    iii = i%save_every_n
    #print("PT")

    #set up map to help keep track of swaps
    #swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    #log_Ls = []
    #for j in range(n_chain):
    #    log_Ls.append(log_likelihood[j,iii])
    log_Ls = log_likelihood[:,iii].copy()
    samples_cur = samples[:,iii,:].copy()

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    #for swap_chain in reversed(range(n_chain-1)):
    for swap_chain in range(n_chain-2, -1, -1): #same as reversed(range(n_chain-1)) but supported in numba
        log_acc_ratio = -log_Ls[swap_chain] / Ts[swap_chain]
        log_acc_ratio += -log_Ls[swap_chain+1] / Ts[swap_chain+1]
        log_acc_ratio += log_Ls[swap_chain+1] / Ts[swap_chain]
        log_acc_ratio += log_Ls[swap_chain] / Ts[swap_chain+1]

        acc_decide = np.log(uniform(0.0, 1.0, 1))
        if acc_decide<=log_acc_ratio:# and do_PT:
            #swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[0,swap_chain]+=1
        else:
            a_no[0,swap_chain]+=1

        log_L_temp = log_Ls[swap_chain+1]
        log_Ls[swap_chain+1] = log_Ls[swap_chain]
        log_Ls[swap_chain] = log_L_temp

        x0_temp = x0s[swap_chain+1]
        x0s[swap_chain+1] = x0s[swap_chain]
        x0s[swap_chain] = x0_temp

        FLI_temp = FLIs[swap_chain+1]
        FLIs[swap_chain+1] = FLIs[swap_chain]
        FLIs[swap_chain] = FLI_temp

        sample_temp = samples_cur[swap_chain+1,:].copy()
        samples_cur[swap_chain+1,:] = samples_cur[swap_chain,:]
        samples_cur[swap_chain,:] = sample_temp


    #loop through the chains and record the new samples and log_Ls
    samples[:,iii+1,:] = samples_cur

#    FLIs_new = []
#    x0s_new = []
#    for j in range(n_chain):
#        samples[j,iii+1,:] = np.copy(samples[swap_map[j],iii,:])
#        log_likelihood[j,iii+1] = log_likelihood[swap_map[j],iii]
#        FLIs_new.append(FLIs[swap_map[j]])
#        x0s_new.append(x0s[swap_map[j]])
#
#    FLIs[:] = FLIs_new
#    x0s[:] = List(x0s_new)


################################################################################
#
#DRAW FROM PRIOR JUMP
#
################################################################################
def do_draw_from_prior(n_chain, pta, samples, i, Ts, a_yes, a_no, x0s, FLIs, FastPrior, par_names, par_names_cw_ext, log_likelihood):
    #print("FISHER")
    for j in range(n_chain):
        samples_current = np.copy(samples[j,i,:])
        new_point = np.copy(samples[j,i,:])

        jump_select = np.random.randint(len(par_names_cw_ext))
        new_point[par_names.index(par_names_cw_ext[jump_select])] = pta.params[par_names.index(par_names_cw_ext[jump_select])].sample()

        x0s[j].update_params(new_point)

        log_L = FLIs[j].get_lnlikelihood(x0s[j])
        log_acc_ratio = log_L/Ts[j]
        log_acc_ratio += FastPrior.get_lnprior(new_point)
        log_acc_ratio += -log_likelihood[j,i]/Ts[j]
        log_acc_ratio += -FastPrior.get_lnprior(samples_current)

        if np.log(np.random.uniform())<=log_acc_ratio:
            samples[j,i+1,:] = np.copy(new_point)
            log_likelihood[j,i+1] = log_L
            a_yes[1,j]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            x0s[j].update_params(samples_current)
            log_likelihood[j,i+1] = log_likelihood[j,i]
            a_no[1,j]+=1


################################################################################
#
#CALCULATE FISHER DIAGONAL
#
################################################################################
def get_fisher_diagonal(T_chain, samples_current, par_names, par_names_cw_ext, x0, FLI_loc, epsilon=1e-2):
    dim = len(par_names)
    fisher_diag = np.zeros(dim)

    MMs_orig = FLI_loc.MMs.copy()
    NN_orig = FLI_loc.NN.copy()

    nn = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)


    #future locations
    mms = np.zeros(dim)
    pps = np.zeros(dim)
    dist_count = 0

    #calculate diagonal elements
    for i in range(dim):
        paramsPP = np.copy(samples_current)
        paramsMM = np.copy(samples_current)
        paramsPP[i] += 2*epsilon
        paramsMM[i] -= 2*epsilon

        if par_names[i] in par_names_cw_ext:
            #use fast likelihood
            x0.update_params(paramsPP)

            #pps[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
            pps[i] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            x0.update_params(paramsMM)

            #mms[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
            mms[i] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            #revert changes
            x0.update_params(samples_current)

        elif "_cw0_p_dist" in par_names[i]:
            x0.update_params(paramsPP)

            FLI_loc.update_pulsar_distance(x0, dist_count)
            pps[i] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            x0.update_params(paramsMM)

            FLI_loc.update_pulsar_distance(x0, dist_count)
            mms[i] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            #revert changes
            x0.update_params(samples_current)

            FLI_loc.MMs[dist_count] = MMs_orig[dist_count]
            FLI_loc.NN[dist_count] = NN_orig[dist_count]
            #TODO remove
            FLI_loc.cos_gwtheta = x0.cos_gwtheta
            FLI_loc.gwphi = x0.gwphi
            FLI_loc.log10_fgw = x0.log10_fgw
            FLI_loc.log10_mc = x0.log10_mc#

            dist_count += 1

        else:
            #must be one of the intrinsic parameters
            x0.update_params(paramsPP)

            FLI_loc.update_intrinsic_params(x0)
            pps[i] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            x0.update_params(paramsMM)

            FLI_loc.update_intrinsic_params(x0)
            mms[i] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

            #revert changes
            x0.update_params(samples_current)

            FLI_loc.MMs[:] = MMs_orig
            FLI_loc.NN[:] = NN_orig

            FLI_loc.cos_gwtheta = x0.cos_gwtheta
            FLI_loc.gwphi = x0.gwphi
            FLI_loc.log10_fgw = x0.log10_fgw
            FLI_loc.log10_mc = x0.log10_mc#

    #calculate diagonal elements of the Hessian from a central finite element scheme
    #note the minus sign compared to the regular Hessian
    fisher_diag = -(pps - 2.0*nn + mms)/(4.0*epsilon*epsilon)

    #correct for the given temperature of the chain
    fisher_diag = fisher_diag/T_chain

    #filer out nans and negative values - set them to 1.0 which will result in
    Fisher_diag = np.where(np.isfinite(fisher_diag), fisher_diag, 1.0)
    FISHER_diag = np.where(fisher_diag>0.0, Fisher_diag, 1.0)

    #filter values smaller than 4 and set those to 4 -- Neil's trick -- effectively not allow jump Gaussian stds larger than 0.5=1/sqrt(4)
    eig_limit = 4.0
    W = np.where(FISHER_diag>eig_limit, FISHER_diag, eig_limit)

    return 1/np.sqrt(W)

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

#TODO use this in all cosine proposals
@njit()
def reflect_cosines(cos_in,angle_in,rotfac=np.pi,modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
        which requires also rotating the signal by rotfac each time, then mod the angle by modfac"""
    if cos_in < -1.:
        cos_in = -1.+(-(cos_in+1.))%4
        angle_in += rotfac
        #if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
    if cos_in > 1.:
        cos_in = 1.-(cos_in-1.)%4
        angle_in += rotfac
    if cos_in < -1.:
        cos_in = -1.+(-(cos_in+1.))%4
        angle_in += rotfac
    angle_in = angle_in%modfac
    return cos_in,angle_in
