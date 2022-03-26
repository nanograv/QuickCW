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
#config.THREADING_LAYER = 'tbb'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

from numba import jit,njit,prange,objmode
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
#@profile
def QuickCW(N, T_max, n_chain, psrs, noise_json=None, n_status_update=100, n_int_block=1000, save_every_n=10_000, thin=10, samples_precision=np.single, savefile=None, save_first_n_chains=1, n_update_fisher=100_000, T_ladder=None, freq_bounds=[3.5e-9, 1e-7]):
    #freq = 1e-8
    #safety checks on input variables
    assert n_int_block%2==0 and n_int_block>=4 #need to have n_int block>=4 a multiple of 2
    #in order to always do at least n*(1 extrinsic+1 pt swap)+(1 intrinsic+1 pt swaps)
    assert save_every_n%n_int_block == 0 #or we won't save
    assert n_update_fisher%n_int_block == 0 #or we won't update fisher
    assert int(N/(n_status_update))%n_int_block == 0 #or we won't print status updates
    assert N%save_every_n == 0 #or we won't save a complete block
    assert N%n_int_block == 0 #or we won't execute the right number of blocks
    assert save_first_n_chains <= n_chain #or we would try to save more chains than we have

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
    #log10_fgw = parameter.LinearExp(np.log10(3.5e-9), -7.0)('0_log10_fgw')
    log10_fgw = parameter.Uniform(np.log10(freq_bounds[0]), np.log10(freq_bounds[1]))('0_log10_fgw')

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

    #log10_h = parameter.Uniform(-18, -11)('0_log10_h')
    log10_h = parameter.LinearExp(-18, -11)('0_log10_h')

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
    s = ef + eq + ec + rn + cw + tm
    #s = ef + eq + ec + cw + tm
    #s = ef + cw + tm

    models = [s(psr) for psr in psrs]

    pta = signal_base.PTA(models)

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)

    #print(noisedict)
    pta.set_default_params(noisedict)

    print(pta.summary())
    print(pta.params)

    FastPrior = CWFastPrior.FastPrior(pta)
    print(FastPrior.uniform_par_ids)
    FPI = FastPriorInfo(FastPrior.uniform_par_ids, FastPrior.uniform_lows, FastPrior.uniform_highs,
                        FastPrior.lin_exp_par_ids, FastPrior.lin_exp_lows, FastPrior.lin_exp_highs,
                        FastPrior.normal_par_ids, FastPrior.normal_mus, FastPrior.normal_sigs)

    par_names = List(pta.param_names)
    par_names_cw = List(['0_cos_gwtheta', '0_cos_inc', '0_gwphi', '0_log10_fgw', '0_log10_h',
                         '0_log10_mc', '0_phase0', '0_psi'])
    par_names_cw_ext = List(['0_cos_inc', '0_log10_h', '0_phase0', '0_psi'])
    par_names_cw_int = List(['0_cos_gwtheta', '0_gwphi', '0_log10_fgw', '0_log10_mc'])

    par_names_noise = []

    par_inds_cw_p_phase_ext = np.zeros(len(pta.pulsars),dtype=np.int64)
    par_inds_cw_p_dist_int = np.zeros(len(pta.pulsars),dtype=np.int64)

    for i,psr in enumerate(pta.pulsars):
        par_inds_cw_p_dist_int[i] = len(par_names_cw)
        par_names_cw.append(psr + "_cw0_p_dist")
        par_inds_cw_p_phase_ext[i] = len(par_names_cw)
        par_names_cw.append(psr + "_cw0_p_phase")
        par_names_cw_ext.append(psr + "_cw0_p_phase")
        par_names_cw_int.append(psr + "_cw0_p_dist")
        par_names_noise.append(psr + "_red_noise_gamma")
        par_names_noise.append(psr + "_red_noise_log10_A")

    n_par_tot = len(par_names)

    cw_ext_lows = []
    cw_ext_highs = []
    for par in [pta.params[iii] for iii in [par_names.index(ppp) for ppp in par_names_cw_ext]]:
        print(par)
        cw_ext_lows.append(float(par._typename.split('=')[1].split(',')[0]))
        cw_ext_highs.append(float(par._typename.split('=')[2][:-1]))
    
    cw_ext_lows = np.array(cw_ext_lows)
    cw_ext_highs = np.array(cw_ext_highs)
    print(cw_ext_lows)
    print(cw_ext_highs)

    if T_ladder is None:
        #using geometric spacing
        c = T_max**(1.0/(n_chain-1))
        Ts = c**np.arange(n_chain)
        print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\nTemperature ladder is:\n".format(n_chain,c),Ts)
    else:
        Ts = np.array(T_ladder)
        n_chain = Ts.size
        print("Using {0} temperature chains with custom spacing: ".format(n_chain),Ts)

    #set up samples array
    samples = np.zeros((n_chain, save_every_n+1, len(par_names)))

    #set up log_likelihood array
    log_likelihood = np.zeros((n_chain,save_every_n+1))

    print("Setting up first sample")
    for j in range(n_chain):
        #samples[j,0,:] = np.array([par.sample() for par in pta.params])
        samples[j,0,:] = np.array([CWFastPrior.get_sample_helper(i, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                    FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                    FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs) for i in range(len(par_names))])

        #set non-external parameters to injected for testing
        #samples[j,0,par_names.index('0_cos_gwtheta')] = np.cos(np.pi/3.0)
        #samples[j,0,par_names.index('0_gwphi')] = 4.5
        #samples[j,0,par_names.index('0_log10_fgw')] = np.log10(2e-8)
        #samples[j,0,par_names.index('0_log10_mc')] = np.log10(5e9)
        for psr in pta.pulsars:
            #samples[j,0,par_names.index(psr + "_cw0_p_dist")] = 0.0
            samples[j,0,par_names.index(psr + "_red_noise_gamma")] = noisedict[psr + "_red_noise_gamma"]
            samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = noisedict[psr + "_red_noise_log10_A"]

        #also set external parameters for further testing
        #samples[j,0,par_names.index("0_cos_inc")] = np.cos(1.0)
        #samples[j,0,par_names.index("0_log10_h")] = np.log10(1e-15)
        #samples[j,0,par_names.index("0_log10_h")] = np.log10(5e-15)
        #samples[j,0,par_names.index("0_phase0")] = 1.0
        #samples[j,0,par_names.index("0_psi")] = 1.0
        p_phases = [2.6438308,3.2279381,2.9511881,5.3586592,1.0639523,
                    2.1564047,1.1287014,5.9545189,4.3189053,1.3181107,
                    0.1205947,1.1594364,3.5189818,5.8613215,3.6653746,
                    0.0653161,4.3756635,2.2423111,4.5429403,5.1370920,
                    1.9794586,1.0159356,2.9529407,1.7553771,1.5110336,
                    4.0558141,1.6855663,3.5614665,4.1527070,5.2239841,
                    4.4504891,4.8126553,3.6622998,4.4647441,2.8561429,
                    0.6874573,0.3762146,1.6691351,0.8147172,0.3051969,
                    1.6177042,2.8609930,5.0392969,0.3359030,1.0489710]
        #for ii, psr in enumerate(pta.pulsars):
        #    samples[j,0,par_names.index(psr + "_cw0_p_phase")] = p_phases[ii]

    #set up master object for creating fast likelihoods
    x0_swap = CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[j,0],par_names,par_names_cw_ext)
    flm = CWFastLikelihoodNumba.FastLikeMaster(psrs,pta,dict(zip(par_names, samples[j, 0, :])),x0_swap)
    FLI_swap = flm.get_new_FastLike(x0_swap, dict(zip(par_names, samples[0, 0, :])))

    #set up fast likelihoods
    x0s = List([])
    FLIs  = List([])
    for j in range(n_chain):
        x0s.append( CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[j,0],par_names,par_names_cw_ext))
        FLIs.append(flm.get_new_FastLike(x0s[j], dict(zip(par_names, samples[j, 0, :]))))

    #make extra x0s to help parallelizing MTMCMC updates
    x0_extras = List([])
    for k in range(50):
        x0_extras.append(CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[0,0],par_names,par_names_cw_ext))

    #calculate the diagonal elements of the fisher matrix
    fisher_diag = np.ones((n_chain, len(par_names)))
    for j in range(n_chain):
        fisher_diag[j,:] = get_fisher_diagonal(samples[j,0,:], par_names, par_names_cw_ext, par_names_noise, x0_swap, flm, FLI_swap)
        print(fisher_diag[j,:])

    #calculate RN fisher eigenvectors (using offdiagonals as well)
    eig_rn = np.broadcast_to(np.eye(2)*0.5, (n_chain, len(pta.pulsars), 2, 2) ).copy()
    print("Calculating RN fishers")
    for j in range(n_chain):
        for i in range(len(pta.pulsars)):
            if j==0:
                print(par_names_noise[2*i:2*(i+1)])
            rn_eigvec = get_fisher_eigenvectors(samples[j,0,:], par_names, par_names_noise[2*i:2*(i+1)], pta)
            if np.all(rn_eigvec):
                eig_rn[j,i,:,:] = rn_eigvec[:,:]
            if j==0:
                print(rn_eigvec)

    #calculate common morphological fisher eigenvectors (using offdiagonals as well)
    eig_common = np.broadcast_to(np.eye(4)*0.5, (n_chain, 4, 4) ).copy()
    print("Calculating sky location/frequency/chirp mass fishers")
    for j in range(n_chain):
        common_eigvec = get_fisher_eigenvectors(samples[j,0,:], par_names, par_names_cw_int[:4], pta)
        if np.all(common_eigvec):
            eig_common[j,:,:] = common_eigvec[:,:]
        if j==0:
            print(common_eigvec)
    
    #set up differencial evolution
    de_history = np.zeros((n_chain, cm.de_history_size, len(par_names)))
    for j in range(n_chain):
        for i in range(cm.de_history_size):
            #de_history[j,i,:] = np.array([par.sample() for par in pta.params])
            de_history[j,i,:] = np.array([CWFastPrior.get_sample_helper(i, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                           FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                           FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs) for i in range(len(par_names))])

    #setting up arrays to record acceptance and swaps
    a_yes = np.zeros((20,n_chain),dtype=np.int64)
    a_no = np.zeros((20,n_chain),dtype=np.int64)

    acc_fraction = a_yes/(a_no+a_yes)

    #list to hold at what step delayed rejection stage are accepted/rejected
    n_dr_delays = []

    #printing info about initial parameters
    for j in range(n_chain):
        print("j="+str(j))
        print(samples[j,0,:])
        #log_likelihood[j,0] = pta.get_lnlikelihood(samples[j,0,:])
        log_likelihood[j,0] = FLIs[j].get_lnlikelihood(x0s[j])
        print("log_likelihood="+str(log_likelihood[j,0]))
        print("log_prior="+str(pta.get_lnprior(samples[j,0,:])))
        print("log_prior="+str(FastPrior.get_lnprior(samples[j,0,:])))

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
        itrn = i*n_int_block #index overall
        itrb = itrn%save_every_n #index within the block of saved values
        if itrb==0 and i!=0:
            acc_fraction = a_yes/(a_no+a_yes)
            #np.savez(savefile, samples=samples[0,:i*n_int_block,:], par_names=par_names, acc_fraction=acc_fraction, log_likelihood=log_likelihood[:,:i*n_int_block])
            if savefile is not None:
                if itrn>save_every_n:
                    print("Append to HDF5 file...")
                    with h5py.File(savefile, 'a') as f:
                        f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                        f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:save_first_n_chains,:-1:thin,:]
                        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                        f['acc_fraction'][...] = np.copy(acc_fraction)
                        f['fisher_diag'][...] = np.copy(fisher_diag)
                else:
                    print("Create HDF5 file...")
                    with h5py.File(savefile, 'w') as f:
                        f.create_dataset('samples_cold', data=samples[:save_first_n_chains,:-1:thin,:], dtype=samples_precision, compression="gzip", chunks=True, maxshape=(save_first_n_chains,int(N/thin),samples.shape[2]))
                        f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape=(samples.shape[0],int(N/thin)))
                        f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
                        f.create_dataset('acc_fraction', data=acc_fraction)
                        f.create_dataset('fisher_diag', data=fisher_diag)
            #clear out log_likelihood and samples arrays
            samples_now = samples[:,-1,:]
            log_likelihood_now = log_likelihood[:,-1]
            samples = np.zeros((n_chain, save_every_n+1, len(par_names)))
            log_likelihood = np.zeros((n_chain,save_every_n+1))
            samples[:,0,:] = samples_now
            log_likelihood[:,0] = log_likelihood_now
        if itrn%(N//n_status_update)==0:
            acc_fraction = a_yes/(a_no+a_yes)
            print('Progress: {0:2.2f}% '.format(itrn/N*100) +
                        'Acceptance fraction #columns: chain number; rows: proposal type (for morphological: w/o and w/ projection perturbation) (dist-prior, dist-DE, dist-fisher, RN-prior, RN-DE, RN-fisher, common-prior, common-DE, common-fisher, PT, proj):')
                        #'Acceptance fraction #columns: chain number; rows: proposal type (cos_gwtheta, gwphi, fgw, mc, p_dists, RN, PT, all ext):')
                        #'Acceptance fraction #columns: chain number; rows: proposal type (cos_gwtheta, cos_inc, gwphi, fgw, h, mc, phase0, psi, p_phases, p_dists, PT, all ext):')
            t_itr = perf_counter()
            print('at t= '+str(t_itr-ti_loop)+'s')
            print(acc_fraction)
            #print(acc_fraction[[0,2,3,5,9,10,11,12],:])
            #print(acc_fraction[:,0])
            #print(acc_fraction[:,-1])
            print("New log_L=", str(FLIs[0].get_lnlikelihood(x0s[0])))#,FLIs[0].resres,FLIs[0].logdet,FLIs[0].pos,FLIs[0].pdist,FLIs[0].NN,FLIs[0].MMs)))
            #print("Old log_L=", str(pta.get_lnlikelihood(samples[0,itrb,:])))
            #print("MT accepted w/o projection update:", len(np.where(np.array(index_yes)==0)[0]))
            #print("MT rejected w/o projection update:", len(np.where(np.array(index_no)==0)[0]))
            #print("MT accepted w/ projection update:", len(np.where(np.array(index_yes)>0)[0]))
            #print("MT rejected w/ projection update:", len(np.where(np.array(index_no)>0)[0]))
            #print("Mean DR delay:", str(np.nanmean(np.array(n_dr_delays)[np.where(np.array(n_dr_delays)>0)])))
            #n_dr_delays_pos = np.array(n_dr_delays)[np.where(np.array(n_dr_delays)>0)]
            #if n_dr_delays_pos.size>0:
            #    print("DR delay percentiles (10,50,90):", np.percentile(n_dr_delays_pos, 10), np.percentile(n_dr_delays_pos, 50), np.percentile(n_dr_delays_pos, 90))
            #    print("DR accepted at step #0:", len(np.where(np.array(n_dr_delays)==0)[0]))
            #    print("DR accepted at later step:", len(np.where(np.array(n_dr_delays)>0)[0]))
            #    print("DR not accepted:", len(np.where(np.isnan(np.array(n_dr_delays)))[0]))

        #always do pt steps in extrinsic
        do_extrinsic_block(n_chain, samples, itrb, Ts, x0s, FLIs, FPI, len(par_names), len(par_names_cw_ext), log_likelihood, n_int_block-2, fisher_diag, a_yes, a_no, cw_ext_lows, cw_ext_highs)
        #update intrinsic parameters once a block
        #FLI_swap = do_intrinsic_update_dr(n_chain, psrs, pta, samples, itrb+n_int_block-2, Ts, a_yes_counts, a_no_counts, x0s, FLIs, FPI, par_names, par_names_cw_int, par_names_noise, len(par_names_cw_ext), log_likelihood, fisher_diag, flm, FLI_swap, de_history, n_dr_delays)
        FLI_swap = do_intrinsic_update_mt(n_chain, psrs, pta, samples, itrb+n_int_block-2, Ts, a_yes, a_no, x0s, x0_extras, FLIs, FPI, par_names, par_names_cw_int, par_names_noise, len(par_names_cw_ext), log_likelihood, fisher_diag, eig_rn, eig_common, flm, FLI_swap, de_history, cw_ext_lows, cw_ext_highs)
        do_pt_swap(n_chain, samples, itrb+n_int_block-1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, fisher_diag)

        #update de history array
        for j in range(n_chain):
            de_history[j,i%cm.de_history_size] = np.copy(samples[j,itrb,:])

        if itrn%n_update_fisher==0 and i!=0:
            print("Updating Fisher diagonals")
            for j in range(n_chain):
                #compute fisher matrix at random recent points in the posterior
                fisher_diag[j,:] = get_fisher_diagonal(samples[j,np.random.randint(itrb+n_int_block+1),:], par_names, par_names_cw_ext, par_names_noise, x0_swap, flm, FLI_swap)
            if itrn%(n_update_fisher*10)==0:
                    for j in range(n_chain):
                        for jj in range(len(pta.pulsars)):
                            rn_eigvec = get_fisher_eigenvectors(samples[j,np.random.randint(itrb+n_int_block+1),:], par_names, par_names_noise[2*jj:2*(jj+1)], pta)
                            if np.all(rn_eigvec):
                                eig_rn[j,jj,:,:] = rn_eigvec[:,:]
                        common_eigvec = get_fisher_eigenvectors(samples[j,np.random.randint(itrb+n_int_block+1),:], par_names, par_names_cw_int[:4], pta)
                        if np.all(common_eigvec):
                            eig_common[j,:,:] = common_eigvec[:,:]

    acc_fraction = a_yes/(a_no+a_yes)
    print("Append to HDF5 file...")
    if savefile is not None:
        with h5py.File(savefile, 'a') as f:
            f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
            f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:save_first_n_chains,:-1:thin,:]
            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
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
#version using multiple try mcmc (based on Table 6 of https://vixra.org/pdf/1712.0244v3.pdf)
#@profile
def do_intrinsic_update_mt(n_chain, psrs, pta, samples, itrb, Ts, a_yes, a_no, x0s, x0_extras, FLIs, FPI, par_names, par_names_cw_int, par_names_noise, n_par_ext, log_likelihood, fisher_diag, eig_rn, eig_common, flm, FLI_swap, de_history, cw_ext_lows, cw_ext_highs):
    #print("EXT")
    for j in range(n_chain):
        assert np.isclose(FLIs[j].cos_gwtheta, x0s[j].cos_gwtheta)
        assert np.isclose(FLIs[j].gwphi, x0s[j].gwphi)
        assert np.isclose(FLIs[j].log10_fgw, x0s[j].log10_fgw)
        assert np.isclose(FLIs[j].log10_mc, x0s[j].log10_mc)
        assert np.allclose(FLIs[j].rn_gammas, x0s[j].rn_gammas)
        assert np.allclose(FLIs[j].rn_log10_As, x0s[j].rn_log10_As)
        #save MMs and NN so we can revert them if the chain is rejected
        MMs_save = FLIs[j].MMs.copy()
        NN_save = FLIs[j].NN.copy()
        
        samples_current = np.copy(samples[j,itrb,:])
        
        total_weight = cm.dist_jump_weight + cm.rn_jump_weight + cm.common_jump_weight
        which_jump = np.random.choice(3, p=[cm.dist_jump_weight/total_weight,
                                            cm.rn_jump_weight/total_weight,
                                            cm.common_jump_weight/total_weight])
        #if j==0:
        #    print("which_jump = ", str(which_jump))
        if which_jump==0: #update psr distances
            n_jump_loc = cm.n_dist_main
            idx_choose_psr = np.random.choice(x0s[j].Npsr,cm.n_dist_main,replace=False)
            #idx_choose_psr[0] = pta.pulsars.index(par_names_cw_int[jump_select][:-11])
            idx_choose = x0s[j].idx_dists[idx_choose_psr]
            scaling = 2.38/np.sqrt(n_jump_loc)
            #scaling = 1.0
            #scaling = 0.5
        elif which_jump==1: #update per psr RN
            #n_jump_loc = cm.n_noise_main*2 #2 parameters per pulsar
            n_jump_loc = 2*len(psrs)
            #idx_choose_psr = np.random.randint(0,x0s[j].Npsr,cm.n_noise_main)
            idx_choose_psr = list(range(len(psrs)))
            idx_choose = np.concatenate((x0s[j].idx_rn_gammas,x0s[j].idx_rn_log10_As))
            scaling = 2.38/np.sqrt(n_jump_loc/2)
            #scaling = 1/np.sqrt(n_jump_loc)
        elif which_jump==2: #update common intrinsic parameters (chirp mass, frequency, sky location[2])
            n_jump_loc = 4# 2+cm.n_dist_extra
            idx_choose = np.array([par_names.index(par_names_cw_int[itrk]) for itrk in range(4)])
            scaling = 2.38/np.sqrt(n_jump_loc)
            #scaling = 1.0
            #scaling = 0.5

        #decide what kind of jump we do
        if which_jump==1: #RN jump --> don't do prior draws, only fisher and DE
            total_type_weight = cm.de_prob + cm.fisher_prob
            which_jump_type = np.random.choice(3, p=[0.0,
                                                 cm.de_prob/total_type_weight,
                                                 cm.fisher_prob/total_type_weight])
        else:
            if j==(n_chain-1): #hottest chain and not RN --> only do prior draws
                which_jump_type = 0
            else: #not hottest chain and not RN --> choose jump type based in default probabilities of them
                total_type_weight = cm.prior_draw_prob + cm.de_prob + cm.fisher_prob
                which_jump_type = np.random.choice(3, p=[cm.prior_draw_prob/total_type_weight,
                                                         cm.de_prob/total_type_weight,
                                                         cm.fisher_prob/total_type_weight])
        if which_jump_type==0: #do prior draw
            #print("Prior draw")            
            new_point = np.copy(samples_current)
            #new_point[idx_choose] = np.array([par.sample() for par in [pta.params[iii] for iii in idx_choose]])
            new_point[idx_choose] = np.array([CWFastPrior.get_sample_helper(iii, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                                 FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                                 FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs) for iii in idx_choose])
        elif which_jump_type==1: #do differential evolution step
            de_indices = np.random.choice(de_history.shape[1], size=2, replace=False)
            ndim = idx_choose.size
            alpha0 = 2.38/np.sqrt(2*ndim)
            alpha = np.random.normal(scale=cm.sigma_de)            

            x1 = np.copy(de_history[j,de_indices[0],idx_choose])
            x2 = np.copy(de_history[j,de_indices[1],idx_choose])
            
            new_point = np.copy(samples_current)
            new_point[idx_choose] += alpha0*(1+alpha)*(x1-x2)
            
            #big_jump_decide = uniform(0.0, 1.0, 1)
            #if big_jump_decide<0.1: #do big jump
            #    new_point[idx_choose] += (1+alpha)*(x1-x2)
            #else: #do smaller jump scaled by alpha0
            #    new_point[idx_choose] += alpha0*(1+alpha)*(x1-x2)
        elif which_jump_type==2: #do regular fisher jump
            if which_jump==1: #use RN eigenvectors
                which_eig = np.random.choice(2, size=int(n_jump_loc/2))
                jump = np.zeros(len(par_names))
                for ll in range(int(n_jump_loc/2)):
                    idx_psr = np.array([idx_choose[ll], idx_choose[which_eig.size+ll]])
                    jump[idx_psr] = np.copy(eig_rn[j,idx_choose_psr[ll],which_eig[ll],:])*np.random.normal(0., 1.)*scaling
                new_point = samples_current + jump
            elif which_jump==2:
                which_eig = np.random.choice(4, size=1)
                jump = np.zeros(len(par_names))
                jump[idx_choose] = np.copy(eig_common[j,which_eig,:])*np.random.normal(0., 1.)*scaling
                new_point = samples_current + jump
            else: #use diagonal fishers
                #fisher_diag_loc = scaling * np.sqrt(Ts[j])*fisher_diag[j][idx_choose]
                fisher_diag_loc = scaling * fisher_diag[j][idx_choose]
                jump = np.zeros(len(par_names))
                jump[idx_choose] += fisher_diag_loc*np.random.normal(0.,1.,n_jump_loc)
                new_point = samples_current + jump

        #TODO check wrapping is working right
        new_point = correct_intrinsic(new_point,x0s[j])

        if which_jump==0: #update psr distances
            x0s[j].update_params(new_point)
            FLIs[j].update_pulsar_distances(x0s[j], idx_choose_psr)
        elif which_jump==1: #update per psr RN
            x0s[j].update_params(new_point)
            flm.recompute_FastLike(FLI_swap,x0s[j],dict(zip(par_names, new_point)))
        elif which_jump==2: #update common intrinsic parameters (chirp mass, frequency, sky location[2])
            x0s[j].update_params(new_point)
            FLIs[j].update_intrinsic_params(x0s[j])

        #save current MM and NN
        MMs_new = FLIs[j].MMs.copy()
        NN_new = FLIs[j].NN.copy()
        
        #check the maximum toa is not such that the source has already merged, and if so automatically reject the proposal
        w0 = np.pi * 10.0**x0s[j].log10_fgw
        mc = 10.0**x0s[j].log10_mc * const.Tsun

        if (1. - 256./5. * mc * const.Tsun**(5./3.) * w0**(8./3.) * (FLIs[j].max_toa - cm.tref)) < 0:
            #set these so that the step is rejected
            acc_ratio = -1
            acc_decide = 0.
            print("Rejected due to too fast evolution.")
        else:
            log_prior_old = CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                            FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                            FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_posterior_old = log_likelihood[j,itrb]/Ts[j] + log_prior_old
            
            #do multiple try MCMC step with random draws of projection parameters
            random_draws_from_prior = np.random.uniform(np.repeat(cw_ext_lows,cm.n_multi_try), np.repeat(cw_ext_highs,cm.n_multi_try))
            random_normals = np.random.normal(0.,1.,len(cw_ext_lows)*cm.n_multi_try)

            tries, mt_weights, log_Ls = get_mt_weights(new_point, j, FPI, x0s, x0_extras, which_jump, FLIs, FLI_swap, Ts,
                                                       log_posterior_old, cm.n_multi_try, random_draws_from_prior, random_normals, fisher_diag)
            
            #not sure why but still can get nans here...
            mt_weights = np.where(np.isnan(mt_weights), 0.0, mt_weights)

            if np.sum(mt_weights)==0.0:
                acc_ratio = -1
                acc_decide = 0.0
                #print("Something weird happened. Here's the new point we tried:")
                #print(new_point)
                #print(j)
                #print(log_Ls[0])
                #print(log_likelihood[j,itrb])
                #print(CWFastPrior.get_lnprior_helper(tries[0,:], FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                #                                                 FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs))
            else:
                #print(mt_weights)
                chosen_trial = np.random.choice(cm.n_multi_try, p=mt_weights/np.sum(mt_weights))

                if which_jump != 1: #need to set back FLIs to old state to calculate likelihoods at reference points
                    x0s[j].update_params(samples_current)
                    FLIs[j].MMs = MMs_save
                    FLIs[j].NN = NN_save
                    FLIs[j].cos_gwtheta = x0s[j].cos_gwtheta
                    FLIs[j].gwphi = x0s[j].gwphi
                    FLIs[j].log10_fgw = x0s[j].log10_fgw
                    FLIs[j].log10_mc = x0s[j].log10_mc
                    #FLIs[j].rn_gammas = x0s[j].rn_gammas.copy()
                    #FLIs[j].rn_log10_As = x0s[j].rn_log10_As.copy()

                ref_tries, ref_mt_weights = get_ref_mt_weights(samples_current, j, FPI, x0s, x0_extras, which_jump, FLIs, FLI_swap, Ts,
                                                               log_posterior_old, cm.n_multi_try, random_draws_from_prior, random_normals, fisher_diag, tries, chosen_trial)

                ref_mt_weights[chosen_trial] = 1.0

                acc_ratio = np.sum(mt_weights)/np.sum(ref_mt_weights)
            
                acc_decide = uniform(0.0, 1.0, 1)

        #if j==0 and np.sum(mt_weights)>0:
        #    print('-'*30)
        #    print(tries[0,:])
        #    print(tries[1,:])
        #    print(ref_tries[0,:])
        #    print(ref_tries[1,:])
        #    print(mt_weights)
        #    print(ref_mt_weights)
        #    print("Chosen trial:",str(chosen_trial))
        #    print(mt_weights[chosen_trial])
        #    print(ref_mt_weights[chosen_trial])
        #    print(acc_ratio)
        #    print(acc_decide)
        if acc_decide<=acc_ratio:
            #if j==0:
            #    print("yay")
            #print("yay")
            x0s[j].update_params(tries[chosen_trial,:])

            samples[j,itrb+1,:] = tries[chosen_trial,:]

            if which_jump==1:
                #swap the temporary FLI for the old one
                FLI_temp = FLIs[j]
                FLIs[j] = FLI_swap
                FLI_swap = FLI_temp
            else:
                #since we reverted to old ones for calculating the reference ponint likelihoods, revert that
                FLIs[j].MMs = MMs_new
                FLIs[j].NN = NN_new
                FLIs[j].cos_gwtheta = x0s[j].cos_gwtheta
                FLIs[j].gwphi = x0s[j].gwphi
                FLIs[j].log10_fgw = x0s[j].log10_fgw
                FLIs[j].log10_mc = x0s[j].log10_mc
                #FLIs[j].rn_gammas = x0s[j].rn_gammas.copy()
                #FLIs[j].rn_log10_As = x0s[j].rn_log10_As.copy()

            log_likelihood[j,itrb+1] = log_Ls[chosen_trial]
            if chosen_trial==0:
                a_yes[6*which_jump+2*which_jump_type,j] += 1
            else:
                a_yes[6*which_jump+2*which_jump_type+1,j] += 1
        else:
            #if j==0:
            #    print("Nay, REJECT!")
            #print("Nay, REJECT!")
            samples[j,itrb+1,:] = np.copy(samples_current)

            log_likelihood[j,itrb+1] = log_likelihood[j,itrb]

            #Add to both elements of a_no, so we can get acceptance over total jumps w/ and w/o projection perturbation
            a_no[6*which_jump+2*which_jump_type,j] += 1
            a_no[6*which_jump+2*which_jump_type+1,j] += 1
            
            x0s[j].update_params(samples_current)

            if not which_jump==1:
                #don't needs to do anything if which_jump==1 because we didn't update FLIs[j] at all,
                #and FLI_swap will just be completely overwritten next time it is used

                #revert the changes to FastLs
                FLIs[j].MMs = MMs_save
                FLIs[j].NN = NN_save
                #revert the safety tracking parameters that were altered by update_intrinsic
                FLIs[j].cos_gwtheta = x0s[j].cos_gwtheta
                FLIs[j].gwphi = x0s[j].gwphi
                FLIs[j].log10_fgw = x0s[j].log10_fgw
                FLIs[j].log10_mc = x0s[j].log10_mc#
                #FLIs[j].rn_gammas = x0s[j].rn_gammas.copy()
                #FLIs[j].rn_log10_As = x0s[j].rn_log10_As.copy()
                #print("sky location, frequency, or chirp mass update")

        assert np.isclose(FLIs[j].cos_gwtheta, x0s[j].cos_gwtheta)
        assert np.isclose(FLIs[j].gwphi, x0s[j].gwphi)
        assert np.isclose(FLIs[j].log10_fgw, x0s[j].log10_fgw)
        assert np.isclose(FLIs[j].log10_mc, x0s[j].log10_mc)
        assert np.allclose(FLIs[j].rn_gammas, x0s[j].rn_gammas)
        assert np.allclose(FLIs[j].rn_log10_As, x0s[j].rn_log10_As)

    return FLI_swap

@njit(parallel=True, fastmath=True)
def get_mt_weights(new_point, j, FPI, x0s, x0_extras, which_jump, FLIs, FLI_swap, Ts, log_posterior_old, n_multi_try, random_draws_from_prior, random_normals, fisher_diag):
    """Helper function to quickly return multiple tries and their likelihoods fo MTMCMC"""
    #set up needed arrays
    tries = np.zeros((n_multi_try, new_point.shape[0]))
    mt_weights = np.zeros(n_multi_try)
    log_Ls = np.zeros(n_multi_try)

    jump_idx = x0s[j].idx_cw_ext

    loopsize = len(x0_extras)

    #print("-"*30)
    #print(FLI_swap.cos_gwtheta)
    #print(FLIs[j].cos_gwtheta)
    
    #get mt_weights --------------------------------------------------------------------------------------------------------
    for KK in prange(loopsize):
        for kk in range(int(n_multi_try/loopsize)):
            itrkk = KK*int(n_multi_try/loopsize)+kk
            tries[itrkk,:] = np.copy(new_point)
            
            if itrkk!=0: #leave projection parameters for the 0th trial as they were - perturb for the rest
                for ii,ll in enumerate(jump_idx):
                    fisher_loc = fisher_diag[j][ll]
                    if fisher_loc<0.5: #not maxed out fisher --> do fisher update
                        tries[itrkk,ll] += fisher_loc*random_normals[itrkk + n_multi_try*ii]
                    else: #parameters with maxed out fisher --> do prior draw
                        tries[itrkk,ll] = random_draws_from_prior[itrkk + n_multi_try*ii]

            tries[itrkk,:] = correct_extrinsic(tries[itrkk,:],x0_extras[KK])
            x0_extras[KK].update_params(tries[itrkk,:])
            #print(x0_extras[KK].cos_gwtheta)

            if which_jump == 1:
                log_Ls[itrkk] = FLI_swap.get_lnlikelihood(x0_extras[KK])
            else:
                log_Ls[itrkk] = FLIs[j].get_lnlikelihood(x0_extras[KK])
            log_prior_new = CWFastPrior.get_lnprior_helper(tries[itrkk,:], FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                           FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                           FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_posterior_new = log_Ls[itrkk]/Ts[j] + log_prior_new

            if log_prior_new>-np.inf:
                mt_weights[itrkk] = np.exp(log_posterior_new - log_posterior_old)
            else:
                mt_weights[itrkk] = 0.0
            
            #TODO: check actually why we get nans sometimes
            #this if loop avoids the situation where for some funky reason
            #log_posterior_new=nan, which gives errors below
            #not a problem in regular MCMCs, because a nan acceptance probability results in a rejection
            if np.isnan(mt_weights[itrkk]):
                mt_weights[itrkk] = 0.0

    return tries, mt_weights, log_Ls


@njit(parallel=True, fastmath=True)
def get_ref_mt_weights(samples_current, j, FPI, x0s, x0_extras, which_jump, FLIs, FLI_swap, Ts, log_posterior_old, n_multi_try, random_draws_from_prior, random_normals, fisher_diag, tries, chosen_trial):
    """Helper function to quickly return multiple tries and their likelihoods fo MTMCMC"""
    #set up needed arrays
    ref_tries = np.zeros((n_multi_try, samples_current.shape[0]))
    ref_mt_weights = np.zeros(n_multi_try)

    jump_idx = x0s[j].idx_cw_ext

    loopsize = len(x0_extras)

    #get ref_mt_weights ----------------------------------------------------------------------------------------------------
    for KK in prange(loopsize):
        for kk in range(int(n_multi_try/loopsize)):
            itrkk = KK*int(n_multi_try/loopsize)+kk
            ref_tries[itrkk,:] = np.copy(samples_current)

            if itrkk==0: #use exact same projection parameters as for chosen trial for 0th trial - perturb otherwise
                for ii,ll in enumerate(jump_idx):
                    ref_tries[itrkk,ll] = tries[chosen_trial,ll]
            else:
                for ii,ll in enumerate(jump_idx):
                    fisher_loc = fisher_diag[j][ll]
                    if fisher_loc<0.5: #not maxed out fisher --> do fisher update
                        jump = fisher_loc*random_normals[itrkk + n_multi_try*ii]
                        ref_tries[itrkk,ll] = tries[chosen_trial,ll] + jump
                    else: #parameters with maxed out fisher --> do prior draw
                        tries[itrkk,ll] = random_draws_from_prior[itrkk + n_multi_try*ii]
                        ref_tries[itrkk,ll] = random_draws_from_prior[itrkk + n_multi_try*ii]

            ref_tries[itrkk,:] = correct_extrinsic(ref_tries[itrkk,:],x0_extras[KK])
            x0_extras[KK].update_params(ref_tries[itrkk,:])

            log_L = FLIs[j].get_lnlikelihood(x0_extras[KK])
            log_prior_ref = CWFastPrior.get_lnprior_helper(ref_tries[itrkk,:], FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_posterior_ref = log_L/Ts[j] + log_prior_ref

            if log_prior_ref>-np.inf:
                ref_mt_weights[itrkk] = np.exp(log_posterior_ref - log_posterior_old)
            else:
                ref_mt_weights[itrkk] = 0.0
            
            #TODO: check actually why we get nans sometimes
            #this if loop avoids the situation where for some funky reason
            #log_posterior_new=nan, which gives errors below
            #not a problem in regular MCMCs, because a nan acceptance probability results in a rejection
            if np.isnan(ref_mt_weights[itrkk]):
                ref_mt_weights[itrkk] = 0.0

    return ref_tries, ref_mt_weights

@njit()
def correct_extrinsic(sample,x0):
    """correct extrinsic parameters for phases and cosines"""
    #TODO check these are the right parameters to be shifting
    sample[x0.idx_cos_inc],sample[x0.idx_psi] = reflect_cosines(sample[x0.idx_cos_inc],sample[x0.idx_psi],np.pi/2,np.pi)
    sample[x0.idx_phase0] = sample[x0.idx_phase0]%(2*np.pi)
    sample[x0.idx_phases] = sample[x0.idx_phases]%(2*np.pi)
    return sample

@njit()
def correct_intrinsic(sample,x0):
    """correct intrinsic parameters for phases and cosines"""
    #TODO check these are the right parameters to be shifting
    sample[x0.idx_cos_gwtheta],sample[x0.idx_gwphi] = reflect_cosines(sample[x0.idx_cos_gwtheta],sample[x0.idx_gwphi],np.pi,2*np.pi)
    #sample[x0.idx_rn_gammas] = np.abs(sample[x0.idx_rn_gammas])
    #sample[x0.idx_rn_gammas] = 7.0-np.abs(7.0-np.abs(sample[x0.idx_rn_gammas])) #making sure gamma is within 0.0 and 7.0
    #sample[x0.idx_rn_log10_As] = -11-np.abs((-20+np.abs(20+sample[x0.idx_rn_log10_As]))+11) #making sure log10_A is within -20 and -11
    for idx in x0.idx_rn_gammas:
        sample[idx] = reflect_into_range(sample[idx], 0.0, 7.0)
    for idx in x0.idx_rn_log10_As:
        sample[idx] = reflect_into_range(sample[idx], -20.0, -11.0)
    sample[x0.idx_log10_fgw] = reflect_into_range(sample[x0.idx_log10_fgw], np.log10(3.5e-9), -7.0)
    sample[x0.idx_log10_mc] = reflect_into_range(sample[x0.idx_log10_mc], 7.0, 10.0)
    
    return sample

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################
@njit(parallel=True)
def do_extrinsic_block(n_chain, samples, itrb, Ts, x0s, FLIs, FPI, n_par_tot, n_par_ext, log_likelihood, n_int_block, fisher_diag, a_yes, a_no, cw_ext_lows, cw_ext_highs):

    for k in range(0,n_int_block,2):
        for j in prange(0,n_chain):
            samples_current = samples[j,itrb+k,:]

            if k%10==0: #every 10th k (so every 5th jump) do a prior draw
                new_point = np.copy(samples_current)
                jump_idx = x0s[j].idx_cw_ext
                for ii, idx in enumerate(jump_idx):
                    new_point[idx] = uniform(cw_ext_lows[ii], cw_ext_highs[ii])
            else:
                jump = np.zeros(n_par_tot)
                jump_idx = x0s[j].idx_cw_ext
                jump[jump_idx] = 2.38/np.sqrt(n_par_ext)*np.sqrt(Ts[j])*fisher_diag[j][jump_idx]*np.random.normal(0.,1.,n_par_ext)
                new_point = samples_current + jump

            new_point = correct_extrinsic(new_point,x0s[j])

            x0s[j].update_params(new_point)

            log_L = FLIs[j].get_lnlikelihood(x0s[j])#FLIs[j].resres,FLIs[j].logdet,FLIs[j].pos,FLIs[j].pdist,FLIs[j].NN,FLIs[j].MMs)
            log_acc_ratio = log_L/Ts[j]
            log_acc_ratio += CWFastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                       FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                       FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
            log_acc_ratio += -log_likelihood[j,itrb+k]/Ts[j]
            log_acc_ratio += -CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
                                                                              FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
                                                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)

            acc_decide = np.log(uniform(0.0, 1.0, 1))
            if acc_decide<=log_acc_ratio:
                samples[j,itrb+k+1,:] = new_point
                log_likelihood[j,itrb+k+1] = log_L
                a_yes[cm.idx_full,j] += 1
            else:
                samples[j,itrb+k+1,:] = samples[j,itrb+k,:]
                log_likelihood[j,itrb+k+1] = log_likelihood[j,itrb+k]
                a_no[cm.idx_full,j] += 1

                x0s[j].update_params(samples_current)

        do_pt_swap(n_chain, samples, itrb+k+1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood,fisher_diag)


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
@njit()
def do_pt_swap(n_chain, samples, itrb, Ts, a_yes, a_no, x0s, FLIs, log_likelihood,fisher_diag):
    #print("PT")

    #print("PT")

    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,itrb])

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
            a_yes[cm.idx_PT,swap_chain] += 1
            #a_yes[0,swap_chain]+=1
        else:
            a_no[cm.idx_PT,swap_chain] += 1
            #a_no[0,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    FLIs_new = []
    x0s_new = []
    fisher_diag_new = np.zeros_like(fisher_diag)
    for j in range(n_chain):
        samples[j,itrb+1,:] = samples[swap_map[j],itrb,:]
        fisher_diag_new[j,:] = fisher_diag[swap_map[j],:]
        log_likelihood[j,itrb+1] = log_likelihood[swap_map[j],itrb]
        FLIs_new.append(FLIs[swap_map[j]])
        x0s_new.append(x0s[swap_map[j]])

    fisher_diag[:] = fisher_diag_new
    FLIs[:] = List(FLIs_new)
    x0s[:] = List(x0s_new)

def do_pt_swap_alt(n_chain, samples, itrb, Ts, a_yes, a_no, x0s, FLIs, log_likelihood,fisher_diag):
    """modification to swap routine that is easier to adapt to arbitrary swap proposals"""
    #print("PT")

    #print("PT")

    #set up map to help keep track of swaps
    #swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    #log_Ls = []
    #for j in range(n_chain):
    #    log_Ls.append(log_likelihood[j,iii])
    log_Ls = log_likelihood[:,itrb].copy()
    samples_cur = samples[:,itrb,:].copy()

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
    samples[:,itrb+1,:] = samples_cur

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


@njit()
def fisher_synthetic_FLI_helper(samples_fisher,paramsPP,paramsMM,x0_swap,FLI_swap,epsilon,default_sigma,MMs0,NN0,resres_array0,logdet_array0,MMsp,NNp,resres_arrayp,logdet_arrayp,MMsm,NNm,resres_arraym,logdet_arraym):
    """helper to construct synthetic likelihoods for each pulsar from input MMs and NNs one by one"""
    pps = np.zeros(x0_swap.Npsr)
    mms = np.zeros(x0_swap.Npsr)
    nns = np.zeros(x0_swap.Npsr)
    fisher_diag = np.zeros(x0_swap.Npsr)
    #isolate elements that change for maximum numerical accuracy
    FLI_swap.MMs[:] = 0.
    FLI_swap.NN[:] = 0.
    FLI_swap.resres_array[:] = 0.
    FLI_swap.logdet_array[:] = 0.
    FLI_swap.resres = 0.
    FLI_swap.logdet = 0.

    for ii in range(x0_swap.Npsr):
        x0_swap.update_params(samples_fisher)
        FLI_swap.MMs[ii] = MMs0[ii]
        FLI_swap.NN[ii] = NN0[ii]
        FLI_swap.resres_array[ii] = resres_array0[ii]
        FLI_swap.logdet_array[ii] = logdet_array0[ii]
        FLI_swap.resres = np.sum(FLI_swap.resres_array)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(FLI_swap.logdet_array)

        nns[ii] = FLI_swap.get_lnlikelihood(x0_swap)

        x0_swap.update_params(paramsPP)
        FLI_swap.MMs[ii] = MMsp[ii]
        FLI_swap.NN[ii] = NNp[ii]
        FLI_swap.resres_array[ii] = resres_arrayp[ii]
        FLI_swap.logdet_array[ii] = logdet_arrayp[ii]
        FLI_swap.resres = np.sum(FLI_swap.resres_array)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(FLI_swap.logdet_array)

        pps[ii] = FLI_swap.get_lnlikelihood(x0_swap)

        x0_swap.update_params(paramsMM)
        FLI_swap.MMs[ii] = MMsm[ii]
        FLI_swap.NN[ii] = NNm[ii]
        FLI_swap.resres_array[ii] = resres_arraym[ii]
        FLI_swap.logdet_array[ii] = logdet_arraym[ii]
        FLI_swap.resres = np.sum(FLI_swap.resres_array)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(FLI_swap.logdet_array)

        mms[ii] = FLI_swap.get_lnlikelihood(x0_swap)

        fisher_diag[ii] = -(pps[ii] - 2*nns[ii] + mms[ii])/(4*epsilon*epsilon)

        if np.isnan(fisher_diag[ii]) or fisher_diag[ii] <= 0. :
            fisher_diag[ii] = 1./default_sigma**2#1/cm.sigma_cw0_p_phase_default**2

        #reset elements to 0
        FLI_swap.MMs[ii] = 0.
        FLI_swap.NN[ii] = 0.
        FLI_swap.resres_array[ii] = 0.
        FLI_swap.logdet_array[ii] = 0.
        FLI_swap.resres = 0.
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(FLI_swap.logdet_array)

    #revert all elements to old values
    x0_swap.update_params(samples_fisher)
    FLI_swap.MMs[ii] = MMs0[ii]
    FLI_swap.NN[ii] = NN0[ii]
    FLI_swap.resres_array[ii] = resres_array0[ii]
    FLI_swap.logdet_array[ii] = logdet_array0[ii]
    FLI_swap.resres = np.sum(FLI_swap.resres_array)
    FLI_swap.logdet = FLI_swap.logdet_base + np.sum(FLI_swap.logdet_array)
    return pps,mms,nns,fisher_diag


################################################################################
#
#CALCULATE FISHER DIAGONAL
#
################################################################################
def get_fisher_diagonal(samples_fisher, par_names, par_names_cw_ext, par_names_noise, x0_swap, flm, FLI_swap):
    dim = len(par_names)
    fisher_diag = np.zeros(dim)

    #this does not change so can be set to 0 for improved numerical accuracy in derivatives
    logdet_base_old = FLI_swap.logdet_base
    FLI_swap.logdet_base = 0.
    #logdet_old = FLI_loc.logdet
    #resres_old = FLI_loc.resres

    #logdet_array_old = FLI_loc.logdet_array.copy()
    #resres_array_old = FLI_loc.resres_array.copy()

    #MMs_old = FLI_loc.MMs.copy()
    #NN_old = FLI_loc.NN.copy()


    x0_swap.update_params(samples_fisher)
    #we will update FLI_swap later to prevent having to do it twice

    #future locations
    mms = np.zeros(dim)
    pps = np.zeros(dim)
    nns = np.zeros(dim)

    if not cm.use_default_noise_sigma:
        #the noise parameters are very expensive to calculate individually so calculate them all en masse
        #flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, samples_fisher)))

        paramsPP_gamma = np.copy(samples_fisher)
        paramsMM_gamma = np.copy(samples_fisher)

        epsilon = cm.eps['red_noise_gamma']
        paramsPP_gamma[x0_swap.idx_rn_gammas] += 2*epsilon
        paramsMM_gamma[x0_swap.idx_rn_gammas] -= 2*epsilon

        x0_swap.update_params(paramsPP_gamma)
        flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsPP_gamma)))

        MMsp_gamma = FLI_swap.MMs.copy()
        NNp_gamma = FLI_swap.NN.copy()
        resres_arrayp_gamma = FLI_swap.resres_array.copy()
        logdet_arrayp_gamma = FLI_swap.logdet_array.copy()

        x0_swap.update_params(paramsMM_gamma)
        flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsMM_gamma)))

        MMsm_gamma = FLI_swap.MMs.copy()
        NNm_gamma = FLI_swap.NN.copy()
        resres_arraym_gamma = FLI_swap.resres_array.copy()
        logdet_arraym_gamma = FLI_swap.logdet_array.copy()

        paramsPP_log10_A = np.copy(samples_fisher)
        paramsMM_log10_A = np.copy(samples_fisher)

        epsilon = cm.eps['red_noise_log10_A']
        paramsPP_log10_A[x0_swap.idx_rn_log10_As] += 2*epsilon
        paramsMM_log10_A[x0_swap.idx_rn_log10_As] -= 2*epsilon

        x0_swap.update_params(paramsPP_log10_A)
        flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsPP_log10_A)))

        MMsp_log10_A = FLI_swap.MMs.copy()
        NNp_log10_A = FLI_swap.NN.copy()
        resres_arrayp_log10_A = FLI_swap.resres_array.copy()
        logdet_arrayp_log10_A = FLI_swap.logdet_array.copy()

        x0_swap.update_params(paramsMM_log10_A)
        flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsMM_log10_A)))

        MMsm_log10_A = FLI_swap.MMs.copy()
        NNm_log10_A = FLI_swap.NN.copy()
        resres_arraym_log10_A = FLI_swap.resres_array.copy()
        logdet_arraym_log10_A = FLI_swap.logdet_array.copy()

        #no need to revert FLI_swap because it is always overwritten completely when used
        x0_swap.update_params(samples_fisher)

    #put the reset here to avoid having to do it both before and after
    flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, samples_fisher)))

    MMs0 = FLI_swap.MMs.copy()
    NN0 = FLI_swap.NN.copy()
    resres_array0 = FLI_swap.resres_array.copy()
    logdet_array0 = FLI_swap.logdet_array.copy()

    #nn = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

    if not cm.use_default_noise_sigma:
        epsilon = cm.eps['red_noise_gamma']
        pps_loc,mms_loc,nns_loc,fisher_diag_loc = fisher_synthetic_FLI_helper(samples_fisher,paramsPP_gamma,paramsMM_gamma,x0_swap,FLI_swap,epsilon,cm.sigma_noise_default,MMs0,NN0,resres_array0,logdet_array0,MMsp_gamma,NNp_gamma,resres_arrayp_gamma,logdet_arrayp_gamma,MMsm_gamma,NNm_gamma,resres_arraym_gamma,logdet_arraym_gamma)
        pps[x0_swap.idx_rn_gammas] = pps_loc
        mms[x0_swap.idx_rn_gammas] = mms_loc
        nns[x0_swap.idx_rn_gammas] = nns_loc
        fisher_diag[x0_swap.idx_rn_gammas] = fisher_diag_loc

        epsilon = cm.eps['red_noise_log10_A']
        pps_loc,mms_loc,nns_loc,fisher_diag_loc = fisher_synthetic_FLI_helper(samples_fisher,paramsPP_log10_A,paramsMM_log10_A,x0_swap,FLI_swap,epsilon,cm.sigma_noise_default,MMs0,NN0,resres_array0,logdet_array0,MMsp_log10_A,NNp_log10_A,resres_arrayp_log10_A,logdet_arrayp_log10_A,MMsm_log10_A,NNm_log10_A,resres_arraym_log10_A,logdet_arraym_log10_A)
        pps[x0_swap.idx_rn_log10_As] = pps_loc
        mms[x0_swap.idx_rn_log10_As] = mms_loc
        nns[x0_swap.idx_rn_log10_As] = nns_loc
        fisher_diag[x0_swap.idx_rn_log10_As] = fisher_diag_loc

        #double check everything is reset
        FLI_swap.MMs[:] = MMs0
        FLI_swap.NN[:] = NN0
        FLI_swap.resres_array[:] = resres_array0
        FLI_swap.logdet_array[:] = logdet_array0
        FLI_swap.resres = np.sum(resres_array0)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

        x0_swap.update_params(samples_fisher)

    if not cm.use_default_cw0_p_sigma:
        #the distances and phases can be done as a block as well as the noises
        paramsPP = np.copy(samples_fisher)
        paramsMM = np.copy(samples_fisher)

        epsilon = cm.eps['cw0_p_dist']

        paramsPP[x0_swap.idx_dists] += 2*epsilon
        paramsMM[x0_swap.idx_dists] -= 2*epsilon

        #turn off all elements which do not vary with distance
        MMs1 = MMs0.copy()
        MMs1[:,:2,:2] = 0.
        NN1 = NN0.copy()
        NN1[:,:2] = 0.

        #use fast likelihood
        #note this does not update either logdet or resres so we can ignore that
        x0_swap.update_params(paramsPP)
        FLI_swap.update_pulsar_distances(x0_swap, np.arange(0,x0_swap.Npsr))

        MMsp = FLI_swap.MMs.copy()
        NNp = FLI_swap.NN.copy()
        MMsp[:,:2,:2] = 0.
        NNp[:,:2] = 0.

        x0_swap.update_params(paramsMM)
        FLI_swap.update_pulsar_distances(x0_swap, np.arange(0,x0_swap.Npsr))

        MMsm = FLI_swap.MMs.copy()
        NNm = FLI_swap.NN.copy()
        MMsm[:,:2,:2] = 0.
        NNm[:,:2] = 0.


        #resres and logdet contributions are constant for distance so ignore them
        rrad = np.zeros(x0_swap.Npsr)
        ldad = np.zeros(x0_swap.Npsr)

        pps_loc,mms_loc,nns_loc,fisher_diag_loc = fisher_synthetic_FLI_helper(samples_fisher,paramsPP,paramsMM,x0_swap,FLI_swap,epsilon,cm.sigma_cw0_p_dist_default,MMs1,NN1,rrad,ldad,MMsp,NNp,rrad,ldad,MMsm,NNm,rrad,ldad)
        pps[x0_swap.idx_dists] = pps_loc
        mms[x0_swap.idx_dists] = mms_loc
        nns[x0_swap.idx_dists] = nns_loc
        fisher_diag[x0_swap.idx_dists] = fisher_diag_loc

        #double check everything is reset
        FLI_swap.MMs[:] = MMs0
        FLI_swap.NN[:] = NN0
        FLI_swap.resres_array[:] = resres_array0
        FLI_swap.logdet_array[:] = logdet_array0
        FLI_swap.resres = np.sum(resres_array0)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

        x0_swap.update_params(samples_fisher)

    if not cm.use_default_cw0_p_sigma:
        #the distances and phases can be done as a block as well as the noises
        paramsPP = np.copy(samples_fisher)
        paramsMM = np.copy(samples_fisher)

        epsilon = cm.eps['cw0_p_phase']

        paramsPP[x0_swap.idx_phases] += 2*epsilon
        paramsMM[x0_swap.idx_phases] -= 2*epsilon

        #turn off all elements which do not vary with phase
        #MM and NN do not need to be updated at all for phases
        MMs1 = MMs0.copy()
        MMs1[:,:2,:2] = 0.
        NN1 = NN0.copy()
        NN1[:,:2] = 0.

        #resres and logdet contributions are constant for phase so ignore them
        rrad = np.zeros(x0_swap.Npsr)
        ldad = np.zeros(x0_swap.Npsr)

        pps_loc,mms_loc,nns_loc,fisher_diag_loc = fisher_synthetic_FLI_helper(samples_fisher,paramsPP,paramsMM,x0_swap,FLI_swap,epsilon,cm.sigma_cw0_p_phase_default,MMs1,NN1,rrad,ldad,MMs1,NN1,rrad,ldad,MMs1,NN1,rrad,ldad)
        pps[x0_swap.idx_phases] = pps_loc
        mms[x0_swap.idx_phases] = mms_loc
        nns[x0_swap.idx_phases] = nns_loc
        fisher_diag[x0_swap.idx_phases] = fisher_diag_loc

        #double check everything is reset
        FLI_swap.MMs[:] = MMs0
        FLI_swap.NN[:] = NN0
        FLI_swap.resres_array[:] = resres_array0
        FLI_swap.logdet_array[:] = logdet_array0
        FLI_swap.resres = np.sum(resres_array0)
        FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

        x0_swap.update_params(samples_fisher)


    assert np.all(fisher_diag>=0.)

    phase_count = 0

    #calculate diagonal elements
    for i in range(dim):
        paramsPP = np.copy(samples_fisher)
        paramsMM = np.copy(samples_fisher)

        if '_cw0_p_phase' in par_names[i]:
            if cm.use_default_cw0_p_sigma:
                fisher_diag[i] = 1/cm.sigma_cw0_p_phase_default**2
            #otherwise should already have been done

        elif par_names[i] in par_names_cw_ext:
            epsilon = cm.eps[par_names[i]]

            paramsPP[i] += 2*epsilon
            paramsMM[i] -= 2*epsilon

            FLI_swap.logdet_array[:] = 0.
            FLI_swap.resres_array[:] = 0.
            FLI_swap.logdet = 0.
            FLI_swap.resres =0.

            nns[i] = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            #use fast likelihood
            x0_swap.update_params(paramsPP)

            pps[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            x0_swap.update_params(paramsMM)

            mms[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)


            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher_diag[i] = -(pps[i] - 2*nns[i] + mms[i])/(4*epsilon*epsilon)

            #revert changes
            x0_swap.update_params(samples_fisher)

            FLI_swap.resres_array[:] = resres_array0
            FLI_swap.logdet_array[:] = logdet_array0
            FLI_swap.resres = np.sum(resres_array0)
            FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

        elif i in x0_swap.idx_dists:
            if cm.use_default_cw0_p_sigma:
                fisher_diag[i] = 1/cm.sigma_cw0_p_dist_default**2
            #should already have been done otherwise

        elif (i in x0_swap.idx_rn_gammas) or (i in x0_swap.idx_rn_log10_As):
            #continue
            if cm.use_default_noise_sigma :
                fisher_diag[i] = 1./cm.sigma_noise_default**2
            #already did all of the above otherwise
        else:
            epsilon = cm.eps[par_names[i]]

            paramsPP[i] += 2*epsilon
            paramsMM[i] -= 2*epsilon

            FLI_swap.logdet_array[:] = 0.
            FLI_swap.resres_array[:] = 0.
            FLI_swap.logdet = 0.
            FLI_swap.resres =0.

            nns[i] = FLI_swap.get_lnlikelihood(x0_swap)


            #must be one of the intrinsic parameters
            x0_swap.update_params(paramsPP)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0. #these are reset to nonzero by calling update_intrinsic, but they do not vary so don't include them in the likelihood
            FLI_swap.resres =0.
            pps[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            x0_swap.update_params(paramsMM)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0.
            FLI_swap.resres = 0. 
            mms[i] = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher_diag[i] = -(pps[i] - 2*nns[i] + mms[i])/(4*epsilon*epsilon)

            #revert changes
            x0_swap.update_params(samples_fisher)

            FLI_swap.cos_gwtheta = x0_swap.cos_gwtheta
            FLI_swap.gwphi = x0_swap.gwphi
            FLI_swap.log10_fgw = x0_swap.log10_fgw
            FLI_swap.log10_mc = x0_swap.log10_mc#

            FLI_swap.MMs[:] = MMs0
            FLI_swap.NN[:] = NN0
            FLI_swap.resres_array[:] = resres_array0
            FLI_swap.logdet_array[:] = logdet_array0
            FLI_swap.resres = np.sum(resres_array0)
            FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

    #update FLI_swap and x0_swap to at least a self consistent state
    FLI_swap.logdet_base = logdet_base_old
    x0_swap.update_params(samples_fisher)
    FLI_swap.MMs[:] = MMs0
    FLI_swap.NN[:] = NN0
    FLI_swap.resres_array[:] = resres_array0
    FLI_swap.logdet_array[:] = logdet_array0
    FLI_swap.resres = np.sum(resres_array0)
    FLI_swap.logdet = FLI_swap.logdet_base + np.sum(logdet_array0)

    #FLI_swap and x0_swap does not need to be reverted because it will be overwritten anyway
    #revert everything to original point
    #x0.update_params(samples_old)
    #FLI_swap.resres = resres_old
    #FLI_swap.logdet = logdet_old
    #FLI_swap.resres_array[:] = resres_array_old
    #FLI_loc.logdet_array[:] = logdet_array_old
    #FLI_swap.MMs = MMs_old
    #FLI_swap.NN = NN_old

    ##revert tracking safety parameters
    #FLI_swap.cos_gwtheta = x0.cos_gwtheta
    #FLI_swap.gwphi = x0.gwphi
    #FLI_swap.log10_fgw = x0.log10_fgw
    #FLI_swap.log10_mc = x0.log10_mc#

    #assert FLI_swap.cos_gwtheta == x0.cos_gwtheta
    #assert FLI_swap.gwphi == x0.gwphi
    #assert FLI_swap.log10_fgw == x0.log10_fgw
    #assert FLI_swap.log10_mc == x0.log10_mc
    #assert np.all(FLI_swap.rn_gammas==x0.rn_gammas)
    #assert np.all(FLI_swap.rn_log10_As==x0.rn_log10_As)


    #correct for the given temperature of the chain
    #fisher_diag = fisher_diag#/T_chain

    #filer out nans and negative values - set them to 1.0 which will result in
    fisher_diag[(~np.isfinite(fisher_diag))|(fisher_diag<0.)] = 1.
    #Fisher_diag = np.where(np.isfinite(fisher_diag), fisher_diag, 1.0)
    #FISHER_diag = np.where(fisher_diag>0.0, Fisher_diag, 1.0)

    #filter values smaller than 4 and set those to 4 -- Neil's trick -- effectively not allow jump Gaussian stds larger than 0.5=1/sqrt(4)
    eig_limit = 4.0
    #W = np.where(FISHER_diag>eig_limit, FISHER_diag, eig_limit)
    fisher_diag[fisher_diag<eig_limit] = eig_limit

    return 1/np.sqrt(fisher_diag)

################################################################################
#
#CALCULATE RN FISHER EIGENVECTORS
#
################################################################################
def get_fisher_eigenvectors(params, par_names, par_names_to_perturb, pta, epsilon=1e-4):
    try:
        dim = len(par_names_to_perturb)
        fisher = np.zeros((dim,dim))

        #lnlikelihood at specified point
        nn = pta.get_lnlikelihood(params)

        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[par_names.index(par_names_to_perturb[i])] += 2*epsilon
            paramsMM[par_names.index(par_names_to_perturb[i])] -= 2*epsilon
            
            #lnlikelihood at +-epsilon positions
            pp = pta.get_lnlikelihood(paramsPP)
            mm = pta.get_lnlikelihood(paramsMM)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

        #calculate off-diagonal elements
        for i in range(dim):
            for j in range(i+1,dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPP[par_names.index(par_names_to_perturb[j])] += epsilon
                paramsMM[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsPM[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[j])] += epsilon

                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)
                pm = pta.get_lnlikelihood(paramsPM)
                mp = pta.get_lnlikelihood(paramsMP)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

        #Filter nans and infs and replace them with 1s
        #this will imply that we will set the eigenvalue to 100 a few lines below
        FISHER = np.where(np.isfinite(fisher), fisher, 1.0)
        if not np.array_equal(FISHER, fisher):
            print("Changed some nan elements in the Fisher matrix to 1.0")

        #Find eigenvalues and eigenvectors of the Fisher matrix
        w, v = np.linalg.eig(FISHER)

        #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
        eig_limit = 4.0#1.0#0.25

        W = np.where(np.abs(w)>eig_limit, w, eig_limit)

        return (np.sqrt(1.0/np.abs(W))*v).T

    except:
        print("An Error occured in the eigenvalue calculation")
        print(par_names_to_perturb)
        print(params)
        return np.array(False)

@jitclass([('uniform_par_ids',nb.int64[:]),('uniform_lows',nb.float64[:]),('uniform_highs',nb.float64[:]),\
           ('lin_exp_par_ids',nb.int64[:]),('lin_exp_lows',nb.float64[:]),('lin_exp_highs',nb.float64[:]),\
           ('normal_par_ids',nb.int64[:]),('normal_mus',nb.float64[:]),('normal_sigs',nb.float64[:])])
class FastPriorInfo:
    """simple jitclass to store the various elements of fast prior calculation in a way that can be accessed quickly from a numba environment"""
    def __init__(self, uniform_par_ids, uniform_lows, uniform_highs, lin_exp_par_ids, lin_exp_lows, lin_exp_highs, normal_par_ids, normal_mus, normal_sigs):
        self.uniform_par_ids = uniform_par_ids
        self.uniform_lows = uniform_lows
        self.uniform_highs = uniform_highs
        self.lin_exp_par_ids = lin_exp_par_ids
        self.lin_exp_lows = lin_exp_lows
        self.lin_exp_highs = lin_exp_highs
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

@njit()
def reflect_into_range(x, x_low, x_high):
    if x<=x_high and x>=x_low:
        return x
    elif x<x_low:
        return 2*x_low - x
    elif x>x_high:
        return 2*x_high - x
