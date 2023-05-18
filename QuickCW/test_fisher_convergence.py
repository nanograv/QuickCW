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

#import re

import h5py

import QuickCW.CWFastLikelihoodNumba as CWFastLikelihoodNumba
import QuickCW.CWFastPrior as CWFastPrior
import QuickCW.const_mcmc as cm

import glob
import json

from QuickCW.QuickCW import do_intrinsic_update,do_extrinsic_block,FastPriorInfo,do_pt_swap,do_pt_swap_alt,summarize_a_ext,reflect_cosines,get_fisher_diagonal
import QuickCW.CWFastLikelihoodNumba as CWFastLikelihoodNumba

eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4}
################################################################################
#
#CALCULATE FISHER DIAGONAL
#
################################################################################
def get_fisher_diagonal_alt(T_chain, samples_current, par_names, par_names_cw_ext, x0, FLI_loc, epsilon=1e-2):
    dim = len(par_names)
    #dim = 10
    fisher_diag = np.zeros(dim)

    MMs_orig = FLI_loc.MMs.copy()
    NN_orig = FLI_loc.NN.copy()
    resres_orig = FLI_loc.resres
    logdet_orig = FLI_loc.logdet

    #we do not need the components that do not change for the derivatives, and adding them introduces numerical inaccuracy
    FLI_loc.resres = 0.
    FLI_loc.logdet = 0.

    nn = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)


    #future locations
    n_eps = 20
    mms = np.zeros((dim,n_eps))
    pps = np.zeros((dim,n_eps))
    fisher_diag = np.zeros((dim,n_eps))

    epses = np.zeros((dim,n_eps))
    eps_base = np.zeros(dim)
    eps_base[0:8] = np.array([1.e-4,1.e-4,1.e-4,1.e-5,1.e-5,1.e-4,1.e-4,1.e-4])
    cw_p_phase_eps = 1.e-3
    cw_p_dist_eps = 1.e-3 
    eps_base[8::2] = cw_p_dist_eps
    eps_base[9::2] = cw_p_phase_eps
    eps_mults = 10**np.linspace(-5,2,n_eps)#np.logspace(1.e-8,1.e0,n_eps)
    epses[:] = np.outer(eps_base,eps_mults)
    #epses[:] = 1.e-2

    #calculate diagonal elements
    for itre in range(n_eps):
        dist_count = 0
        phase_count = 0
        for i in range(8,dim):
            paramsPP = np.copy(samples_current)
            paramsMM = np.copy(samples_current)
            paramsPP[i] += 2*epses[i,itre]
            paramsMM[i] -= 2*epses[i,itre]

            if "_cw0_p_phase" in par_names[i]:
                FLI_loc.MMs[:phase_count,:,:] = 0.
                FLI_loc.MMs[phase_count,:2,:2] = 0.
                FLI_loc.MMs[phase_count+1:,:,:] = 0.

                FLI_loc.NN[:phase_count,:] = 0.
                FLI_loc.NN[phase_count,:2] = 0.
                FLI_loc.NN[phase_count+1:,:] = 0.

                nn_alt2 = FLI_loc.get_lnlikelihood(x0)

                x0.update_params(paramsPP)

                #FLI_loc.update_pulsar_distance(x0, dist_count)
                pps[i,itre] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                x0.update_params(paramsMM)

                #FLI_loc.update_pulsar_distance(x0, dist_count)
                mms[i,itre] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                #revert changes
                x0.update_params(samples_current)

                FLI_loc.MMs = MMs_orig.copy()
                FLI_loc.NN = NN_orig.copy()

                #use fast likelihood
                x0.update_params(paramsPP)

                #pps[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
                pps_alt = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                x0.update_params(paramsMM)

                #mms[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
                mms_alt = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                #revert changes
                x0.update_params(samples_current)
                fisher_diag_alt = -(pps_alt - 2.0*nn + mms_alt)/(4.0*epses[i,itre])

                FLI_loc.resres = resres_orig
                fisher_diag[i,itre] = -(pps[i,itre] - 2.0*nn_alt2 + mms[i,itre])/(4.0*epses[i,itre]**2)
#                fisher_diag[i,itre] = CWFastLikelihoodNumba.get_hess_fisher_phase_helper(x0,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs,phase_count)
                #print(fisher_diag[i,itre],fisher_diag_alt)
#                print('fish',fisher_diag_alt,fisher_diag[i,itre])
                #print(pps_alt,pps[i,itre],pps_alt2)
                #print(mms_alt,mms[i,itre],mms_alt2)
                #print(nn_alt,nn,nn_alt2)
                #assert np.isclose(fisher_diag_alt,fisher_diag_alt2,rtol=1.e-8)
#                assert np.isclose(fisher_diag_alt,fisher_diag[i,itre],rtol=1.e-3)




                phase_count += 1
            elif par_names[i] in par_names_cw_ext:
                #use fast likelihood
                x0.update_params(paramsPP)

                #pps[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
                pps[i,itre] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                x0.update_params(paramsMM)

                #mms[i] = CWFastLikelihoodNumba.get_lnlikelihood_helper(x0,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)
                mms[i,itre] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                #revert changes
                x0.update_params(samples_current)
                fisher_diag[i,itre] = -(pps[i,itre] - 2.0*nn + mms[i,itre])/(4.0*epses[i,itre]**2)

            elif "_cw0_p_dist" in par_names[i]:
                FLI_loc.MMs[:dist_count,:,:] = 0.
                FLI_loc.MMs[dist_count,:2,:2] = 0.
                FLI_loc.MMs[dist_count+1:,:,:] = 0.

                FLI_loc.NN[:dist_count,:] = 0.
                FLI_loc.NN[dist_count,:2] = 0.
                FLI_loc.NN[dist_count+1:,:] = 0.

                nn_alt2 = FLI_loc.get_lnlikelihood(x0)

                x0.update_params(paramsPP)

                FLI_loc.update_pulsar_distance(x0, dist_count)
                pps[i,itre] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                x0.update_params(paramsMM)

                FLI_loc.update_pulsar_distance(x0, dist_count)
                mms[i,itre] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                #revert changes
                x0.update_params(samples_current)

                FLI_loc.MMs = MMs_orig.copy()
                FLI_loc.NN = NN_orig.copy()
#                FLI_loc.resres = resres_orig
                fisher_diag[i,itre] = -(pps[i,itre] - 2.0*nn_alt2 + mms[i,itre])/(4.0*epses[i,itre]**2)
                #print('fish',fisher_diag_alt,fisher_diag[i,itre],fisher_diag_alt2)
                #print(pps_alt,pps[i,itre],pps_alt2)
                #print(mms_alt,mms[i,itre],mms_alt2)
                #print(nn_alt,nn,nn_alt2)
                #assert np.isclose(fisher_diag_alt,fisher_diag_alt2,rtol=1.e-8)
                #assert np.isclose(fisher_diag_alt,fisher_diag[i,itre],rtol=1.e-2)




                dist_count += 1

            else:
                #must be one of the intrinsic parameters
                x0.update_params(paramsPP)

                FLI_loc.update_intrinsic_params(x0)
                pps[i,itre] = FLI_loc.get_lnlikelihood(x0)#FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                x0.update_params(paramsMM)

                FLI_loc.update_intrinsic_params(x0)
                mms[i,itre] = FLI_loc.get_lnlikelihood(x0)#,FLI_loc.resres,FLI_loc.logdet,FLI_loc.pos,FLI_loc.pdist,FLI_loc.NN,FLI_loc.MMs)

                #revert changes
                x0.update_params(samples_current)

                FLI_loc.MMs[:] = MMs_orig
                FLI_loc.NN[:] = NN_orig

                FLI_loc.cos_gwtheta = x0.cos_gwtheta
                FLI_loc.gwphi = x0.gwphi
                FLI_loc.log10_fgw = x0.log10_fgw
                FLI_loc.log10_mc = x0.log10_mc#
                fisher_diag[i,itre] = -(pps[i,itre] - 2.0*nn + mms[i,itre])/(4.0*epses[i,itre]**2)

    #calculate diagonal elements of the Hessian from a central finite element scheme
    #note the minus sign compared to the regular Hessian

    #revert the constant components 
    FLI_loc.resres = resres_orig
    FLI_loc.logdet = logdet_orig

    ##correct for the given temperature of the chain
    #fisher_diag = fisher_diag/T_chain

    ##filer out nans and negative values - set them to 1.0 which will result in
    #fisher_diag[(~np.isfinite(fisher_diag))|fisher_diag<0.] = 1.
    ##Fisher_diag = np.where(np.isfinite(fisher_diag), fisher_diag, 1.0)
    ##FISHER_diag = np.where(fisher_diag>0.0, Fisher_diag, 1.0)

    ##filter values smaller than 4 and set those to 4 -- Neil's trick -- effectively not allow jump Gaussian stds larger than 0.5=1/sqrt(4)
    #eig_limit = 4.0
    #fisher_diag[fisher_diag<eig_limit] = eig_limit
    ##W = np.where(FISHER_diag>eig_limit, FISHER_diag, eig_limit)

    return fisher_diag,eps_mults


################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
#def QuickCW(N, T_max, n_chain, psrs, noise_json=None, n_status_update=100, n_int_block=100, n_extrinsic_step=1000, save_every_n=10_000, thin=10, savefile=None, n_update_fisher=100_000):
if __name__ == '__main__':

#with open('data/fast_like_test_psrs_A2e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad.pkl', 'rb') as psr_pkl:
    with open('data/fast_like_test_psrs_A2e-15_M5e9_f2e-8_evolve_no_gwb.pkl', 'rb') as psr_pkl:
#with open('data/fast_like_test_psrs.pkl', 'rb') as psr_pkl:
        psrs = pickle.load(psr_pkl)

#N = 30_000
#N = 500_000
    N = 2000
    T_max = 5
    n_chain = 8
    save_every_n = 100
    n_status_update=100
    n_extrinsic_step=10
    n_update_fisher=100_000
    n_int_block = 100
    thin = 10


    noise_json = 'data/channelized_12p5yr_v3_full_noisedict_gp_ecorr.json'

#savefile = 'results/quickCW_test_A2e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.npz'
#savefile = 'results/quickCW_test_A1e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.npz'
    savefile = None#'results/quickCW_test_A1e-15_M5e9_f2e-8_evolve_no_gwb_no_rn_no_ecorr_no_equad_v1.h5'
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
            samples[j,0,par_names.index(psr + "_red_noise_gamma")] = noisedict[psr + "_red_noise_gamma"]
            samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = noisedict[psr + "_red_noise_log10_A"]

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

    print('setting up FastLikeMaster')
    x0 = CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[j,0],par_names,par_names_cw_ext)
    flm = CWFastLikelihoodNumba.FastLikeMaster(psrs,pta,dict(zip(par_names, samples[j, 0, :])),x0)
    params_orig = dict(zip(par_names, samples[j, 0, :]))
    FLI_temp = flm.get_new_FastLike(x0,params_orig)


    n_jump_loc = 2*len(psrs)
    #idx_choose_psr = np.random.randint(0,x0s[j].Npsr,cm.n_noise_main)
    idx_choose_psr = list(range(len(psrs)))
    idx_choose = np.concatenate((x0.idx_rn_gammas,x0.idx_rn_log10_As))
    scaling = 2.38/np.sqrt(n_jump_loc)
    jump = np.zeros(len(par_names))
    jump[idx_choose] = np.random.normal(0.,1.,n_jump_loc)
    new_point = samples[0,0]+jump

    #params_hold1 = dict(zip(par_names, samples[0, 0, :]))
    #params_hold2 = dict(zip(par_names, samples[0, 0, :]))
    #params_hold3 = dict(zip(par_names, new_point))
    #FLI_hold1 = CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, params_hold1, x0)
    #FLI_hold2 = CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, params_hold2, x0)
    #FLI_hold3 = CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, params_hold3, x0)

    #x0.update_params(new_point)
    #CWFastLikelihoodNumba.update_FLI_rn(FLI_hold2,x0,idx_choose_psr,psrs,pta,par_names,new_point)
    #FLI_hold4 = flm.recompute_FastLike(x0,new_point)
    #flm2 = CWFastLikelihoodNumba.FastLikeMaster(psrs,pta,dict(zip(par_names, new_point)),x0)
    #FLI_hold5 = flm2.recompute_FastLike(x0,new_point)

    Npsr = len(psrs)
    #for i in range(0,Npsr):
    #    assert np.all(FLI_hold1.Nvecs[i]==FLI_hold2.Nvecs[i])
    #    assert np.all(FLI_hold1.Nvecs[i]==FLI_hold3.Nvecs[i])
    #    assert np.all(FLI_hold1.Nvecs[i]==FLI_hold4.Nvecs[i])
    #    assert np.all(FLI_hold1.Nvecs[i]==FLI_hold5.Nvecs[i])

    #    assert np.all(flm.TNTs[i]==flm2.TNTs[i])
    #    #assert np.all(FLI_hold1.TNTs[i]==FLI_hold3.TNTs[i])

    #    assert np.all(flm.TNvs[i]==flm2.TNvs[i])
    #    #assert np.all(FLI_hold1.TNvs[i]==FLI_hold3.TNvs[i])

    #    assert np.all(FLI_hold1.Nrs[i]==FLI_hold2.Nrs[i])
    #    assert np.all(FLI_hold1.Nrs[i]==FLI_hold3.Nrs[i])
    #    assert np.all(FLI_hold1.Nrs[i]==FLI_hold4.Nrs[i])
    #    assert np.all(FLI_hold1.Nrs[i]==FLI_hold5.Nrs[i])

    #    assert np.all(FLI_hold1.toas[i]==FLI_hold2.toas[i])
    #    assert np.all(FLI_hold1.toas[i]==FLI_hold3.toas[i])
    #    assert np.all(FLI_hold1.toas[i]==FLI_hold4.toas[i])
    #    assert np.all(FLI_hold1.toas[i]==FLI_hold5.toas[i])

    #    assert np.all(FLI_hold1.residuals[i]==FLI_hold2.residuals[i])
    #    assert np.all(FLI_hold1.residuals[i]==FLI_hold3.residuals[i])
    #    assert np.all(FLI_hold1.residuals[i]==FLI_hold4.residuals[i])
    #    assert np.all(FLI_hold1.residuals[i]==FLI_hold5.residuals[i])

    #    assert np.all(FLI_hold1.isqNvecs[i]==FLI_hold2.isqNvecs[i])
    #    assert np.all(FLI_hold1.isqNvecs[i]==FLI_hold3.isqNvecs[i])
    #    assert np.all(FLI_hold1.isqNvecs[i]==FLI_hold4.isqNvecs[i])
    #    assert np.all(FLI_hold1.isqNvecs[i]==FLI_hold5.isqNvecs[i])
    #    
    #    assert np.all(FLI_hold1.pos[i]==FLI_hold2.pos[i])
    #    assert np.all(FLI_hold1.pos[i]==FLI_hold3.pos[i])
    #    assert np.all(FLI_hold1.pos[i]==FLI_hold4.pos[i])
    #    assert np.all(FLI_hold1.pos[i]==FLI_hold5.pos[i])

    #    assert np.all(FLI_hold1.pdist[i]==FLI_hold2.pdist[i])
    #    assert np.all(FLI_hold1.pdist[i]==FLI_hold3.pdist[i])
    #    assert np.all(FLI_hold1.pdist[i]==FLI_hold4.pdist[i])
    #    assert np.all(FLI_hold1.pdist[i]==FLI_hold5.pdist[i])

    #    #asse/var/folders/x2/ldvv5mnn3516rwwy1_mk51bw0000gn/T/com.apple.mail/com.apple.mail.drag-T0x600003180bc0.tmp.NBoQf6/A\ few\ days\ left\!\ Confirm\ your\ identity\ to\ keep\ using\ Venmo\ the\ way\ you\ always\ have.eml rt FLI_hold1.logdet == FLI_hold3.logdet

    #assert FLI_hold1.max_toa == FLI_hold2.max_toa
    #assert FLI_hold1.max_toa == FLI_hold3.max_toa
    #assert FLI_hold1.max_toa == FLI_hold4.max_toa
    #assert FLI_hold1.max_toa == FLI_hold5.max_toa

    #assert flm.logdet == flm2.logdet
    #import sys
    #sys.exit()


    #set up fast likelihoods
    #x0s = List([])
    #FLIs  = List([])
    #for j in range(n_chain):
    #    x0s.append( CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[j,0],par_names,par_names_cw_ext))
    #    FLIs.append(CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0s[j]))



    FLI_temp.update_intrinsic_params(x0)

    MMs2 = FLI_temp.MMs.copy()
    NN2 = FLI_temp.NN.copy()

    FLI_temp.update_intrinsic_params(x0)
    MMs1 = FLI_temp.MMs.copy()
    NN1 = FLI_temp.NN.copy()

    
    #assert np.allclose(MMs1,MMs2)
    #assert np.allclose(NN1,NN2)

    flm.recompute_FastLike(FLI_temp,x0,dict(zip(par_names, samples[j, 0, :])))


    #CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0)
    print("started timings")
    n_run_FLI = 10
    t0 = perf_counter()
    for m in range(n_run_FLI):
        flm.recompute_FastLike(FLI_temp,x0,dict(zip(par_names, samples[j, 0, :])))
        #CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0s[j])
    tf = perf_counter()
    print('FLI alt create_time %8.5fs'%((tf-t0)/n_run_FLI))

    #n_run_FLI = 10
    #t0 = perf_counter()
    #for m in range(n_run_FLI):
    #    CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0)
    #tf = perf_counter()
    #print('FLI create_time %8.5fs'%((tf-t0)/n_run_FLI))

    FLI_temp.update_intrinsic_params(x0)

    n_run_int = 1000
    t0 = perf_counter()
    for m in range(n_run_int):
        FLI_temp.update_intrinsic_params(x0)
        #CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0s[j])
    tf = perf_counter()
    print('FLI updatetime2 %8.5fs'%((tf-t0)/n_run_int))

    import sys
    sys.exit()

    n_run_int = 1000
    t0 = perf_counter()
    for m in range(n_run_int):
        FLI_temp.update_intrinsic_params(x0)
        #CWFastLikelihoodNumba.get_FastLikeInfo(psrs, pta, dict(zip(par_names, samples[j, 0, :])), x0s[j])
    tf = perf_counter()
    print('FLI updatetime %8.5fs'%((tf-t0)/n_run_int))



    import sys
    sys.exit()


    #calculate the diagonal elements of the fisher matrix
    #    print(fisher_diag[j,:])


    ##setting up arrays to record acceptance and swaps
    #a_yes=np.zeros((11,n_chain),dtype=np.int64) #columns: chain number; rows: proposal type (PT, cos_gwtheta, cos_inc, gwphi, fgw, h, mc, phase0, psi, p_phases, p_dists)
    #a_no=np.zeros((11,n_chain),dtype=np.int64)
    #acc_fraction = a_yes/(a_no+a_yes)

    ##printing info about initial parameters
    #for j in range(n_chain):
    #    print("j="+str(j))
    #    print(samples[j,0,:])
    #    #log_likelihood[j,0] = pta.get_lnlikelihood(samples[j,0,:])
    #    log_likelihood[j,0] = FLIs[j].get_lnlikelihood(x0s[j])
    #    print("log_likelihood="+str(log_likelihood[j,0]))
    #    #print("log_prior="+str(pta.get_lnprior(samples[j,0,:])))
    #    print("log_prior="+str(FastPrior.get_lnprior(samples[j,0,:])))

    #stop_iter = N

    #ext_update = List([False,]*n_chain)
    #sample_updates = List([np.copy(samples[j,0,:]) for j in range(n_chain)])

    tf_init = perf_counter()
    print('finished initialization steps in '+str(tf_init-ti)+'s')
    ti_loop = perf_counter()

    print(FLIs[j].get_lnlikelihood(x0s[j]))
    #t0 = perf_counter()
    #FLIs[j].update_pulsar_distances_alt(x0s[j], np.arange(0,x0s[j].Npsr)) #TODO validate assumption on psr index ranges
    #tf = perf_counter()
    #MM0s = FLIs[j].MMs.copy()
    #NN0s = FLIs[j].NN.copy()

    #print('run0 advance time '+str(tf-t0))
    print(FLIs[j].get_lnlikelihood(x0s[j])) 
    t0 = perf_counter()
    FLIs[j].update_pulsar_distances(x0s[j], np.arange(0,x0s[j].Npsr)) #TODO validate assumption on psr index ranges
    tf = perf_counter()
    MM1s = FLIs[j].MMs.copy()
    NN1s = FLIs[j].NN.copy()
    print('run1 advance time '+str(tf-t0))
    print(FLIs[j].get_lnlikelihood(x0s[j]))
    t0 = perf_counter()
    for kk in range(x0s[j].Npsr):
        FLIs[j].update_pulsar_distance(x0s[j], kk) #TODO validate assumption on psr index ranges
    tf = perf_counter()
    MM2s = FLIs[j].MMs.copy()
    NN2s = FLIs[j].NN.copy()
    print('run2 advance time '+str(tf-t0))
    print(FLIs[j].get_lnlikelihood(x0s[j]))
    t0 = perf_counter()
    FLIs[j].update_intrinsic_params(x0s[j]) #TODO validate assumption on psr index ranges
    tf = perf_counter()
    MM3s = FLIs[j].MMs.copy()
    NN3s = FLIs[j].NN.copy()
    print('run3 advance time '+str(tf-t0))
    print(FLIs[j].get_lnlikelihood(x0s[j]))

    fisher_diag = np.ones((n_chain, len(par_names)))
    for j in range(n_chain):
        fisher_diag[j,:] = get_fisher_diagonal(Ts[j], samples[j,0,:], par_names, par_names_cw_ext, x0s[j], FLIs[j])

    #ti_dist4 = perf_counter()
    #n_run4 = 100
    #for mm in range(n_run4):
    #    FLIs[j].update_pulsar_distances(x0s[j], np.arange(0,x0s[j].Npsr)) #TODO validate assumption on psr index ranges
    #tf_dist4 = perf_counter()
    #if n_run4>0:
    #    print('time for updating all distances together alt%.5e s'%((tf_dist4-ti_dist4)/n_run4))

    ti_dist2 = perf_counter()
    n_run2 = 100
    for mm in range(n_run2):
        FLIs[j].update_intrinsic_params(x0s[j]) #TODO validate assumption on psr index ranges
    tf_dist2 = perf_counter()
    if n_run2>0:
        print('time for updating all distances instrinsic %.5e s'%((tf_dist2-ti_dist2)/n_run2))

    ti_dist0 = perf_counter()
    n_run0 = 1000
    for mm in range(n_run0):
        FLIs[j].update_pulsar_distances(x0s[j], np.arange(0,x0s[j].Npsr)) #TODO validate assumption on psr index ranges
    tf_dist0 = perf_counter()
    if n_run0>0:
        print('time for updating all distances together   %.5e s'%((tf_dist0-ti_dist0)/n_run0))


    ti_dist1 = perf_counter()
    n_run1 = 100
    for mm in range(n_run1):
        for k in range(0,x0s[j].Npsr):
            FLIs[j].update_pulsar_distance(x0s[j], k) #TODO validate assumption on psr index ranges
    tf_dist1 = perf_counter()
    if n_run0>0:
        print('time for updating all distances separately %.5e s'%((tf_dist1-ti_dist1)/n_run1))

    n_fs = 201 
    MMs_mat = np.zeros((n_fs,45,4,4))
    NNs_mat = np.zeros((n_fs,45,4))
    logLs_mat = np.zeros(n_fs)
    #log10fs_try = x0s[j].log10_fgw+np.linspace(-0.5,0.5,n_fs)
    idx_try = 27
    dists_try = x0s[j].cw_p_dists[idx_try]+np.linspace(-1.,1.,n_fs)
    for itrf in range(0,n_fs):
        #x0s[j].log10_fgw = log10fs_try[itrf]
        x0s[j].cw_p_dists[idx_try] = dists_try[itrf]
        #FLIs[j].update_intrinsic_params(x0s[j])
        FLIs[j].logdet = 0.
        FLIs[j].resres = 0.
        FLIs[j].update_pulsar_distance(x0s[j],idx_try)
        MMs_mat[itrf] = FLIs[j].MMs.copy()
        NNs_mat[itrf] = FLIs[j].NN.copy()
        logLs_mat[itrf] = FLIs[j].get_lnlikelihood(x0s[j])

    import matplotlib.pyplot as plt
    plt.semilogy(dists_try,np.abs(MMs_mat[:,idx_try,0,0]))
    plt.semilogy(dists_try,np.abs(MMs_mat[:,idx_try,1,1]))
    plt.semilogy(dists_try,np.abs(MMs_mat[:,idx_try,2,2]))
    plt.semilogy(dists_try,np.abs(MMs_mat[:,idx_try,3,3]))
    plt.show()

    plt.semilogy(dists_try,np.abs(NNs_mat[:,idx_try,0]))
    plt.semilogy(dists_try,np.abs(NNs_mat[:,idx_try,1]))
    plt.semilogy(dists_try,np.abs(NNs_mat[:,idx_try,2]))
    plt.semilogy(dists_try,np.abs(NNs_mat[:,idx_try,3]))
    plt.show()

    plt.plot(dists_try,logLs_mat)
    plt.show()
    import sys
    sys.exit()

    ti_fisher = perf_counter()  
    fisher_diags,eps_mults = get_fisher_diagonal_alt(Ts[0], samples[0,0,:], par_names, par_names_cw_ext, x0s[0], FLIs[0])
    scale_idx = np.argmin(np.abs(eps_mults-1.))
    tf_fisher = perf_counter()
    print('fisher time in '+str(tf_fisher-ti_fisher))
    import matplotlib.pyplot as plt
    plt.loglog(eps_mults,np.abs(fisher_diags.T/fisher_diags[:,scale_idx]).T[0:8].T)
    plt.legend(par_names[0:8])
    plt.xlabel('eps/eps_0')
    plt.ylabel('fisher/fisher_0')
    plt.show()

    plt.loglog(eps_mults,np.abs(fisher_diags.T/fisher_diags[:,scale_idx]).T[8::2].T)
    plt.xlabel('eps/eps_0')
    plt.ylabel('fisher/fisher_0')
    plt.title('cw_p_dists')
    plt.show()

    plt.loglog(eps_mults,np.abs(fisher_diags.T/fisher_diags[:,scale_idx]).T[9::2].T)
    plt.show()

    ##############################################################################
    #
    # Main MCMC iteration
    #
    ##############################################################################
#    for i in range(int(N/n_int_block)):
#        if (i*n_int_block)%save_every_n==0 and i!=0:
#            acc_fraction = a_yes/(a_no+a_yes)
#            #np.savez(savefile, samples=samples[0,:i*n_int_block,:], par_names=par_names, acc_fraction=acc_fraction, log_likelihood=log_likelihood[:,:i*n_int_block])
#            if savefile is not None:
#                if i*n_int_block>save_every_n:
#                    print("Append to HDF5 file...")
#                    with h5py.File(savefile, 'a') as f:
#                        f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
#                        f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
#                        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
#                        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
#                        f['acc_fraction'][...] = np.copy(acc_fraction)
#                        f['fisher_diag'][...] = np.copy(fisher_diag)
#                else:
#                    print("Create HDF5 file...")
#                    with h5py.File(savefile, 'w') as f:
#                        f.create_dataset('samples_cold', data=samples[0,:-1:thin,:], compression="gzip", chunks=True, maxshape=(int(N/thin),samples.shape[2]))
#                        f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape=(samples.shape[0],int(N/thin)))
#                        f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
#                        f.create_dataset('acc_fraction', data=acc_fraction)
#                        f.create_dataset('fisher_diag', data=fisher_diag)
#            #clear out log_likelihood and samples arrays
#            samples_now = samples[:,-1,:]
#            log_likelihood_now = log_likelihood[:,-1]
#            samples = np.zeros((n_chain, save_every_n+1, len(par_names)))
#            log_likelihood = np.zeros((n_chain,save_every_n+1))
#            samples[:,0,:] = np.copy(samples_now)
#            log_likelihood[:,0] = np.copy(log_likelihood_now)
#        if (i*n_int_block)%n_update_fisher==0 and i!=0:
#            print("Updating Fisher diagonals")
#            for j in range(n_chain):
#                fisher_diag[j,:] = get_fisher_diagonal(Ts[j], samples[j,(i*n_int_block)%save_every_n,:], par_names, par_names_cw_ext, x0s[j], FLIs[j])
#        if i%n_status_update==0:
#            acc_fraction = a_yes/(a_no+a_yes)
#            print('Progress: {0:2.2f}% '.format(i*n_int_block/N*100) +'Acceptance fraction #columns: chain number; rows: proposal type (PT, cos_gwtheta, cos_inc, gwphi, fgw, h, mc, phase0, psi, p_phases, p_dists):')
#            t_itr = perf_counter()
#            print('at t= '+str(t_itr-ti_loop)+'s')
#            print(acc_fraction)
#            print("New log_L=", str(FLIs[0].get_lnlikelihood(x0s[0])))#,FLIs[0].resres,FLIs[0].logdet,FLIs[0].pos,FLIs[0].pdist,FLIs[0].NN,FLIs[0].MMs)))
#            #print("Old log_L=", str(pta.get_lnlikelihood(samples[0,(i*n_int_block)%save_every_n,:])))
#        #update extrinsic parameters sometimes
#        if i%n_extrinsic_step==0 and i!=0:
#            do_intrinsic_update(n_chain, pta, samples, i*n_int_block, Ts, a_yes, a_no, x0s, FLIs, FastPrior, par_names, par_names_cw_int, log_likelihood, save_every_n, fisher_diag)
#            a_yes_counts_loc,a_no_counts_loc = do_extrinsic_block(n_chain, samples, i*n_int_block+1, Ts, x0s, FLIs, FPI, len(par_names), len(par_names_cw_ext), log_likelihood, n_int_block-1, save_every_n, fisher_diag)
#            a_yes += summarize_a_ext(a_yes_counts_loc,par_inds_cw_p_phase_ext)
#            a_no += summarize_a_ext(a_no_counts_loc,par_inds_cw_p_phase_ext)
#            do_pt_swap(n_chain, samples, (i+1)*n_int_block-1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n)
#        else:
#            a_yes_counts_loc,a_no_counts_loc = do_extrinsic_block(n_chain, samples, i*n_int_block, Ts, x0s, FLIs, FPI, len(par_names), len(par_names_cw_ext), log_likelihood, n_int_block, save_every_n, fisher_diag)
#            a_yes += summarize_a_ext(a_yes_counts_loc,par_inds_cw_p_phase_ext)
#            a_no += summarize_a_ext(a_no_counts_loc,par_inds_cw_p_phase_ext)
#            do_pt_swap(n_chain, samples, (i+1)*n_int_block-1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood, save_every_n)
#
#    acc_fraction = a_yes/(a_no+a_yes)
#    print("Append to HDF5 file...")
#    if savefile is not None:
#        with h5py.File(savefile, 'a') as f:
#            f['samples_cold'].resize((f['samples_cold'].shape[0] + int((samples.shape[1] - 1)/thin)), axis=0)
#            f['samples_cold'][-int((samples.shape[1]-1)/thin):] = np.copy(samples[0,:-1:thin,:])
#            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
#            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = np.copy(log_likelihood[:,:-1:thin])
#            f['acc_fraction'][...] = np.copy(acc_fraction)
#            f['fisher_diag'][...] = np.copy(fisher_diag)
#        #return samples, par_names, acc_fraction, pta, log_likelihood
    tf = perf_counter()
    print('whole function time ='+str(tf-ti)+'s')
    print('loop time ='+str(tf-ti_loop)+'s')

