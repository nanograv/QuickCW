"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""
#import pickle

from time import perf_counter
#import glob
import json

import numpy as np
#np.seterr(all='raise')
#make sure to use the right threading layer
from numba import config
config.THREADING_LAYER = 'omp'
#config.THREADING_LAYER = 'tbb'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

import enterprise
#from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
#from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
#from enterprise.signals import deterministic_signals

from enterprise_extensions import deterministic

import QuickCW.const_mcmc as cm
from QuickCW.QuickMCMCUtils import MCMCChain, ChainParams

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
#@profile
def QuickCW(chain_params: ChainParams, psrs:list, noise_json:str=None,
            use_legacy_equad: bool=False, include_ecorr:bool=True,
            amplitude_prior:str='UL',
            gwb_gamma_prior:str=None,
            return_only_pta:bool=False):
    """Set up all essential objects for QuickCW to do MCMC iterations"""
    print("Began Main Loop")

    ti = perf_counter()

    #Get observing timespan
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # define white noise signals
    if use_legacy_equad:
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
    else:
        efq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
    #ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    #ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection)
    if include_ecorr:
        #give ecorr a name so that we can use the usual noisefiles created for Kernel ecorr
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')

    log10_A = parameter.Uniform(-20, -11)
    #log10_A = parameter.Uniform(-18, -11)
    gamma = parameter.Uniform(0, 7)

    # define powerlaw PSD and red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    cos_gwtheta = parameter.Uniform(-1,1)('0_cos_gwtheta')
    gwphi = parameter.Uniform(0,2*np.pi)('0_gwphi')

    #set lower frequency bound to 1/Tspan if it's nan
    if np.isnan(chain_params.freq_bounds[0]):
        print('Found lower frequency bound of nan - Setting it to 1/T.')
        chain_params.freq_bounds[0] = 1/Tspan
    log10_fgw = parameter.Uniform(np.log10(chain_params.freq_bounds[0]), np.log10(chain_params.freq_bounds[1]))('0_log10_fgw')

    m_max = 10

    log10_mc = parameter.Uniform(7,m_max)('0_log10_mc')

    phase0 = parameter.Uniform(0, 2*np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

    p_phase = parameter.Uniform(0, 2*np.pi)
    p_dist = parameter.Normal(0, 1)

    if amplitude_prior=='detection':
        log10_h = parameter.Uniform(-18, -11)('0_log10_h')
    elif amplitude_prior=='UL':
        log10_h = parameter.LinearExp(-18, -11)('0_log10_h')
    else:
        raise NotImplementedError("amplitude_prior provided not implemented\nuse either 'detection' for uniform in log-amplitude or 'UL' for uniform in amplitude prior")

    cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                                   log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0, psrTerm=True,
                                   p_phase=p_phase, p_dist=p_dist, evolve=True,
                                   psi=psi, cos_inc=cos_inc, tref=cm.tref)
    cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')

    log10_Agw = parameter.Uniform(-20,-11)('gwb_log10_A')

    if gwb_gamma_prior is None:
        gwb_gamma_prior = np.array([0,7])

    gamma_gw = parameter.Uniform(gwb_gamma_prior[0],gwb_gamma_prior[1])('gwb_gamma')
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=chain_params.gwb_comps, Tspan=Tspan, name='gw')

    tm = gp_signals.TimingModel()

    if include_ecorr:
        if use_legacy_equad:
            s = ef + eq + ec + rn + crn + cw + tm
        else:
            s = efq     + ec + rn + crn + cw + tm
    else:
        if use_legacy_equad:
            s = ef + eq      + rn + crn + cw + tm
        else:
            s = efq          + rn + crn + cw + tm

    models = [s(psr) for psr in psrs]

    t1 = perf_counter()
    print("Begin Loading Pulsar Timing Array from Enterprise at %8.3fs"%(t1-ti))
    pta = signal_base.PTA(models)
    t1 = perf_counter()
    print("Finished Loading Pulsar Timing Array from Enterprise at %8.3fs"%(t1-ti))

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)

    #print(noisedict)
    pta.set_default_params(noisedict)
    if chain_params.verbosity>1:
        print(pta.summary())
    if chain_params.verbosity>0:
        print("List of parameters in the model with their priors:")
        print(pta.params)

    #get max toa which is needed to mae sure the initial parameters don't result in an already merged system
    max_toa = np.max(psrs[0].toas)
    for i in range(len(psrs)):
        max_toa = max(max_toa,np.max(psrs[i].toas))

    ##############################################################################
    #
    # Make MCMC Object
    #
    ##############################################################################
    mcc = MCMCChain(chain_params,psrs,pta,max_toa,noisedict,ti)
    if return_only_pta:
        return pta
    return pta,mcc
