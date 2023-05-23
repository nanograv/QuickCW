"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""
# import pickle

from time import perf_counter
# import glob
import json
import pickle

import numpy as np
# np.seterr(all='raise')
# make sure to use the right threading layer
from numba import config

config.THREADING_LAYER = 'omp'
# config.THREADING_LAYER = 'tbb'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
# from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
# from enterprise.signals import deterministic_signals

from enterprise_extensions import deterministic

import QuickCW.const_mcmc as cm
from QuickCW.QuickMCMCUtils import MCMCChain, ChainParams

import inspect
from QuickCW.PulsarDistPriors import DMDistParameter, PXDistParameter

################################################################################
#
# MAIN MCMC ENGINE
#
################################################################################
#@profile
def QuickCW(chain_params, psrs, noise_json=None, use_legacy_equad=False, include_ecorr=True, amplitude_prior='UL', gwb_gamma_prior=None, psr_distance_file=None, backend_selection=True):
    """Set up all essential objects for QuickCW to do MCMC iterations

    :param chain_params:        ChainParams object
    :param psrs:                enterprise pulsar objects
    :param noise_json:          JSON file with noise dictionary [None]
    :param use_legacy_equad:    Option to use old convention for equad [False]
    :param include_ecorr:       Option to include ECORR white noise [True]
    :param amplitude_prior:     Prior to use on CW amplitude; 'UL' indicates uniform in amplitude prior (used for upper limits); 'detection' indicates uniform in log amplitude prior (used for Bayes factor calculation/detection)
    :param gwb_gamma_prior:     Option to specify prior range on GWB spectral index gamma; None means we use the default np.array([0,7]) [None]
    :param psr_distance_file:   File containing parallax and DM distance information for pulsars; If None, we use Gaussian prior with pulsar distance and error from psr objects [None]
    :param backend_selection:   Option to use an enterprise Selection based on backend; Usually use True for real data False for simulated data [True]

    :return pta:                enterprise PTA object
    :return mcc:                MCMCChain onject
    """
    print("Began Main Loop")

    ti = perf_counter()

    # Get observing timespan
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    # define selection by observing backend
    if backend_selection:
        selection = selections.Selection(selections.by_backend)
    else:
        selection = selections.Selection(selections.no_selection)

    # define white noise signals
    if use_legacy_equad:
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
    else:
        efq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
    # ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    # ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection)
    if include_ecorr:
        # give ecorr a name so that we can use the usual noisefiles created for Kernel ecorr
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')

    log10_A = parameter.Uniform(-20, -11)
    # log10_A = parameter.Uniform(-18, -11)
    gamma = parameter.Uniform(0, 7)

    # define powerlaw PSD and red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    log10_Agw = parameter.Uniform(-20,-11)('gwb_log10_A')

    if gwb_gamma_prior is None:
        gwb_gamma_prior = np.array([0,7])

    gamma_gw = parameter.Uniform(gwb_gamma_prior[0],gwb_gamma_prior[1])('gwb_gamma')
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=chain_params.gwb_comps, Tspan=Tspan, name='gw')

    tm = gp_signals.TimingModel()

    if include_ecorr:
        if use_legacy_equad:
            s_base = ef + eq + ec + rn + crn + tm
        else:
            s_base = efq     + ec + rn + crn + tm
    else:
        if use_legacy_equad:
            s_base = ef + eq      + rn + crn + tm
        else:
            s_base = efq          + rn + crn + tm


    #cos_gwtheta = parameter.Uniform(-1,1)('0_cos_gwtheta')
    #gwphi = parameter.Uniform(0,2*np.pi)('0_gwphi')
    cos_gwtheta = parameter.Uniform(chain_params.cos_gwtheta_bounds[0],chain_params.cos_gwtheta_bounds[1])('0_cos_gwtheta')
    gwphi = parameter.Uniform(chain_params.gwphi_bounds[0],chain_params.gwphi_bounds[1])('0_gwphi')

    # set lower frequency bound to 1/Tspan if it's nan
    if np.isnan(chain_params.freq_bounds[0]):
        print('Found lower frequency bound of nan - Setting it to 1/T.')
        chain_params.freq_bounds[0] = 1 / Tspan
    log10_fgw = parameter.Uniform(np.log10(chain_params.freq_bounds[0]), np.log10(chain_params.freq_bounds[1]))(
        '0_log10_fgw')

    m_max = 10

    log10_mc = parameter.Uniform(7, m_max)('0_log10_mc')

    phase0 = parameter.Uniform(0, 2 * np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

    p_phase = parameter.Uniform(0, 2*np.pi)

    if amplitude_prior == 'detection':
        log10_h = parameter.Uniform(-18, -11)('0_log10_h')
    elif amplitude_prior == 'UL':
        log10_h = parameter.LinearExp(-18, -11)('0_log10_h')
    else:
        raise NotImplementedError(
            "amplitude_prior provided not implemented\nuse either 'detection' for uniform in log-amplitude or 'UL' for uniform in amplitude prior")

    if psr_distance_file is None: #No pulsar distance file --> use Gaussian prior with pulsar distance and error from psr objects
        if np.any(np.array([psr.pdist[0] for psr in psrs])==0): #raise error if this is used with psr objects having zero distance
            raise ValueError("It looks like some of the pulsar distances used are zero. Please provide psr_distance_file or use psr objects with nonzero distance.")

        p_dist = parameter.Normal(0, 1)

        cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                                       log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0, psrTerm=True,
                                       p_phase=p_phase, p_dist=p_dist, evolve=True,
                                       psi=psi, cos_inc=cos_inc, tref=cm.tref)
        cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')

        s = s_base + cw

        models = [s(psr) for psr in psrs]
    else: #provided pulsar distance file --> use information in that file for setting up pulsar distance priors
        if (np.any(np.array([psr.pdist[0] for psr in psrs])>0)) | np.any(np.array([psr.pdist[1] for psr in psrs])!=1): #raise error if this is used while any of the pulsars have non-zero distance
            raise ValueError("You are running in a mode using parallax and DM based pulsar distance priors, but some of the pulsar object have non-zero distances or non-unit variances. This method requires the pulsar objects to have zero mean and unit variance. Use psr objects that satisfy that or switch to using Gaussian priors based on distances in psr objects by setting psr_distance_file=None.")

        #load provided pulsar distance file
        with open(psr_distance_file, 'rb') as fp:
            pulsar_distances = pickle.load(fp)

        models = []
        for psr in psrs:
            # arguments for deterministic.cw_delay
            cw_delay_args = dict(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                                           log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0, psrTerm=True,
                                           p_phase=p_phase, evolve=True,
                                           psi=psi, cos_inc=cos_inc, tref=cm.tref)
            # arguments for deterministic.CWSignal
            CWSignal_args = dict(psrTerm=True, name='cw0')
            # create CW signal applying pulsar distance prior
            cw = per_pulsar_prior(psr, pulsar_distances, cw_delay_args, CWSignal_args)
            s = s_base + cw

            models.append(s(psr))

    t1 = perf_counter()
    print("Begin Loading Pulsar Timing Array from Enterprise at %8.3fs" % (t1 - ti))
    pta = signal_base.PTA(models)
    t1 = perf_counter()
    print("Finished Loading Pulsar Timing Array from Enterprise at %8.3fs" % (t1 - ti))

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)

    # print(noisedict)
    pta.set_default_params(noisedict)
    if chain_params.verbosity > 1:
        print(pta.summary())
    if chain_params.verbosity > 0:
        print("List of parameters in the model with their priors:")
        print(pta.params)

    # get max toa which is needed to mae sure the initial parameters don't result in an already merged system
    max_toa = np.max(psrs[0].toas)
    for i in range(len(psrs)):
        max_toa = max(max_toa, np.max(psrs[i].toas))

    ##############################################################################
    #
    # Make MCMC Object
    #
    ##############################################################################
    mcc = MCMCChain(chain_params, psrs, pta, max_toa, noisedict, ti)
    return pta, mcc


def get_default_args(func):
    """Gets default arguments from a python function"""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def per_pulsar_prior(enterprise_pulsar: Pulsar, pulsar_distances: dict,
                     cw_delay_args: dict=None, CWSignal_args: dict=None):
    """Creates a CW signal applying distance priors to individual pulsars based on DM or PX in the pulsar_distances dict
    
    :param enterprise_pulsar:   enterprise pulsar object
    :param pulsar_distances:    dictionary containing pulsar distance info
    :param cw_delay_args:       arguments to be passed on to deterministic.cw_delay
    :param CWSignal_args:       arguments to be passed on to deterministic.CWSignal

    :return cw:                 enterprise signal object with the CW model
    """
    if cw_delay_args is None:
        # could maybe replace this by using an empty dictionary instead
        cw_delay_args = get_default_args(deterministic.cw_delay)
    if CWSignal_args is None:
        CWSignal_args = get_default_args(deterministic.CWSignal)

    if 'DM' in pulsar_distances[enterprise_pulsar.name]:  # use DM distance prior for this pulsar
        p_dist = DMDistParameter(pulsar_distances[enterprise_pulsar.name][0],
                                 pulsar_distances[enterprise_pulsar.name][1])
    elif 'PX' in pulsar_distances[enterprise_pulsar.name]:  # use parallax distance prior for this pulsar
        p_dist = PXDistParameter(pulsar_distances[enterprise_pulsar.name][0],
                                 pulsar_distances[enterprise_pulsar.name][1])

    cw_wf = deterministic.cw_delay(p_dist=p_dist, **cw_delay_args)

    cw = deterministic.CWSignal(cw_wf, **CWSignal_args)

    return cw

