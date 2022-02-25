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
import time

import CWFastLikelihoodNumba
import const_mcmc as cm
#####################################################################
#
# READ IN DATA
#
#####################################################################

#with open('fast_like_test_psrs.pkl', 'rb') as psr_pkl:
#with open('fast_like_test_psrs_all45.pkl', 'rb') as psr_pkl:
with open('data/fast_like_test_psrs_no_cw_no_gwb.pkl', 'rb') as psr_pkl:
    Psrs = pickle.load(psr_pkl)

psrs = Psrs*5

print(len(psrs))
print(sum([len(psr.toas) for psr in psrs]))


#####################################################################
#
# SET UP PTA OBJECT
#
#####################################################################

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
#m_max = 10
#for run speed testing
m_max = 9

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

noise_json = 'data/channelized_12p5yr_v3_full_noisedict_gp_ecorr.json'
with open(noise_json, 'r') as fp:
    noisedict = json.load(fp)

#print(noisedict)
pta.set_default_params(noisedict)

#print(pta.params)

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

####################################################################
#
# SET UP RANDOM PARAMETER DICTS TO CALL LIKELIHOOD WITH
#
#####################################################################
Ntrial = 100

samples = np.zeros((Ntrial, len(pta.params)))
sample_first = np.array([par.sample() for par in pta.params])

for i in range(Ntrial):
    samples[i,:] = np.copy(sample_first)

np.random.seed(1995)
rand_cos_inc = np.array([pta.params[1].sample() for i in range(Ntrial)])
rand_log10_h = np.array([pta.params[4].sample() for i in range(Ntrial)])
rand_phase0 = np.array([pta.params[6].sample() for i in range(Ntrial)])
rand_psi = np.array([pta.params[7].sample() for i in range(Ntrial)])

rand_p_phases = np.zeros((len(psrs),Ntrial))
for j in range(len(psrs)):
    rand_p_phases[j,:] = np.array([pta.params[9].sample() for i in range(Ntrial)])

for i in range(Ntrial):
    samples[i, par_names.index("0_cos_inc")] = rand_cos_inc[i]
    samples[i, par_names.index("0_log10_h")] = rand_log10_h[i]
    samples[i, par_names.index("0_phase0")] = rand_phase0[i]
    samples[i, par_names.index("0_psi")] = rand_psi[i]

    for j, psr in enumerate(psrs):
        samples[i, par_names.index(psr.name+"_cw0_p_phase")] = rand_p_phases[j,i]
#####################################################################
#
# SPEED TEST WITH CANONICAL LIKELIHOOD
#
#####################################################################
print("Running speed test with old likelihood...")

#do one likelihood call before timing since the first ene is always slower, because it does some storing of matrices
tic = time.perf_counter()
_ = pta.get_lnlikelihood(samples[0,:])
toc = time.perf_counter()

print("Old likelihood first call runtime: {0:.1f} ms".format((toc-tic)*1000))

tic = time.perf_counter()

old_log_Ls = []
for i in range(Ntrial):
    old_log_Ls.append(pta.get_lnlikelihood(samples[i,:]))
    #print("log_L_old=",str(old_log_Ls[i]))

old_log_Ls = np.array(old_log_Ls)

toc = time.perf_counter()

print("Old likelihood runtime: {0:.1f} ms".format((toc-tic)/Ntrial*1000))

#####################################################################
#
# SET UP FAST LIKELIHOOD OBJECT
#
#####################################################################
print("Setting up new likelihood object...")

tic = time.perf_counter()
x0 = CWFastLikelihoodNumba.CWInfo(len(pta.pulsars),samples[0,:],par_names,par_names_cw_ext)
flm = CWFastLikelihoodNumba.FastLikeMaster(psrs,pta,dict(zip(par_names, samples[0,:])),x0)
FLI = flm.get_new_FastLike(x0, dict(zip(par_names, samples[0,:])))
toc = time.perf_counter()

print("Fast likelihood setup time: {0:.3f} s".format((toc-tic)))

#####################################################################
#
# SPEED TEST WITH FAST LIKELIHOOD
#
#####################################################################
print("Running speed test with fast likelihood...")

tic = time.perf_counter()
x0.update_params(samples[0,:])
_ = FLI.get_lnlikelihood(x0)
toc = time.perf_counter()

print("Fast likelihood first call runtime: {0:.3f} ms".format((toc-tic)*1000))

tic = time.perf_counter()

Nmult = 1000

new_log_Ls = []
for i in range(Nmult*Ntrial):
    x0.update_params(samples[i%Ntrial,:])
    new_log_Ls.append(FLI.get_lnlikelihood(x0))
    #print("log_L_new=",str(new_log_Ls[i]))

new_log_Ls = np.array(new_log_Ls)

toc = time.perf_counter()

print("Fast likelihood runtime: {0:.3f} ms".format((toc-tic)/Ntrial/Nmult*1000))

#####################################################################
#
# SPEED TEST OF SHAPE PARAMETER UPDATES
#
#####################################################################
print("Timing common shape parameter updates -------------------------------------------------------")
Ntrial = 100

samples = np.zeros((Ntrial, len(pta.params)))
for i in range(Ntrial):
    samples[i,:] = np.copy(sample_first)

np.random.seed(1995)
rand_cos_gwtheta = np.array([pta.params[0].sample() for i in range(Ntrial)])
rand_gwphi = np.array([pta.params[2].sample() for i in range(Ntrial)])
rand_log10_fgw = np.array([pta.params[3].sample() for i in range(Ntrial)])
rand_log10_mc = np.array([pta.params[5].sample() for i in range(Ntrial)])

for i in range(Ntrial):
    samples[i, par_names.index("0_cos_gwtheta")] = rand_cos_gwtheta[i]
    samples[i, par_names.index("0_gwphi")] = rand_gwphi[i]
    samples[i, par_names.index("0_log10_fgw")] = rand_log10_fgw[i]
    samples[i, par_names.index("0_log10_mc")] = rand_log10_mc[i]

#check old likelihood runtime to make sure it has the same speed here
tic = time.perf_counter()

old_log_Ls = []
for i in range(Ntrial):
    #for j in range(8):
    #    print(par_names[j])
    #    print(samples[i,j])
    old_log_Ls.append(pta.get_lnlikelihood(samples[i,:]))
    #print("log_L_old=",str(old_log_Ls[i]))

old_log_Ls = np.array(old_log_Ls)

toc = time.perf_counter()

print("Old likelihood runtime: {0:.1f} ms".format((toc-tic)/Ntrial*1000))

tic = time.perf_counter()
x0.update_params(samples[0,:])
FLI.update_intrinsic_params(x0)
_ = FLI.get_lnlikelihood(x0)
toc = time.perf_counter()

print("Fast likelihood first call runtime: {0:.3f} ms".format((toc-tic)*1000))

#new likelihood with common parameter update
tic = time.perf_counter()

Nmult = 1

new_log_Ls = []
for i in range(Nmult*Ntrial):
    x0.update_params(samples[i%Ntrial,:])
    FLI.update_intrinsic_params(x0)
    new_log_Ls.append(FLI.get_lnlikelihood(x0))
    #print("log_L_new=",str(new_log_Ls[i]))

new_log_Ls = np.array(new_log_Ls)

toc = time.perf_counter()

print("Fast likelihood runtime: {0:.3f} ms".format((toc-tic)/Ntrial/Nmult*1000))

#####################################################################
#
# SPEED TEST OF DISTANCE UPDATES
#
#####################################################################
print("Timing distance updates -------------------------------------------------------")
Ntrial = 100

samples = np.zeros((Ntrial, len(pta.params)))
for i in range(Ntrial):
    samples[i,:] = np.copy(sample_first)

np.random.seed(1995)

rand_p_dists = np.zeros((len(psrs),Ntrial))
for j in range(len(psrs)):
    rand_p_dists[j,:] = np.array([pta.params[8].sample() for i in range(Ntrial)])

for i in range(Ntrial):
    for j, psr in enumerate(psrs):
        samples[i, par_names.index(psr.name+"_cw0_p_dist")] = rand_p_dists[j,i]

#check old likelihood runtime to make sure it has the same speed here
tic = time.perf_counter()

old_log_Ls = []
for i in range(Ntrial):
    #for j in range(8):
    #    print(par_names[j])
    #    print(samples[i,j])
    old_log_Ls.append(pta.get_lnlikelihood(samples[i,:]))
    #print("log_L_old=",str(old_log_Ls[i]))

old_log_Ls = np.array(old_log_Ls)

toc = time.perf_counter()

print("Old likelihood runtime: {0:.1f} ms".format((toc-tic)/Ntrial*1000))

tic = time.perf_counter()
x0.update_params(samples[0,:])
FLI.update_intrinsic_params(x0)
#print(x0.cos_gwtheta, FLI.cos_gwtheta)
FLI.update_pulsar_distances(x0, List(range(len(psrs))))
_ = FLI.get_lnlikelihood(x0)
toc = time.perf_counter()

print("Fast likelihood first call runtime: {0:.3f} ms".format((toc-tic)*1000))

#new likelihood with common parameter update
tic = time.perf_counter()

Nmult = 1

new_log_Ls = []
for i in range(Nmult*Ntrial):
    x0.update_params(samples[i%Ntrial,:])
    FLI.update_pulsar_distances(x0, List(range(len(psrs))))
    new_log_Ls.append(FLI.get_lnlikelihood(x0))
    #print("log_L_new=",str(new_log_Ls[i]))

new_log_Ls = np.array(new_log_Ls)

toc = time.perf_counter()

print("Fast likelihood runtime: {0:.3f} ms".format((toc-tic)/Ntrial/Nmult*1000))

#####################################################################
#
# SPEED TEST OF RN UPDATES
#
#####################################################################
print("Timing red noise updates -------------------------------------------------------")
Ntrial = 100

samples = np.zeros((Ntrial, len(pta.params)))
for i in range(Ntrial):
    samples[i,:] = np.copy(sample_first)

np.random.seed(1995)

rand_red_noise_gammas = np.zeros((len(psrs),Ntrial))
rand_red_noise_log10_As = np.zeros((len(psrs),Ntrial))
for j in range(len(psrs)):
    rand_red_noise_gammas[j,:] = np.array([pta.params[10].sample() for i in range(Ntrial)])
    rand_red_noise_log10_As[j,:] = np.array([pta.params[11].sample() for i in range(Ntrial)])

for i in range(Ntrial):
    for j, psr in enumerate(psrs):
        samples[i, par_names.index(psr.name+"_red_noise_gamma")] = rand_red_noise_gammas[j,i]
        samples[i, par_names.index(psr.name+"_red_noise_log10_A")] = rand_red_noise_log10_As[j,i]

#check old likelihood runtime to make sure it has the same speed here
tic = time.perf_counter()

old_log_Ls = []
for i in range(Ntrial):
    #for j in range(8):
    #    print(par_names[j])
    #    print(samples[i,j])
    old_log_Ls.append(pta.get_lnlikelihood(samples[i,:]))
    #print("log_L_old=",str(old_log_Ls[i]))

old_log_Ls = np.array(old_log_Ls)

toc = time.perf_counter()

print("Old likelihood runtime: {0:.1f} ms".format((toc-tic)/Ntrial*1000))

tic = time.perf_counter()
x0.update_params(samples[0,:])
FLI.update_intrinsic_params(x0)
#print(x0.cos_gwtheta, FLI.cos_gwtheta)
flm.recompute_FastLike(FLI,x0,dict(zip(par_names, samples[0,:])))
_ = FLI.get_lnlikelihood(x0)
toc = time.perf_counter()

print("Fast likelihood first call runtime: {0:.3f} ms".format((toc-tic)*1000))

#new likelihood with common parameter update
tic = time.perf_counter()

Nmult = 1

new_log_Ls = []
for i in range(Nmult*Ntrial):
    x0.update_params(samples[i%Ntrial,:])
    flm.recompute_FastLike(FLI,x0,dict(zip(par_names, samples[i%Ntrial,:])))
    new_log_Ls.append(FLI.get_lnlikelihood(x0))
    #print("log_L_new=",str(new_log_Ls[i]))

new_log_Ls = np.array(new_log_Ls)

toc = time.perf_counter()

print("Fast likelihood runtime: {0:.3f} ms".format((toc-tic)/Ntrial/Nmult*1000))



#####################################################################
#
# PLOT NEW AND OLD LIKELIHOOD
#
#####################################################################
"""
plt.plot(old_log_Ls, np.abs((new_log_Ls-old_log_Ls)/old_log_Ls), ls='', marker='o', label='data')
plt.yscale('log')
plt.xlabel("Old log likelihood")
plt.ylabel("Fast log likelihood fractional difference")
plt.tight_layout()
plt.savefig("fast_vs_old_likelihood.png", dpi=300)
"""
