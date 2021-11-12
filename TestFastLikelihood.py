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
import time

import CWFastLikelihoodCython

#####################################################################
#
# READ IN DATA
#
#####################################################################

#with open('fast_like_test_psrs.pkl', 'rb') as psr_pkl:
with open('fast_like_test_psrs_all45.pkl', 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)

print(len(psrs))

#####################################################################
#
# SET UP PTA OBJECT
#
#####################################################################

freq = 1e-8

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

log_f = np.log10(freq)
log10_fgw = parameter.Constant(log_f)('0_log10_fgw')

if freq>=191.3e-9:
    m = (1./(6**(3./2)*np.pi*freq*u.Hz))*(1./4)**(3./5)*(c.c**3/c.G)
    m_max = np.log10(m.to(u.Msun).value)
else:
    m_max = 10

log10_mc = parameter.Uniform(7,m_max)('0_log10_mc')

phase0 = parameter.Uniform(0, 2*np.pi)('0_phase0')
psi = parameter.Uniform(0, np.pi)('0_psi')
cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

p_phase = parameter.Uniform(0, 2*np.pi)
p_dist = parameter.Normal(0, 1)

#log10_h = parameter.Uniform(-18, -11)('0_log10_h')
log10_h = parameter.LinearExp(-18, -11)('0_log10_h')

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

s = ef + eq + ec + rn + cw + tm

models = [s(psr) for psr in psrs]

pta = signal_base.PTA(models)

with open('fast_like_test_psrs_noisedict.json', 'r') as fp:
    noisedict = json.load(fp)

#print(noisedict)
pta.set_default_params(noisedict)

#####################################################################
#
# SET UP RANDOM PARAMETER DICTS TO CALL LIKELIHOOD WITH
#
#####################################################################

xxx = {"0_cos_gwtheta":0.5,
       "0_cos_inc":0.0,
       "0_gwphi":1.0,
       "0_log10_fgw":-8.0,
       "0_log10_h":-17.0,
       "0_log10_mc":10.0,
       "0_phase0":0.0,
       "0_psi":0.0}

for i, psr in enumerate(psrs):
    xxx[psr.name+"_cw0_p_dist"] = 0.0
    xxx[psr.name+"_cw0_p_phase"] = 0.0
    xxx[psr.name+"_red_noise_gamma"] = 3.0
    xxx[psr.name+"_red_noise_log10_A"] = -15.0

tic = time.perf_counter()
_ = pta.get_lnlikelihood(xxx)
toc = time.perf_counter()

print("Old likelihood first call runtime: {0:.1f} ms".format((toc-tic)*1000))

for i, psr in enumerate(psrs):
    xxx[psr.name+"_cw0_p_dist"] = 0.0
    xxx[psr.name+"_cw0_p_phase"] = 0.0
    xxx[psr.name+"_red_noise_gamma"] = 4.0
    xxx[psr.name+"_red_noise_log10_A"] = -16.0

Ntrial = 100

np.random.seed(1995)
rand_cos_gwtheta = np.array([pta.params[0].sample() for i in range(Ntrial)])
rand_cos_inc = np.array([pta.params[1].sample() for i in range(Ntrial)])
rand_gwphi = np.array([pta.params[2].sample() for i in range(Ntrial)])
rand_log10_h = np.array([pta.params[3].sample() for i in range(Ntrial)])
rand_log10_mc = np.array([pta.params[4].sample() for i in range(Ntrial)])
rand_phase0 = np.array([pta.params[5].sample() for i in range(Ntrial)])
rand_psi = np.array([pta.params[6].sample() for i in range(Ntrial)])

rand_p_phases = np.zeros((len(psrs),Ntrial))
for j in range(len(psrs)):
    rand_p_phases[j,:] = np.array([pta.params[8].sample() for i in range(Ntrial)])

#####################################################################
#
# SPEED TEST WITH CANONICAL LIKELIHOOD
#
#####################################################################
print("Running speed test with old likelihood...")

#do one likelihood call before timing since the first ene is always slower, because it does some storing of matrices
tic = time.perf_counter()
_ = pta.get_lnlikelihood(xxx)
toc = time.perf_counter()

print("Old likelihood first call runtime: {0:.1f} ms".format((toc-tic)*1000))

tic = time.perf_counter()

old_log_Ls = []
for i in range(Ntrial):
    xxx["0_cos_inc"] = rand_cos_inc[i]
    xxx["0_log10_h"] = rand_log10_h[i]
    xxx["0_phase0"] = rand_phase0[i]
    xxx["0_psi"] = rand_psi[i]

    for j, psr in enumerate(psrs):
        xxx[psr.name+"_cw0_p_phase"] = rand_p_phases[j,i]
    
    #print(xxx)
    old_log_Ls.append(pta.get_lnlikelihood(xxx))
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
FastLCython = CWFastLikelihoodCython.CWFastLikelihood(psrs, pta, xxx)
toc = time.perf_counter()

print("Fast likelihood setup time: {0:.3f} s".format((toc-tic)))

#####################################################################
#
# SPEED TEST WITH FAST LIKELIHOOD
#
#####################################################################
print("Running speed test with fast likelihood...")

tic = time.perf_counter()

new_log_Ls = []
for i in range(Ntrial):
    xxx["0_cos_inc"] = rand_cos_inc[i]
    xxx["0_log10_h"] = rand_log10_h[i]
    xxx["0_phase0"] = rand_phase0[i]
    xxx["0_psi"] = rand_psi[i]
    
    for j, psr in enumerate(psrs):
        xxx[psr.name+"_cw0_p_phase"] = rand_p_phases[j,i]
    
    new_log_Ls.append(FastLCython.get_lnlikelihood(xxx))
    #print("log_L_new=",str(new_log_Ls[i]))

new_log_Ls = np.array(new_log_Ls)

toc = time.perf_counter()

print("Fast likelihood runtime: {0:.3f} ms".format((toc-tic)/Ntrial*1000))
#####################################################################
#
# PLOT NEW AND OLD LIKELIHOOD
#
#####################################################################
plt.plot(old_log_Ls, np.abs((new_log_Ls-old_log_Ls)/old_log_Ls), ls='', marker='o', label='data')
plt.yscale('log')
plt.xlabel("Old log likelihood")
plt.ylabel("Fast log likelihood fractional difference")
plt.tight_layout()
plt.savefig("fast_vs_old_likelihood.png", dpi=300)

