"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
Helpers for Getting Prior Draws"""

import numpy as np
#np.seterr(all='raise')
import numba as nb
from numba import njit
from numba.experimental import jitclass
from numba.typed import List

#import scipy as sc
#from scipy.stats.uniform import uni_pdf
#from scipy.stats.multivariate_normal import norm_pdf
#from numba_stats import norm as norm_numba
#from numba_stats import uniform as uniform_numba

################################################################################
#
#MY VERSION OF GETTING THE LOG PRIOR
#
################################################################################
class FastPrior:
    """helper class to set up information about priors"""
    def __init__(self, pta, psrs, par_names_cw_ext):
        """pta is an enterprise pta and par_names_cw_ext is a list of the extrinsic parameters"""
        self.pta = pta
        self.param_names = List(pta.param_names)
        uniform_pars = []
        uf_lows = []
        uf_highs = []
        lin_exp_pars = []
        le_lows = []
        le_highs = []
        normal_pars = []
        nm_mus = []
        nm_sigs = []
        #track parameters that are normal and distance so we can apply a cutoff to them
        normal_dist_pars = []
        nm_dist_lows = []

        #store ranges for extrinsic parameters
        cw_ext_lows = []
        cw_ext_highs = []
        cw_pars = []

        for par in self.pta.params:
            #print(par)
            if "Uniform" in par._typename:
                uniform_pars.append(par.name)
                uf_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                uf_highs.append(float(par._typename.split('=')[2][:-1]))
            elif "LinearExp" in par._typename:
                lin_exp_pars.append(par.name)
                le_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                le_highs.append(float(par._typename.split('=')[2][:-1]))
            elif "Normal" in par._typename:
                normal_pars.append(par.name)
                nm_mus.append(float(par._typename.split('=')[1].split(',')[0]))
                nm_sigs.append(float(par._typename.split('=')[2][:-1]))
                if "_cw0_p_dist" in par.name:
                    #find the corresponding pulsar distance so we can append it
                    normal_dist_pars.append(par.name)
                    for psr in psrs:
                        if psr.name in par.name:
                            #this should be the lower cutoff of dist_delta such that dist_mu + dist_sigma*dist_delta=0
                            nm_dist_lows.append(-psr.pdist[0]/psr.pdist[1])
                            break

            if par.name in par_names_cw_ext:
                cw_pars.append(par.name)
                cw_ext_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                cw_ext_highs.append(float(par._typename.split('=')[2][:-1]))


        self.uniform_lows = np.array(uf_lows)
        self.uniform_highs = np.array(uf_highs)
        self.lin_exp_lows = np.array(le_lows)
        self.lin_exp_highs = np.array(le_highs)
        self.normal_mus = np.array(nm_mus)
        self.normal_sigs = np.array(nm_sigs)
        self.uniform_par_ids = np.array([self.param_names.index(u_par) for u_par in uniform_pars], dtype='int')
        self.lin_exp_par_ids = np.array([self.param_names.index(l_par) for l_par in lin_exp_pars], dtype='int')
        self.normal_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_pars], dtype='int')
        self.cw_ext_par_ids = np.array([self.param_names.index(c_par) for c_par in cw_pars], dtype='int')

        self.cw_ext_lows = np.array(cw_ext_lows)
        self.cw_ext_highs = np.array(cw_ext_highs)

        #logic for cutting off normally distributed distances so they don't go below 0
        self.normal_dist_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_dist_pars], dtype='int')
        self.normal_dist_lows = np.array(nm_dist_lows)
        self.normal_dist_highs = np.full(self.normal_dist_lows.size,np.inf)

        self.cut_lows = np.hstack([self.uniform_lows,self.lin_exp_lows,self.normal_dist_lows])
        self.cut_highs = np.hstack([self.uniform_highs,self.lin_exp_highs,self.normal_dist_highs])
        self.cut_par_ids = np.hstack([self.uniform_par_ids,self.lin_exp_par_ids,self.normal_dist_par_ids])

        #uniform prior is independent of value
        self.global_uniform = 0.
        for itrp in range(self.uniform_lows.size):
            low = self.uniform_lows[itrp]
            high = self.uniform_highs[itrp]
            self.global_uniform += -np.log(high-low)

        #linear exponential prior has component independent of value
        self.global_lin_exp = 0.
        for itrp in range(self.lin_exp_lows.size):
            low = self.lin_exp_lows[itrp]
            high = self.lin_exp_highs[itrp]
            self.global_lin_exp += np.log(np.log(10))-np.log(10 ** high - 10 ** low)
            #self.global_lin_exp += -np.log(high-low)

        #normal prior has component independent of value
        self.global_normal = 0.
        for itrp in range(self.normal_mus.size):
            self.global_normal += -np.log(2*np.pi)/2.

        #part of the likelihood that is the same independent of the parameter values for all points with finite log prior
        self.global_common = self.global_uniform+self.global_lin_exp+self.global_normal

    def get_lnprior(self, x0):
        """wrapper to get ln prior"""
        return get_lnprior_helper(x0, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs,\
                                      self.global_common)

    def get_sample(self, idx):
        """wrapper to quickly return random prior draw for the (idx)th parameter"""
        return get_sample_helper(idx, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs)

@njit()
def get_sample_helper_full(n_par,uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs):
    """jittable helper for prior draws"""
    res = np.zeros(n_par)
    for itrp,idx in enumerate(uniform_par_ids):
        res[idx] = np.random.uniform(uniform_lows[itrp], uniform_highs[itrp])
    for itrp,idx in enumerate(lin_exp_par_ids):
        res[idx] = np.log10(np.random.uniform(10**lin_exp_lows[itrp], 10**lin_exp_highs[itrp]))
    for itrp,idx in enumerate(normal_par_ids):
        res[idx] = np.random.normal(normal_mus[itrp], normal_sigs[itrp])
    return res

def get_sample_helper(idx, uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs):
    """jittable helper for prior draws"""
    if idx in uniform_par_ids:
        iii = np.argmax(uniform_par_ids==idx)
        return np.random.uniform(uniform_lows[iii], uniform_highs[iii])
    elif idx in lin_exp_par_ids:
        iii = np.argmax(lin_exp_par_ids==idx)
        return np.log10(np.random.uniform(10**lin_exp_lows[iii], 10**lin_exp_highs[iii]))
    else:
        iii = np.argmax(normal_par_ids==idx)
        return np.random.normal(normal_mus[iii], normal_sigs[iii])

@njit()
def get_lnprior_helper(x0, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           global_common):
    """jittable helper for calculating the log prior"""
    n = uniform_par_ids.size
    #sum of uniform priors is either -inf or the same for all values
    #global_uniform = 0.
    #for itrp in range(n):
    #    low = uniform_lows[itrp]
    #    high = uniform_highs[itrp]
    #    global_uniform += -np.log(high-low)
    #global_common = global_uniform

    log_prior = global_common


    #loop through uniform parameters and make sure all are in range
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf

    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        #loop through linear exponential parameters
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf #from enterprise
        else:
            #log_prior += np.log(np.log(10))-np.log(10 ** high - 10 ** low) + value*np.log(10)#from enterprise
            #part is folded in global_common
            log_prior += value*np.log(10)#from enterprise

    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        #loop through normal parameters
        value = x0[par_id]
        #part is handeled in global_common
        #log_prior += -np.log(2*np.pi)/2.-(value-mu)**2/(2*sig**2) #log_pdf_got
        log_prior += -(value-mu)**2/(2*sig**2) #log_pdf_got

    return log_prior
    #return get_lnprior_helper_array(x0s, uniform_par_ids, uniform_lows, uniform_highs,
    #                       lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
    #                       normal_par_ids, normal_mus, normal_sigs)[0]

def get_lnprior(x0,FPI):
    """wrapper to get lnprior from jitted helper"""
    return get_lnprior_helper(x0, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                         FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                         FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                         FPI.global_common)


def get_lnprior_array(samples,FPI):
    """wrapper to get lnprior from jitted helper"""
    return get_lnprior_helper_array(samples, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                           FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                           FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                           FPI.global_common)

@njit()
def get_lnprior_helper_array(x0s, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           global_common):
    """jittable helper for calculating the log prior"""
    npoint = x0s.shape[0]

    n = uniform_par_ids.size
    #sum of uniform priors is either -inf or the same for all values
    #global_uniform = 0.
    #for itrp in range(n):
    #    low = uniform_lows[itrp]
    #    high = uniform_highs[itrp]
    #    global_uniform += -np.log(high-low)
    #global_common = global_uniform


    log_priors = np.zeros(npoint)+global_common


    #loop through uniform parameters and make sure all are in range
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf
        #for itrk in range(npoint):
        #    value = x0s[itrk,par_id]
        #    #log_prior += np.log(uniform_numba.pdf(value, low, high-low))

        #    if not low<=value<=high:
        #        #print('invalid input to uniform lnprior at ',itrk,itrp,', ',value,' not in range=[',low,',',high,']')
        #        #raise ValueError('hello?')
        #        log_priors[itrk] = -np.inf

    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        #loop through linear exponential parameters
        #log_priors[:] += np.log(np.log(10))-np.log(10 ** high - 10 ** low) + x0s[:,par_id]*np.log(10)#from enterprise
        #part is folded in global_common
        log_priors[:] += x0s[:,par_id]*np.log(10)#from enterprise
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf
        #for itrk in range(npoint):
        #    value = x0s[itrk,par_id]
        #    if low>value or value>high:
        #        log_priors[itrk] = -np.inf #from enterprise
        #    else:
        #        log_priors[itrk] += np.log(np.log(10))-np.log(10 ** high - 10 ** low) + value*np.log(10)#from enterprise

    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        #log_priors[:] += -np.log(2*np.pi)/2.-(x0s[:,par_id]-mu)**2/(2*sig**2) #log_pdf_got
        #part is handled in global_common
        log_priors[:] += -(x0s[:,par_id]-mu)**2/(2*sig**2) #log_pdf_got
        #for itrk in range(npoint):
        ##loop through normal parameters
        #    value = x0s[itrk,par_id]
        #    log_priors[itrk] += -np.log(2*np.pi)/2.-(value-mu)**2/(2*sig**2) #log_pdf_got

    return log_priors

def get_sample_idxs(old_point,idx_choose,FPI):
    """get just some indexes reset for a uniform sample, actually just gets a whole new prior draw and picks the idxs needed"""
    new_point = old_point.copy()
    res = get_sample_full(new_point.size,FPI)
    for idx in idx_choose:
        new_point[idx] = res[idx]

    return new_point

def get_sample_full(n_par,FPI):
    """helper to get a sample with the specified indexes redrawn"""
    new_point = get_sample_helper_full(n_par, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)

    #new_point = old_point.copy()
    #for idx in idx_choose:
    #    new_point[idx] = get_sample_helper(idx, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
    #                                            FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
    #                                            FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs)
    return new_point

@jitclass([('uniform_par_ids',nb.int64[:]),('uniform_lows',nb.float64[:]),('uniform_highs',nb.float64[:]),\
           ('lin_exp_par_ids',nb.int64[:]),('lin_exp_lows',nb.float64[:]),('lin_exp_highs',nb.float64[:]),\
           ('normal_par_ids',nb.int64[:]),('normal_mus',nb.float64[:]),('normal_sigs',nb.float64[:]),\
           ('cut_par_ids',nb.int64[:]),('cut_lows',nb.float64[:]),('cut_highs',nb.float64[:]),\
           ('cw_ext_par_ids',nb.int64[:]),('cw_ext_lows',nb.float64[:]),('cw_ext_highs',nb.float64[:]),\
           ('global_common',nb.float64)])
class FastPriorInfo:
    """simple jitclass to store the various elements of fast prior calculation in a way that can be accessed quickly from a numba environment"""
    def __init__(self, uniform_par_ids, uniform_lows, uniform_highs, lin_exp_par_ids, lin_exp_lows, lin_exp_highs, normal_par_ids, normal_mus, normal_sigs,cut_par_ids,cut_lows,cut_highs,cw_ext_par_ids,cw_ext_lows,cw_ext_highs,global_common):
        self.uniform_par_ids = uniform_par_ids
        self.uniform_lows = uniform_lows
        self.uniform_highs = uniform_highs
        self.lin_exp_par_ids = lin_exp_par_ids
        self.lin_exp_lows = lin_exp_lows
        self.lin_exp_highs = lin_exp_highs
        self.normal_par_ids = normal_par_ids
        self.normal_mus = normal_mus
        self.normal_sigs = normal_sigs
        self.cut_par_ids = cut_par_ids
        self.cut_lows = cut_lows
        self.cut_highs = cut_highs
        self.cw_ext_par_ids = cw_ext_par_ids
        self.cw_ext_lows = cw_ext_lows
        self.cw_ext_highs = cw_ext_highs
        self.global_common = global_common

def get_FastPriorInfo(pta,psrs,par_names_cw_ext):
    """get FastPriorInfo object from pta"""
    fp_loc = FastPrior(pta,psrs,par_names_cw_ext)
    FPI = FastPriorInfo(fp_loc.uniform_par_ids, fp_loc.uniform_lows, fp_loc.uniform_highs,\
                        fp_loc.lin_exp_par_ids, fp_loc.lin_exp_lows, fp_loc.lin_exp_highs,\
                        fp_loc.normal_par_ids, fp_loc.normal_mus, fp_loc.normal_sigs,\
                        fp_loc.cut_par_ids,fp_loc.cut_lows,fp_loc.cut_highs,\
                        fp_loc.cw_ext_par_ids,fp_loc.cw_ext_lows,fp_loc.cw_ext_highs,\
                        fp_loc.global_common)
    return FPI
