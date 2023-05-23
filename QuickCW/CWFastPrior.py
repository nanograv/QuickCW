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
    """helper class to set up information about priors

    :param pta:                 enterprise PTA object
    :param psrs:                list of enterprise pulsar objects
    :param par_names_cw_ext:    list of projection parameters (previously called extrinsic parameters)
    """
    def __init__(self, pta, psrs, par_names_cw_ext):
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
        dm_pars = []
        dm_dists = []
        dm_errs = []
        dm_dist_lows = []
        px_pars = []
        px_mus = []
        px_errs = []
        px_dist_lows = []
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
            elif "DMDist" in par._typename:
                dm_pars.append(par.name)
                dm_dists.append(float(par._typename.split('=')[1].split(',')[0]))
                dm_errs.append(float(par._typename.split('=')[2][:-1]))
                dm_dist_lows.append(0.0)
            elif "PXDist" in par._typename:
                px_pars.append(par.name)
                px_dist = float(par._typename.split('=')[1].split(',')[0])
                px_dist_err = float(par._typename.split('=')[2][:-1])
                px_mus.append(1/px_dist)
                px_errs.append(px_dist_err/px_dist**2)
                px_dist_lows.append(0.0)

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
        self.dm_dists = np.array(dm_dists)
        self.dm_errs = np.array(dm_errs)
        self.px_mus = np.array(px_mus)
        self.px_errs = np.array(px_errs)
        self.uniform_par_ids = np.array([self.param_names.index(u_par) for u_par in uniform_pars], dtype='int')
        self.lin_exp_par_ids = np.array([self.param_names.index(l_par) for l_par in lin_exp_pars], dtype='int')
        self.normal_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_pars], dtype='int')
        self.dm_par_ids = np.array([self.param_names.index(dm_par) for dm_par in dm_pars], dtype='int')
        self.px_par_ids = np.array([self.param_names.index(px_par) for px_par in px_pars], dtype='int')
        self.cw_ext_par_ids = np.array([self.param_names.index(c_par) for c_par in cw_pars], dtype='int')

        self.cw_ext_lows = np.array(cw_ext_lows)
        self.cw_ext_highs = np.array(cw_ext_highs)

        #logic for cutting off normally distributed distances so they don't go below 0
        self.normal_dist_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_dist_pars], dtype='int')
        self.normal_dist_lows = np.array(nm_dist_lows)
        self.normal_dist_highs = np.full(self.normal_dist_lows.size,np.inf)

        self.dm_dist_lows = np.array(dm_dist_lows)
        self.dm_dist_highs = np.full(self.dm_dist_lows.size,np.inf)
        self.px_dist_lows = np.array(px_dist_lows)
        self.px_dist_highs = np.full(self.px_dist_lows.size,np.inf)

        self.cut_lows = np.hstack([self.uniform_lows,self.lin_exp_lows,self.normal_dist_lows,self.dm_dist_lows,self.px_dist_lows])
        self.cut_highs = np.hstack([self.uniform_highs,self.lin_exp_highs,self.normal_dist_highs,self.dm_dist_highs,self.px_dist_highs])
        self.cut_par_ids = np.hstack([self.uniform_par_ids,self.lin_exp_par_ids,self.normal_dist_par_ids,self.dm_par_ids,self.px_par_ids])

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

        #dm prior has component independent of the value
        self.global_dm = 0
        for itrp in range(self.dm_dists.size):
            dist = self.dm_dists[itrp]
            err = self.dm_errs[itrp]

            boxheight = 1/((dist+err)-(dist-err))
            gaussheight = 1/(np.sqrt(2*np.pi*(0.25*err)**2))

            area = 1+1*boxheight/gaussheight
            
            self.global_dm += np.log(boxheight/area)

        #part of the likelihood that is the same independent of the parameter values for all points with finite log prior
        self.global_common = self.global_uniform+self.global_lin_exp+self.global_normal+self.global_dm

    def get_lnprior(self, x0):
        """wrapper to get ln prior"""
        return get_lnprior_helper(x0, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs,\
                                      self.dm_par_ids, self.dm_dists, self.dm_errs,\
                                      self.px_par_ids, self.px_mus, self.px_errs,\
                                      self.global_common)

    def get_sample(self, idx):
        """wrapper to quickly return random prior draw for the (idx)th parameter"""
        return get_sample_helper(idx, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs,\
                                      self.dm_par_ids, self.dm_dists, self.dm_errs,\
                                      self.px_par_ids, self.px_mus, self.px_errs)

@njit()
def get_sample_helper_full(n_par,uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs,
                           dm_par_ids, dm_dists, dm_errs,
                           px_par_ids, px_mus, px_errs):
    """jittable helper for prior draws

    :param n_par:           Number of parameters in likelihood/prior
    :param uniform_par_ids: Indices of parameters with uniform prior
    :param uniform_lows:    Lower prior bounds of uniform prior parameters
    :param uniform_highs:   Upper prior bounds of uniform prior parameters
    :param lin_exp_par_ids: Indices of parameters with linear exponential prior
    :param lin_exp_lows:    Lower prior bounds of linear exponential prior parameters
    :param lin_exp_highs:   Upper prior bounds of linear exponential prior parameters
    :param normal_par_ids:  Indices of parameters with normal prior
    :param normal_mus:      Means of normal prior parameters
    :param normal_sigs:     Standard deviations of normal prior parameters
    :param dm_par_ids:      Indices of parameters with DM distance prior
    :param dm_dists:        Mean distances of DM distance prior parameters
    :param dm_errs:         Errors of DM distance prior parameters
    :param px_par_ids:      Indices of parameters with parallax distance prior
    :param px_mus:          Means of parallax distance prior parameters
    :param px_errs:         Errors of parallax distance prior parameters

    :return res:            Array of prior draws
    """
    res = np.zeros(n_par)
    for itrp,idx in enumerate(uniform_par_ids):
        res[idx] = np.random.uniform(uniform_lows[itrp], uniform_highs[itrp])
    for itrp,idx in enumerate(lin_exp_par_ids):
        res[idx] = np.log10(np.random.uniform(10**lin_exp_lows[itrp], 10**lin_exp_highs[itrp]))
    for itrp,idx in enumerate(normal_par_ids):
        res[idx] = np.random.normal(normal_mus[itrp], normal_sigs[itrp])
    for itrp,idx in enumerate(dm_par_ids):
        boxheight = 1/((dm_dists[itrp]+dm_errs[itrp])-(dm_dists[itrp]-dm_errs[itrp]))
        gaussheight = 1/(np.sqrt(2*np.pi*(0.25*dm_errs[itrp])**2))
        area = 1+1*boxheight/gaussheight
        
        #probability of being in the uniform part
        boxprob = 1/area
        
        #decide if we are in the middle or not
        alpha = np.random.uniform(0.0,1.0)
        if alpha<boxprob:
            res[idx] = np.random.uniform(dm_dists[itrp]-dm_errs[itrp], dm_dists[itrp]+dm_errs[itrp])
        else:
            x = np.random.normal(0, 0.25*dm_errs[itrp])
            if x>0.0:
                res[idx] = x+dm_dists[itrp]+dm_errs[itrp]
            else:
                res[idx] = x+dm_dists[itrp]-dm_errs[itrp]
    for itrp,idx in enumerate(px_par_ids):
        res[idx] = 1/np.random.normal(px_mus[itrp], px_errs[itrp])
    return res

def get_sample_helper(idx, uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs,
                           dm_par_ids, dm_dists, dm_errs,
                           px_par_ids, px_mus, px_errs):
    """jittable helper for prior draws - same as get_sample_helper_full but only for a single parameter

    :param idx:             Index of parameters for which we want a prior draw
    :param uniform_par_ids: Indices of parameters with uniform prior
    :param uniform_lows:    Lower prior bounds of uniform prior parameters
    :param uniform_highs:   Upper prior bounds of uniform prior parameters
    :param lin_exp_par_ids: Indices of parameters with linear exponential prior
    :param lin_exp_lows:    Lower prior bounds of linear exponential prior parameters
    :param lin_exp_highs:   Upper prior bounds of linear exponential prior parameters
    :param normal_par_ids:  Indices of parameters with normal prior
    :param normal_mus:      Means of normal prior parameters
    :param normal_sigs:     Standard deviations of normal prior parameters
    :param dm_par_ids:      Indices of parameters with DM distance prior
    :param dm_dists:        Mean distances of DM distance prior parameters
    :param dm_errs:         Errors of DM distance prior parameters
    :param px_par_ids:      Indices of parameters with parallax distance prior
    :param px_mus:          Means of parallax distance prior parameters
    :param px_errs:         Errors of parallax distance prior parameters

    :return:                Prior draw of the parameter of interest
    """
    if idx in uniform_par_ids:
        iii = np.argmax(uniform_par_ids==idx)
        return np.random.uniform(uniform_lows[iii], uniform_highs[iii])
    elif idx in lin_exp_par_ids:
        iii = np.argmax(lin_exp_par_ids==idx)
        return np.log10(np.random.uniform(10**lin_exp_lows[iii], 10**lin_exp_highs[iii]))
    elif idx in normal_par_ids:
        iii = np.argmax(normal_par_ids==idx)
        return np.random.normal(normal_mus[iii], normal_sigs[iii])
    elif idx in dm_par_ids:
        iii = np.argmax(dm_par_ids==idx)
        boxheight = 1/((dm_dists[iii]+dm_errs[iii])-(dm_dists[iii]-dm_errs[iii]))
        gaussheight = 1/(np.sqrt(2*np.pi*(0.25*dm_errs[iii])**2))
        area = 1+1*boxheight/gaussheight

        #probability of being in the uniform part
        boxprob = 1/area

        #decide if we are in the middle or not
        alpha = np.random.uniform(0.0,1.0)
        if alpha<boxprob:
            return np.random.uniform(dm_dists[iii]-dm_errs[iii], dm_dists[iii]+dm_errs[iii])
        else:
            x = np.random.normal(0, 0.25*dm_errs[iii])
            if x>0.0:
                return x+dm_dists[iii]+dm_errs[iii]
            else:
                return x+dm_dists[iii]-dm_errs[iii]
    else:
        iii = np.argmax(px_par_ids==idx)
        return 1/np.random.normal(px_mus[iii], px_errs[iii])

@njit()
def get_lnprior_helper(x0, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           dm_par_ids, dm_dists, dm_errs,\
                           px_par_ids, px_mus, px_errs,\
                           global_common):
    """jittable helper for calculating the log prior

    :param x0:              Array of parameters for which we want to calculate the prior
    :param uniform_par_ids: Indices of parameters with uniform prior
    :param uniform_lows:    Lower prior bounds of uniform prior parameters
    :param uniform_highs:   Upper prior bounds of uniform prior parameters
    :param lin_exp_par_ids: Indices of parameters with linear exponential prior
    :param lin_exp_lows:    Lower prior bounds of linear exponential prior parameters
    :param lin_exp_highs:   Upper prior bounds of linear exponential prior parameters
    :param normal_par_ids:  Indices of parameters with normal prior
    :param normal_mus:      Means of normal prior parameters
    :param normal_sigs:     Standard deviations of normal prior parameters
    :param dm_par_ids:      Indices of parameters with DM distance prior
    :param dm_dists:        Mean distances of DM distance prior parameters
    :param dm_errs:         Errors of DM distance prior parameters
    :param px_par_ids:      Indices of parameters with parallax distance prior
    :param px_mus:          Means of parallax distance prior parameters
    :param px_errs:         Errors of parallax distance prior parameters
    :param global_common:   Part of the log prior independent of the parameter values

    :return log_prior:      Log prior
    """
    log_prior = global_common

    #loop through uniform parameters and make sure all are in range
    n = uniform_par_ids.size
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf

    #loop through linear exponential parameters
    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf #from enterprise
        else:
            log_prior += value*np.log(10)#from enterprise

    #loop through normal parameters
    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        value = x0[par_id]
        log_prior += -(value-mu)**2/(2*sig**2) #log_pdf_got

    #loop through dm distance parameters
    l = dm_par_ids.size
    for itrp in range(l):
        dist = dm_dists[itrp]
        err = dm_errs[itrp]
        par_id = dm_par_ids[itrp]

        value = x0[par_id]

        if value<=(dist-err):
            log_prior += -(value-(dist-err))**2/(2*(0.25*err)**2)
        elif value>=(dist+err):
            log_prior += -(value-(dist+err))**2/(2*(0.25*err)**2)

    #loop trhough parallax distance parameters
    ll = px_par_ids.size
    for itrp in range(ll):
        pi = px_mus[itrp]
        pi_err = px_errs[itrp]
        par_id = px_par_ids[itrp]

        value = x0[par_id]

        log_prior += np.log(1/(np.sqrt(2*np.pi)*pi_err*value**2)*np.exp(-(pi-value**(-1))**2/(2*pi_err**2)))

    return log_prior

def get_lnprior(x0,FPI):
    """wrapper to get lnprior from jitted helper

    :param x0:  Array of parameter values for which we want to calculate the prior
    :param FPI: FastPriorInfo object

    :return:    log prior
    """
    return get_lnprior_helper(x0, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                         FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                         FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                         FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                         FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                         FPI.global_common)


def get_lnprior_array(samples,FPI):
    """wrapper to get lnprior from jitted helper

    :param samples: 2d array of parameter sets for which we want to calculate the prior
    :param FPI:     FastPriorInfo object

    :return:        Array of log prior values
    """
    return get_lnprior_helper_array(samples, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                           FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                           FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                           FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                           FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                           FPI.global_common)

@njit()
def get_lnprior_helper_array(x0s, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           dm_par_ids, dm_dists, dm_errs,\
                           px_par_ids, px_mus, px_errs,\
                           global_common):
    """jittable helper for calculating the log prior for an array of points
    same as get_lnprior_helper, but can be called on an array of parameters

    :param x0s:             2D array of multiple parameter sets for which we want to calculate the prior
    :param uniform_par_ids: Indices of parameters with uniform prior
    :param uniform_lows:    Lower prior bounds of uniform prior parameters
    :param uniform_highs:   Upper prior bounds of uniform prior parameters
    :param lin_exp_par_ids: Indices of parameters with linear exponential prior
    :param lin_exp_lows:    Lower prior bounds of linear exponential prior parameters
    :param lin_exp_highs:   Upper prior bounds of linear exponential prior parameters
    :param normal_par_ids:  Indices of parameters with normal prior
    :param normal_mus:      Means of normal prior parameters
    :param normal_sigs:     Standard deviations of normal prior parameters
    :param dm_par_ids:      Indices of parameters with DM distance prior
    :param dm_dists:        Mean distances of DM distance prior parameters
    :param dm_errs:         Errors of DM distance prior parameters
    :param px_par_ids:      Indices of parameters with parallax distance prior
    :param px_mus:          Means of parallax distance prior parameters
    :param px_errs:         Errors of parallax distance prior parameters
    :param global_common:   Part of the log prior independent of the parameter values

    :return log_priors:     Array of log prior values
    """
    npoint = x0s.shape[0]

    log_priors = np.zeros(npoint)+global_common

    #loop through uniform parameters and make sure all are in range
    n = uniform_par_ids.size
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf

    #loop through linear exponential parameters
    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        log_priors[:] += x0s[:,par_id]*np.log(10)#from enterprise
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf

    #loop through normal parameters
    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        log_priors[:] += -(x0s[:,par_id]-mu)**2/(2*sig**2) #log_pdf_got

    #loop through dm distance parameters
    l = dm_par_ids.size
    for itrp in range(l):
        dist = dm_dists[itrp]
        err = dm_errs[itrp]
        par_id = dm_par_ids[itrp]

        value = x0s[:,par_id]

        log_priors[value<=(dist-err)] += -(value-(dist-err))**2/(2*(0.25*err)**2)
        log_priors[value>=(dist+err)] += -(value-(dist+err))**2/(2*(0.25*err)**2)

    #loop trhough parallax distance parameters
    ll = px_par_ids.size
    for itrp in range(ll):
        pi = px_mus[itrp]
        pi_err = px_errs[itrp]
        par_id = px_par_ids[itrp]

        value = x0s[:,par_id]

        log_priors[:] += np.log(1/(np.sqrt(2*np.pi)*pi_err*value**2)*np.exp(-(pi-value**(-1))**2/(2*pi_err**2)))

    return log_priors

def get_sample_idxs(old_point,idx_choose,FPI):
    """get just some indexes drawn from a prior

    :param old_point:   Old array of parameters
    :param idx_choose:  List of indices for which we want to drawn from the prior
    :param FPI:         FastPriorInfo object

    :return new_point:  New array of parameters
    """
    new_point = old_point.copy()
    res = get_sample_full(new_point.size,FPI)
    for idx in idx_choose:
        new_point[idx] = res[idx]

    return new_point

def get_sample_full(n_par,FPI):
    """helper to get a full prior draw sample

    :param n_par:       Number of parameters
    :param FPI:         FastPriorInfo object

    :return new_point:  Array with parameters drawn from their priors
    """
    new_point = get_sample_helper_full(n_par, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                              FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                              FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                              FPI.px_par_ids, FPI.px_mus, FPI.px_errs)
    return new_point

@jitclass([('uniform_par_ids',nb.int64[:]),('uniform_lows',nb.float64[:]),('uniform_highs',nb.float64[:]),\
           ('lin_exp_par_ids',nb.int64[:]),('lin_exp_lows',nb.float64[:]),('lin_exp_highs',nb.float64[:]),\
           ('normal_par_ids',nb.int64[:]),('normal_mus',nb.float64[:]),('normal_sigs',nb.float64[:]),\
           ('dm_par_ids',nb.int64[:]),('dm_dists',nb.float64[:]),('dm_errs',nb.float64[:]),\
           ('px_par_ids',nb.int64[:]),('px_mus',nb.float64[:]),('px_errs',nb.float64[:]),\
           ('cut_par_ids',nb.int64[:]),('cut_lows',nb.float64[:]),('cut_highs',nb.float64[:]),\
           ('cw_ext_par_ids',nb.int64[:]),('cw_ext_lows',nb.float64[:]),('cw_ext_highs',nb.float64[:]),\
           ('global_common',nb.float64)])
class FastPriorInfo:
    """simple jitclass to store the various elements of fast prior calculation in a way that can be accessed quickly from a numba environment

    :param uniform_par_ids: Indices of parameters with uniform prior
    :param uniform_lows:    Lower prior bounds of uniform prior parameters
    :param uniform_highs:   Upper prior bounds of uniform prior parameters
    :param lin_exp_par_ids: Indices of parameters with linear exponential prior
    :param lin_exp_lows:    Lower prior bounds of linear exponential prior parameters
    :param lin_exp_highs:   Upper prior bounds of linear exponential prior parameters
    :param normal_par_ids:  Indices of parameters with normal prior
    :param normal_mus:      Means of normal prior parameters
    :param normal_sigs:     Standard deviations of normal prior parameters
    :param dm_par_ids:      Indices of parameters with DM distance prior
    :param dm_dists:        Mean distances of DM distance prior parameters
    :param dm_errs:         Errors of DM distance prior parameters
    :param px_par_ids:      Indices of parameters with parallax distance prior
    :param px_mus:          Means of parallax distance prior parameters
    :param px_errs:         Errors of parallax distance prior parameters
    :param cut_par_ids:     Indices of parameters where an extra cut might be needed to stay in prior range (like distance)
    :param cut_lows:        Lower bounds of these parameters
    :param cut_highs:       Upper bounds of these parameters
    :param cw_ext_par_ids:  Indices of projection parameters (previously called extrinsic parameters)
    :param cw_ext_lows:     Lower bounds of extrinsic parameters
    :param cw_ext_highs:    Upper bounds of extrinsic parameters
    :param global_common:   Part of the log prior independent of the parameter values
    """
    def __init__(self, uniform_par_ids, uniform_lows, uniform_highs, lin_exp_par_ids, lin_exp_lows, lin_exp_highs, normal_par_ids, normal_mus, normal_sigs, dm_par_ids, dm_dists, dm_errs, px_par_ids, px_mus, px_errs, cut_par_ids, cut_lows, cut_highs, cw_ext_par_ids, cw_ext_lows, cw_ext_highs, global_common):
        self.uniform_par_ids = uniform_par_ids
        self.uniform_lows = uniform_lows
        self.uniform_highs = uniform_highs
        self.lin_exp_par_ids = lin_exp_par_ids
        self.lin_exp_lows = lin_exp_lows
        self.lin_exp_highs = lin_exp_highs
        self.normal_par_ids = normal_par_ids
        self.normal_mus = normal_mus
        self.normal_sigs = normal_sigs
        self.dm_par_ids = dm_par_ids
        self.dm_dists = dm_dists
        self.dm_errs = dm_errs
        self.px_par_ids = px_par_ids
        self.px_mus = px_mus
        self.px_errs = px_errs
        self.cut_par_ids = cut_par_ids
        self.cut_lows = cut_lows
        self.cut_highs = cut_highs
        self.cw_ext_par_ids = cw_ext_par_ids
        self.cw_ext_lows = cw_ext_lows
        self.cw_ext_highs = cw_ext_highs
        self.global_common = global_common

def get_FastPriorInfo(pta,psrs,par_names_cw_ext):
    """get FastPriorInfo object from pta

    :param pta:                 enterprise PTA object
    :param psrs:                List of enterprise pulsar objects
    :param par_names_cw_ext:    List of parameter names which are projection parameters (previously called extrinsic parameters)
    """
    fp_loc = FastPrior(pta,psrs,par_names_cw_ext)
    FPI = FastPriorInfo(fp_loc.uniform_par_ids, fp_loc.uniform_lows, fp_loc.uniform_highs,\
                        fp_loc.lin_exp_par_ids, fp_loc.lin_exp_lows, fp_loc.lin_exp_highs,\
                        fp_loc.normal_par_ids, fp_loc.normal_mus, fp_loc.normal_sigs,\
                        fp_loc.dm_par_ids, fp_loc.dm_dists, fp_loc.dm_errs,\
                        fp_loc.px_par_ids, fp_loc.px_mus, fp_loc.px_errs,\
                        fp_loc.cut_par_ids,fp_loc.cut_lows,fp_loc.cut_highs,\
                        fp_loc.cw_ext_par_ids,fp_loc.cw_ext_lows,fp_loc.cw_ext_highs,\
                        fp_loc.global_common)
    return FPI
