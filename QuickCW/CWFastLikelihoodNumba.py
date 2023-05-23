"""C 2021 Matthew Digman and Bence Becsy
numba version of fast likelihood"""
import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List
import scipy.linalg

from enterprise import constants as const
from QuickCW.lapack_wrappers import solve_triangular

import QuickCW.const_mcmc as cm

class FastLikeMaster:
    """class to store pta things so they do not have to be recomputed when red noise is recomputed"""
    def __init__(self,psrs,pta,params,x0,includeCW=True,prior_recovery=False):
        """
        get Class for generating the fast CW likelihood.
        
        :param pta:             `enterprise` pta object.
        :param params:          Dictionary of noise parameters.
        :param x0:              CWInfo object, which is partially redundant with params but better handled by numba
        :param includeCW:       Switch if we want to include the contribution of the CW signal or not [True]
        :param prior_recovery:  If True, we return constant likelihood to be used for prior recovery diagnostic test [False]
        """
        self.Npsr = x0.Npsr
        self.pta = pta

        #include switch to easily turn off CW for TD Bayes factor calculation
        self.includeCW = includeCW

        #include switch to get constant log_likelihoods for prior recovery tests
        self.prior_recovery = prior_recovery

        #put the positions into an array
        self.pos = np.zeros((self.Npsr,3))
        self.pdist = np.zeros((self.Npsr,2))
        for i,psr in enumerate(psrs):
            self.pos[i] = psr.pos
            self.pdist[i] = psr.pdist

        #get the N vects without putting them in a matrix
        self.Nvecs = List(self.pta.get_ndiag(params))


        #get the part of the determinant that can be computed right now
        self.logdet = 0.0
        for (l,m) in self.pta.get_rNr_logdet(params):
            self.logdet += m
        #self.logdet += np.sum([m for (l,m) in self.pta.get_rNr_logdet(self.params)])

        #get the other pta results
        self.TNTs = self.pta.get_TNT(params)
        Ts = self.pta.get_basis()

        #invchol_Sigma_Ts = List()
        self.Nrs = List()
        self.isqrNvecs = List()
        #unify types outside numba to avoid slowing down compilation
        #also add more components to logdet

        #put toas and residuals into a numba typed List of arrays, which shouldn't require any actual copies
        self.toas = List([psr.toas for psr in psrs])
        self.residuals = List([psr.residuals for psr in psrs])

        self.resres_rNr = 0.
        self.TNvs = List()
        self.dotTNrs = List()
        for i in range(self.Npsr):
            self.isqrNvecs.append(1/np.sqrt(self.Nvecs[i]))
            self.Nrs.append(self.residuals[i]/np.sqrt(self.Nvecs[i]))
            self.resres_rNr += np.dot(self.Nrs[i], self.Nrs[i])
            self.TNvs.append((Ts[i].T/np.sqrt(self.Nvecs[i])).copy().T) #store F contiguous version
            self.dotTNrs.append(np.dot(self.Nrs[i],self.TNvs[i]))

        #put the rnr part of resres onto logdet
        self.logdet = self.resres_rNr+self.logdet

        self.max_toa = np.max(self.toas[0])
        for i in range(self.Npsr):
            #find the latest arriving signal to prohibit signals that have already merged
            self.max_toa = max(self.max_toa,np.max(self.toas[i]))

    def get_new_FastLike(self,x0,params):
        chol_Sigmas = List()
        phiinvs = List()
        for i in range(self.Npsr):
            chol_Sigmas.append(np.identity((self.TNvs[i].shape[1])).T) #temporary but can't be 0 or else the initialization of FLI will crash
            phiinvs.append(np.ones(self.TNvs[i].shape[1]))

        FLI = FastLikeInfo(self.logdet,self.pos,self.pdist,self.toas,self.Nvecs,self.Nrs,self.max_toa,x0,
                           self.Npsr,self.isqrNvecs,self.TNvs,self.dotTNrs,chol_Sigmas,phiinvs,self.includeCW,self.prior_recovery)
        return self.recompute_FastLike(FLI,x0,params)
    #@profile
    def recompute_FastLike(self,FLI,x0,params, chol_update=False,mask=None):
        if mask is None:
            #mask to skip updating values if set to True
            mask = np.zeros(self.Npsr,dtype=np.bool_)
        if not FLI.prior_recovery:
            pls_temp = self.pta.get_phiinv(params, logdet=True, method='partition')

            if chol_update: #update Cholesky of Sigma instead of recompute
                #chol_Sigmas, logdet_array, new_phiinvs = cholupdate_loop(FLI.chol_Sigmas, List(pls_temp), FLI.phiinvs, self.Npsr)
                #
                #FLI.chol_Sigmas = chol_Sigmas
                #FLI.logdet_array = logdet_array
                #FLI.phiinvs = new_phiinvs

                for i in range(self.Npsr):
                    if mask[i]:
                        continue

                    phiinv_loc,logdetphi_loc = pls_temp[i]
                    chol_Sigma = cholupdate(FLI.chol_Sigmas[i], phiinv_loc-FLI.phiinvs[i])

                    logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)

                    FLI.chol_Sigmas[i][:] = chol_Sigma

                    FLI.logdet_array[i] = logdetphi_loc+logdet_Sigma_loc

                    FLI.phiinvs[i][:] = phiinv_loc

            else:
                for i in range(self.Npsr):
                    if mask[i]:
                        continue

                    phiinv_loc,logdetphi_loc = pls_temp[i]

                    FLI.phiinvs[i][:] = phiinv_loc
                    if phiinv_loc.ndim == 1:
                        #Sigma_alt = self.TNTs[i]+np.diag(phiinv_loc)
                        #overwrite old chol_Sigma so can be done without allocating new array
                        Sigma = create_Sigma(phiinv_loc,self.TNTs[i],FLI.chol_Sigmas[i].T)
                        #assert np.allclose(Sigma,Sigma_alt)
                    else:
                        Sigma = self.TNTs[i]+phiinv_loc

                    #mutate inplace to avoid memory allocation overheads
                    chol_Sigma,lower = scipy.linalg.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)

                    logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)#2 * np.sum(np.log(np.diag(chol_Sigma)))

                    #this should be mutated in place but assign it anyway to be safe
                    FLI.chol_Sigmas[i][:] = chol_Sigma

                    #add the necessary component to logdet
                    FLI.logdet_array[i] = logdetphi_loc+logdet_Sigma_loc


            #set logdet
            FLI.set_resres_logdet(FLI.resres_array,FLI.logdet_array,FLI.logdet_base)

        #list of pulsars updated
        psr_idxs = np.where(~mask)[0]
        
        FLI.update_red_noise(x0, psr_idxs)

        return FLI#FastLikeInfo(resres,logdet,self.pos,self.pdist,self.toas,invchol_Sigma_TNs,self.Nvecs,self.Nrs,self.max_toa,x0,self.Npsr,self.isqrNvecs,self.residuals)

@njit(fastmath=True,parallel=False)
def cholupdate_loop(chol_Sigmas, pls_temp, old_phiinvs, Npsr):
    """Jitted loop over Sigma matrices to update their Cholesky.
    Currently not faster than recalculating from scratch, so not used.

    :param chol_Sigmas:     List of Cholesky decompositions of Sigma matrices
    :param pls_temp:        New list of tuples returned by pta.get_phiinv
    :param old_phiinvs:     Old list of phiinv matrices
    :param Npsr:            Number of pulsars (also number of Sigma matrices)

    :return chol_Sigmas:    List of updated Choleskies
    :return logdet_array:   New array containing logdet values
    :return new_phiinvs:    List of new phiinv matrices
    """
    logdet_array = np.zeros(Npsr)
    new_phiinvs = List()
    for i in range(Npsr):
        phiinv_loc,logdetphi_loc = pls_temp[i]
        chol_Sigmas[i] = cholupdate(chol_Sigmas[i], phiinv_loc-old_phiinvs[i])
        new_phiinvs.append(phiinv_loc)
        logdet_array[i] = logdet_Sigma_helper(chol_Sigmas[i]) + logdetphi_loc

    return chol_Sigmas, logdet_array, new_phiinvs

@njit(fastmath=True,parallel=False)
def cholupdate(L_in,diag_diff):
    """Jitted routine to update the Cholesky of a matrix instead of recomputing it.
    In principle could be faster, but right now it is not, so it's not used.

    :param L_in:        Previous Cholesky that we want to update
    :param diag_diff:   Differences in the diagonal of the original matrix

    :return L:          Updated Cholesky    
    """
    n = L_in.shape[0]
    #L = L_in
    L = np.copy(L_in)
    #for idx, diff in enumerate(diag_diff):
    for idx in range(n):
        diff = diag_diff[idx]
        if diff!=0.0:
            x = np.zeros(n)
            if diff>0:
                x[idx] = np.sqrt(diff)
                sign = 1.0
            elif diff<0:
                x[idx] = np.sqrt(-diff)
                sign = -1.0
            #print(x)
            for k in range(idx,n):
                #print(L[k,k]**2)
                #print(x[k]**2)
                #print(L[k,k]**2 + sign * x[k]**2)
                r = np.sqrt(L[k,k]**2 + sign * x[k]**2)
                c = r / L[k,k]
                s = x[k] / L[k,k]
                L[k,k] = r
                #L[k+1:n,k] = (L[k+1:n,k] + sign * s*x[k+1:n]) / c
                #x[k+1:n] = c * x[k+1:n] - s * L[k+1:n,k]
                #for j in prange(k+1,n): #1 ms (~30% of runtime) this loop
                for j in range(k+1,n): #1 ms (~30% of runtime) this loop
                    L[j,k] = (L[j,k] + sign* s*x[j]) / c
                    x[j] = c * x[j] - s * L[j,k]

    return L


@njit(parallel=True,fastmath=True)
def logdet_Sigma_helper(chol_Sigma):
    """get logdet sigma from cholesky of Sigma

    :param chol_Sigma:  Cholesky decomposition of Sigma matrix

    :return 2*res:      Contribution to logdet from this Sigma matrix
    """
    res = 0.
    for itrj in prange(0,chol_Sigma.shape[0]):
        res += np.log(chol_Sigma[itrj,itrj])
    return 2*res


@njit(parallel=True,fastmath=True)
def create_Sigma(phiinv_loc,TNT,Sigma):
    """create just the upper triangle of the Sigma matrix with phiinv_loc added to the diagonal, lower triangle will be garbage

    :param phiinv_loc:  phiinv matrix
    :param TNT:         Precomputed matrix product of TNT
    :param Sigma:       Initialized mtrix with the right shape to hold Sigma (allows for just overwriting instead of creating new array each time we update this)

    :return Sigma:      Upper trinagle Sigma matrix - lower triangle is junk
    """
    #Sigma = np.zeros((phiinv_loc.size,phiinv_loc.size))
    for itrj1 in prange(0,phiinv_loc.size):
        Sigma[itrj1,itrj1] = TNT[itrj1,itrj1]+phiinv_loc[itrj1]
        for itrj2 in prange(itrj1+1,phiinv_loc.size):
            Sigma[itrj1,itrj2] = TNT[itrj1,itrj2]
    #for itrj1 in prange(0,phiinv_loc.size):
    return Sigma

@jitclass([('Npsr',nb.int64),\
           ('cw_p_dists',nb.float64[:]),('cw_p_phases',nb.float64[:]),('rn_gammas',nb.float64[:]),('rn_log10_As',nb.float64[:]),\
           ('cos_gwtheta',nb.float64),('cos_inc',nb.float64),('gwphi',nb.float64),('log10_fgw',nb.float64),('log10_h',nb.float64),\
           ('log10_mc',nb.float64),('phase0',nb.float64),('psi',nb.float64),('gwb_gamma',nb.float64),('gwb_log10_A',nb.float64),\
           ('idx_dists',nb.int64[:]),('idx_phases',nb.int64[:]),('idx_rn_gammas',nb.int64[:]),('idx_rn_log10_As',nb.int64[:]),\
           ('idx_cos_gwtheta',nb.int64),('idx_cos_inc',nb.int64),('idx_gwphi',nb.int64),('idx_log10_fgw',nb.int64),('idx_log10_h',nb.int64),\
           ('idx_log10_mc',nb.int64),('idx_phase0',nb.int64),('idx_psi',nb.int64),('idx_gwb_gamma',nb.int64),('idx_gwb_log10_A',nb.int64),\
           ('idx_rn',nb.int64[:]),('idx_gwb',nb.int64[:]),('idx_int',nb.int64[:]),('idx_cw_ext',nb.int64[:]),('idx_cw_int',nb.int64[:])])
class CWInfo:
    """simple jitclass to store the various parmeters in a way that can be accessed quickly from a numba environment

    :param Npsr:                Number of pulsars
    :param params_in:           Array of all parameters in the same order as in par_names
    :param par_names:           List of parameter names - must follow certain naming conventions so we can identify parameters
    :param par_names_cw_ext:    Subset of all parameters that describe the CW signal and are projection parameters (previously called extrinsic parameters)
    :param par_names_cw_int:    Subset of all parameters that describe the CW signal and are shape parameters (previously called intrinsic parameters)
    """
    
    def __init__(self,Npsr,params_in,par_names,par_names_cw_ext,par_names_cw_int):
        self.Npsr = Npsr
        self.idx_phases = np.array([par_names.index(par) for par in par_names if "_cw0_p_phase" in par])
        self.idx_dists = np.array([par_names.index(par) for par in par_names if "_cw0_p_dist" in par])

        self.idx_rn_gammas = np.array([par_names.index(par) for par in par_names if "_red_noise_gamma" in par])
        self.idx_rn_log10_As = np.array([par_names.index(par) for par in par_names if "_red_noise_log10_A" in par])
        self.idx_rn = np.concatenate((self.idx_rn_gammas,self.idx_rn_log10_As))

        self.idx_gwb_gamma = par_names.index("gwb_gamma")
        self.idx_gwb_log10_A = par_names.index("gwb_log10_A")
        self.idx_gwb = np.array([self.idx_gwb_gamma, self.idx_gwb_log10_A])

        self.idx_cos_inc = par_names.index("0_cos_inc")
        self.idx_log10_h = par_names.index("0_log10_h")
        self.idx_phase0 = par_names.index("0_phase0")
        self.idx_psi = par_names.index("0_psi")

        self.idx_cos_gwtheta = par_names.index("0_cos_gwtheta")
        self.idx_gwphi = par_names.index("0_gwphi")
        self.idx_log10_fgw = par_names.index("0_log10_fgw")
        self.idx_log10_mc = par_names.index("0_log10_mc")

        self.idx_cw_ext = np.zeros(len(par_names_cw_ext),dtype=np.int64)
        for i,name_ext in enumerate(par_names_cw_ext):
            self.idx_cw_ext[i] = par_names.index(name_ext)

        self.idx_cw_int = np.zeros(len(par_names_cw_int),dtype=np.int64)
        for i,name_int in enumerate(par_names_cw_int):
            self.idx_cw_int[i] = par_names.index(name_int)

        self.idx_int = np.concatenate((self.idx_rn, self.idx_gwb, self.idx_cw_int))


        self.update_params(params_in)

    def update_params(self,params_in):
        self.cw_p_phases = params_in[self.idx_phases]
        self.cw_p_dists = params_in[self.idx_dists]
        self.cos_inc = params_in[self.idx_cos_inc]
        self.log10_h = params_in[self.idx_log10_h]
        self.phase0 = params_in[self.idx_phase0]
        self.psi = params_in[self.idx_psi]
        self.cos_gwtheta = params_in[self.idx_cos_gwtheta]
        self.gwphi = params_in[self.idx_gwphi]
        self.log10_fgw = params_in[self.idx_log10_fgw]
        self.log10_mc = params_in[self.idx_log10_mc]
        self.rn_gammas = params_in[self.idx_rn_gammas]
        self.rn_log10_As = params_in[self.idx_rn_log10_As]
        self.gwb_gamma = params_in[self.idx_gwb_gamma]
        self.gwb_log10_A = params_in[self.idx_gwb_log10_A]

    def validate_consistent(self,params_in):
        """check current params match input params"""
        assert np.all(isclose(self.cw_p_phases,params_in[self.idx_phases]))
        assert np.all(isclose(self.cw_p_dists, params_in[self.idx_dists]))
        assert isclose(self.cos_inc, params_in[self.idx_cos_inc])
        assert isclose(self.log10_h, params_in[self.idx_log10_h])
        assert isclose(self.phase0, params_in[self.idx_phase0])
        assert isclose(self.psi, params_in[self.idx_psi])
        assert isclose(self.cos_gwtheta, params_in[self.idx_cos_gwtheta])
        assert isclose(self.gwphi, params_in[self.idx_gwphi])
        assert isclose(self.log10_fgw, params_in[self.idx_log10_fgw])
        assert isclose(self.log10_mc, params_in[self.idx_log10_mc])
        assert np.all(isclose(self.rn_gammas, params_in[self.idx_rn_gammas]))
        assert np.all(isclose(self.rn_log10_As, params_in[self.idx_rn_log10_As]))
        assert isclose(self.gwb_gamma, params_in[self.idx_gwb_gamma])
        assert isclose(self.gwb_log10_A, params_in[self.idx_gwb_log10_A])
        return True

@njit()
def get_lnlikelihood_helper(x0,resres,logdet,pos,pdist,NN,MMs,includeCW=True,prior_recovery=False):
    """jittable helper for calculating the log likelihood in CWFastLikelihood
    
    :param x0:              CWInfo object
    :param resres:          Array holding the (residual|residual) inner product
    :param logdet:          Log determinant piece of the likelihood
    :param pos:             (number of pulsars, 3) array holding 3d unit vector pointing towards each pulsar given by psr.pos
    :param pdist:           (number of pulsars, 2) array holding distance and error of distance for each pulsar in kpc given by psr.pdist
    :param NN:              N matrices holding (filter|residual) type inner products
    :param MMs:             M matrices holding (filter|filter) type inner products
    :param includeCW:       Switch if we want to include the contribution of the CW signal or not [True]
    :param prior_recovery:  If True, we return constant likelihood to be used for prior recovery diagnostic test [False]

    :return log_L:          Log likelihood value
    """
    if prior_recovery:
        return 0.0
    else:
        fgw = 10.**x0.log10_fgw
        amp = 10.**x0.log10_h / (2*np.pi*fgw)
        mc = 10.**x0.log10_mc * const.Tsun


        sin_gwtheta = np.sqrt(1-x0.cos_gwtheta**2)
        sin_gwphi = np.sin(x0.gwphi)
        cos_gwphi = np.cos(x0.gwphi)

        m = np.array([sin_gwphi, -cos_gwphi, 0.0])
        n = np.array([-x0.cos_gwtheta * cos_gwphi, -x0.cos_gwtheta * sin_gwphi, sin_gwtheta])
        omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -x0.cos_gwtheta])
        sigma = np.zeros(4)

        cos_phase0 = np.cos(x0.phase0)
        sin_phase0 = np.sin(x0.phase0)
        sin_2psi = np.sin(2*x0.psi)
        cos_2psi = np.cos(2*x0.psi)

        log_L = -0.5*resres -0.5*logdet

        if includeCW:
            for i in prange(0,x0.Npsr):
                m_pos = 0.
                n_pos = 0.
                cosMu = 0.
                for j in range(0,3):
                    m_pos += m[j]*pos[i,j]
                    n_pos += n[j]*pos[i,j]
                    cosMu -= omhat[j]*pos[i,j]
                #m_pos = np.dot(m, pos[i])
                #n_pos = np.dot(n, pos[i])
                #cosMu = -np.dot(omhat, pos[i])

                F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
                F_c = (m_pos * n_pos) / (1 - cosMu)

                p_dist = (pdist[i,0] + pdist[i,1]*x0.cw_p_dists[i])*(const.kpc/const.c)

                w0 = np.pi * fgw
                omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

                amp_psr = amp * (w0/omega_p0)**(1.0/3.0)
                phase0_psr = x0.cw_p_phases[i]

                cos_phase0_psr = np.cos(x0.phase0+phase0_psr*2.0)
                sin_phase0_psr = np.sin(x0.phase0+phase0_psr*2.0)

                sigma[0] =  amp*(   cos_phase0 * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term sine
                                  2*sin_phase0 *     x0.cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
                sigma[1] =  amp*(   sin_phase0 * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term cosine
                                  2*cos_phase0 *     x0.cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )
                sigma[2] =  -amp_psr*(   cos_phase0_psr * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term sine
                                  2*sin_phase0_psr *     x0.cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
                sigma[3] =  -amp_psr*(   sin_phase0_psr * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term cosine
                                  2*cos_phase0_psr *     x0.cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )

                for j in range(0,4):
                    log_L += sigma[j]*NN[i,j]

                prodMMPart = 0.
                for j in range(0,4):
                    for k in range(0,4):
                        prodMMPart += sigma[j]*MMs[i,j,k]*sigma[k]

                log_L -= prodMMPart/2#np.dot(sigma,np.dot(MMs[i],sigma))/2

        return log_L

@njit(fastmath=True,parallel=True)
def update_intrinsic_params2(x0,isqrNvecs,Nrs,pos,pdist,toas,NN,MMs,TNvs,chol_Sigmas,idxs,resres_array,dotTNrs):
    '''Calculate inner products N=(res|S), M=(S|S)

    :param x0:              CWInfo object
    :param isqrNvecs:       Inverse squareroot of N vectors
    :param Nrs:             Residuals times inverse sqareroot N vectors
    :param pos:             (number of pulsars, 3) array holding 3d unit vector pointing towards each pulsar given by psr.pos
    :param pdist:           (number of pulsars, 2) array holding distance and error of distance for each pulsar in kpc given by psr.pdist
    :param toas:            List of arrays of TOAs for each pulsar
    :param NN:              N matrices holding (filter|residual) type inner products
    :param MMs:             M matrices holding (filter|filter) type inner products
    :param TNvs:            T vectros times inverse squareroot N vectors
    :param chol_Sigmas:     List of Cholesky decompositions of Sigma matrices
    :param idxs:            Indices of pulsar for which we want to update things
    :param resres_array:    Array containing contributions to (res|res)
    :param dotTNrs:         Precalculated dot product of Nrs and TNvs
    '''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    sin_gwphi = np.sin(x0.gwphi)
    cos_gwphi = np.cos(x0.gwphi)

    omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])



    for ii in idxs:
        MM = np.zeros((4, 4))

        cosMu = -np.dot(omhat, pos[ii])

        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*(const.kpc/const.c)

        omega_p013 = np.sqrt(np.sqrt(np.sqrt((1. + 256./5. * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu)))))

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        #invCholSigmaTN = invchol_Sigma_TNs[ii]
        #SigmaTNrProd = SigmaTNrProds[ii]

        #divide the signals by N
        Nr = Nrs[ii]#residuals[ii]/Nvecs[ii]
        isqrNvec = isqrNvecs[ii]
        toas_in = toas[ii]
        TNv = TNvs[ii]
        chol_Sigma = chol_Sigmas[ii]
        dotTNr = dotTNrs[ii]
        #resres_array[ii] = 0.
        n1,n2 = TNv.shape

        esNr = 0.
        ecNr = 0.
        psNr = 0.
        pcNr = 0.

        esNes = 0.
        ecNec = 0.
        psNps = 0.
        pcNpc = 0.

        ecNes = 0.
        psNes = 0.
        pcNes = 0.
        psNec = 0.
        pcNec = 0.
        pcNps = 0.

        #get the sin and cosine parts
        ET_sin  = np.zeros(n1)
        ET_cos  = np.zeros(n1)

        PT_sin  = np.zeros(n1)
        PT_cos  = np.zeros(n1)

        #break out this loop instead of using numpy syntax so we don't have to store omegas and phases ever
        for itrk in prange(n1):
            #set up filters
            toas_loc = toas_in[itrk] - cm.tref
            #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
            #also replace omega with 1/omega**(1/3), which is the quantity we actually need
            #if sqrt is a native cpu function 3 sqrts will probably be faster than taking the eigth root
            omega13 = np.sqrt(np.sqrt(np.sqrt((1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc))))
            phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

            tp = toas_loc - p_dist*(1-cosMu)
            omega_p13 = np.sqrt(np.sqrt(np.sqrt((1./omega_p013**8 - 256./5. * mc**(5/3) * w0**(8/3) / omega_p013**8 * tp))))

            phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * omega_p013**5 * (1. - omega_p13**5)

            PT_amp = isqrNvec[itrk] * omega_p13
            ET_amp = isqrNvec[itrk] * omega13

            PT_sin[itrk] = PT_amp * np.sin(2*phase_p)
            PT_cos[itrk] = PT_amp * np.cos(2*phase_p)


            ET_sin[itrk] = ET_amp * np.sin(2*phase)
            ET_cos[itrk] = ET_amp * np.cos(2*phase)

            #get the results


            psNps += PT_sin[itrk]*PT_sin[itrk]
            pcNpc += PT_cos[itrk]*PT_cos[itrk]

            psNes += PT_sin[itrk]*ET_sin[itrk]
            pcNes += PT_cos[itrk]*ET_sin[itrk]
            psNec += PT_sin[itrk]*ET_cos[itrk]
            pcNec += PT_cos[itrk]*ET_cos[itrk]
            pcNps += PT_cos[itrk]*PT_sin[itrk]

            #these segments aren't the time limiting factor so we don't see any real speedup from skipping them
            #if not dist_only:
            #Nes = iNvec[itrk]*ET_sin[itrk]
            #Nec = iNvec[itrk]*ET_cos[itrk]
            esNes += ET_sin[itrk]*ET_sin[itrk]
            ecNec += ET_cos[itrk]*ET_cos[itrk]
            ecNes += ET_cos[itrk]*ET_sin[itrk]

            psNr += Nr[itrk]*PT_sin[itrk]
            pcNr += Nr[itrk]*PT_cos[itrk]
            esNr += Nr[itrk]*ET_sin[itrk]
            ecNr += Nr[itrk]*ET_cos[itrk]

        dotSigmaTNrr  = 0.
        dotSigmaTNesr = 0.
        dotSigmaTNecr = 0.
        dotSigmaTNpsr = 0.
        dotSigmaTNpcr = 0.

        dotSigmaTNes = 0.
        dotSigmaTNec = 0.
        dotSigmaTNps = 0.
        dotSigmaTNpc = 0.

        dotSigmaTNeces = 0.
        dotSigmaTNpses = 0.
        dotSigmaTNpces = 0.
        dotSigmaTNpsec = 0.
        dotSigmaTNpcec = 0.
        dotSigmaTNpcps = 0.

        dotTNes = np.zeros(n2)
        dotTNec = np.zeros(n2)
        dotTNps = np.zeros(n2)
        dotTNpc = np.zeros(n2)
        #dotTNr  = np.zeros(n2)

        for itrj in prange(n2):
            for itrk in prange(n1):
                dotTNes[itrj] += TNv[itrk,itrj]*ET_sin[itrk]
                dotTNec[itrj] += TNv[itrk,itrj]*ET_cos[itrk]
                dotTNps[itrj] += TNv[itrk,itrj]*PT_sin[itrk]
                dotTNpc[itrj] += TNv[itrk,itrj]*PT_cos[itrk]
        #        dotTNr[itrj]  += TNv[itrk,itrj]*Nr[itrk]

        #combine into a matrix to allow solve_triangular to work better
        dotTN5 = np.zeros((5,n2)).T
        dotTN5[:,0] = dotTNes
        dotTN5[:,1] = dotTNec
        dotTN5[:,2] = dotTNps
        dotTN5[:,3] = dotTNpc
        #dotTN5[:,4] = dotTNr
        dotTN5[:,4] = dotTNr

        SigmaTN5Prod = solve_triangular(chol_Sigma,dotTN5,lower_a=True,trans_a=False,overwrite_b=True)

        #SigmaTNesProd = solve_triangular(chol_Sigma,dotTNes,lower_a=True,trans_a=False,overwrite_b=True)
        #SigmaTNecProd = solve_triangular(chol_Sigma,dotTNec,lower_a=True,trans_a=False,overwrite_b=True)
        #SigmaTNpsProd = solve_triangular(chol_Sigma,dotTNps,lower_a=True,trans_a=False,overwrite_b=True)
        #SigmaTNpcProd = solve_triangular(chol_Sigma,dotTNpc,lower_a=True,trans_a=False,overwrite_b=True)
        #SigmaTNrProd  = solve_triangular(chol_Sigma,dotTNr ,lower_a=True,trans_a=False,overwrite_b=True)

        for itrj1 in prange(n2):
            SigmaTNesProd = SigmaTN5Prod[itrj1,0]
            SigmaTNecProd = SigmaTN5Prod[itrj1,1]
            SigmaTNpsProd = SigmaTN5Prod[itrj1,2]
            SigmaTNpcProd = SigmaTN5Prod[itrj1,3]
            SigmaTNrProd  = SigmaTN5Prod[itrj1,4]

            dotSigmaTNesr += SigmaTNesProd*SigmaTNrProd
            dotSigmaTNecr += SigmaTNecProd*SigmaTNrProd
            dotSigmaTNpsr += SigmaTNpsProd*SigmaTNrProd
            dotSigmaTNpcr += SigmaTNpcProd*SigmaTNrProd
            dotSigmaTNrr  += SigmaTNrProd*SigmaTNrProd

            dotSigmaTNes += SigmaTNesProd*SigmaTNesProd
            dotSigmaTNec += SigmaTNecProd*SigmaTNecProd
            dotSigmaTNps += SigmaTNpsProd*SigmaTNpsProd
            dotSigmaTNpc += SigmaTNpcProd*SigmaTNpcProd

            dotSigmaTNeces += SigmaTNecProd*SigmaTNesProd
            dotSigmaTNpses += SigmaTNpsProd*SigmaTNesProd
            dotSigmaTNpces += SigmaTNpcProd*SigmaTNesProd
            dotSigmaTNpsec += SigmaTNpsProd*SigmaTNecProd
            dotSigmaTNpcec += SigmaTNpcProd*SigmaTNecProd
            dotSigmaTNpcps += SigmaTNpcProd*SigmaTNpsProd

        #get resres
        resres_array[ii] = -dotSigmaTNrr

        #get NN
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]

        #if dist_only:
        #    MM[0:2,0:2] = MMs[ii,0:2,0:2]
        #else:
        NN[ii,0] = esNr - dotSigmaTNesr
        NN[ii,1] = ecNr - dotSigmaTNecr

        MM[0,0] = esNes - dotSigmaTNes
        MM[1,1] = ecNec - dotSigmaTNec
        MM[1,0] = ecNes - dotSigmaTNeces
        MM[0,1] = MM[1,0]

        MMs[ii,:,:] = MM


@njit(fastmath=True,parallel=True)
def update_intrinsic_params(x0,isqrNvecs,Nrs,pos,pdist,toas,NN,MMs,SigmaTNrProds,invchol_Sigma_TNs,idxs,dist_only=True):
    '''Calculate inner products N=(res|S), M=(S|S)

    :param x0:                  CWInfo object
    :param isqrNvecs:           Inverse squareroot of N vectors
    :param Nrs:                 Residuals times inverse sqareroot N vectors
    :param pos:                 (number of pulsars, 3) array holding 3d unit vector pointing towards each pulsar given by psr.pos
    :param pdist:               (number of pulsars, 2) array holding distance and error of distance for each pulsar in kpc given by psr.pdist
    :param toas:                List of arrays of TOAs for each pulsar
    :param NN:                  N matrices holding (filter|residual) type inner products
    :param MMs:                 M matrices holding (filter|filter) type inner products
    :param SigmaTNrProds:       ---
    :param invchol_Sigma_TNs:   ---
    :param idxs:                Indices of pulsar for which we want to update things
    :param dist_only:           Option to skip parts of calculation whn only updating pulsar distances - not currently used
    '''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    sin_gwphi = np.sin(x0.gwphi)
    cos_gwphi = np.cos(x0.gwphi)

    omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

    MM = np.zeros((4, 4))

    for ii in idxs:
        cosMu = -np.dot(omhat, pos[ii])

        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*(const.kpc/const.c)

        omega_p013 = np.sqrt(np.sqrt(np.sqrt((1. + 256./5. * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu)))))

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        invCholSigmaTN = invchol_Sigma_TNs[ii]
        SigmaTNrProd = SigmaTNrProds[ii]

        #divide the signals by N
        Nr = Nrs[ii]#residuals[ii]/Nvecs[ii]
        isqrNvec = isqrNvecs[ii]
        toas_in = toas[ii]

        esNr = 0.
        ecNr = 0.
        psNr = 0.
        pcNr = 0.

        esNes = 0.
        ecNec = 0.
        psNps = 0.
        pcNpc = 0.

        ecNes = 0.
        psNes = 0.
        pcNes = 0.
        psNec = 0.
        pcNec = 0.
        pcNps = 0.

        #get the sin and cosine parts
        ET_sin  = np.zeros(invCholSigmaTN.shape[1])
        ET_cos  = np.zeros(invCholSigmaTN.shape[1])

        PT_sin  = np.zeros(invCholSigmaTN.shape[1])
        PT_cos  = np.zeros(invCholSigmaTN.shape[1])

        #break out this loop instead of using numpy syntax so we don't have to store omegas and phases ever
        for itrk in prange(invCholSigmaTN.shape[1]):
            #set up filters
            toas_loc = toas_in[itrk] - cm.tref
            #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
            #also replace omega with 1/omega**(1/3), which is the quantity we actually need
            #if sqrt is a native cpu function 3 sqrts will probably be faster than taking the eigth root
            omega13 = np.sqrt(np.sqrt(np.sqrt((1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc))))
            phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

            tp = toas_loc - p_dist*(1-cosMu)
            omega_p13 = np.sqrt(np.sqrt(np.sqrt((1./omega_p013**8 - 256./5. * mc**(5/3) * w0**(8/3) / omega_p013**8 * tp))))

            phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * omega_p013**5 * (1. - omega_p13**5)

            PT_amp = isqrNvec[itrk] * omega_p13
            ET_amp = isqrNvec[itrk] * omega13

            PT_sin[itrk] = PT_amp * np.sin(2*phase_p)
            PT_cos[itrk] = PT_amp * np.cos(2*phase_p)


            ET_sin[itrk] = ET_amp * np.sin(2*phase)
            ET_cos[itrk] = ET_amp * np.cos(2*phase)

            #get the results

            psNr += Nr[itrk]*PT_sin[itrk]
            pcNr += Nr[itrk]*PT_cos[itrk]

            psNps += PT_sin[itrk]*PT_sin[itrk]
            pcNpc += PT_cos[itrk]*PT_cos[itrk]

            psNes += PT_sin[itrk]*ET_sin[itrk]
            pcNes += PT_cos[itrk]*ET_sin[itrk]
            psNec += PT_sin[itrk]*ET_cos[itrk]
            pcNec += PT_cos[itrk]*ET_cos[itrk]
            pcNps += PT_cos[itrk]*PT_sin[itrk]

            #these segments aren't the time limiting factor so we don't see any real speedup from skipping them
            #if not dist_only:
            #Nes = iNvec[itrk]*ET_sin[itrk]
            #Nec = iNvec[itrk]*ET_cos[itrk]
            esNr += Nr[itrk]*ET_sin[itrk]
            ecNr += Nr[itrk]*ET_cos[itrk]
            esNes += ET_sin[itrk]*ET_sin[itrk]
            ecNec += ET_cos[itrk]*ET_cos[itrk]
            ecNes += ET_cos[itrk]*ET_sin[itrk]

        dotSigmaTNesr = 0.
        dotSigmaTNecr = 0.
        dotSigmaTNpsr = 0.
        dotSigmaTNpcr = 0.

        dotSigmaTNes = 0.
        dotSigmaTNec = 0.
        dotSigmaTNps = 0.
        dotSigmaTNpc = 0.

        dotSigmaTNeces = 0.
        dotSigmaTNpses = 0.
        dotSigmaTNpces = 0.
        dotSigmaTNpsec = 0.
        dotSigmaTNpcec = 0.
        dotSigmaTNpcps = 0.

        #although we could do this in the upper loop without saving ET_sin/ET_cos arrays etc,
        #in my testing it was faster to do it in the transposed order as long as invCholSigmaTN is C contiguous
        for itrj in prange(invCholSigmaTN.shape[0]):
            SigmaTNesProd = 0.
            SigmaTNecProd = 0.
            SigmaTNpsProd = 0.
            SigmaTNpcProd = 0.
            #this loop is currently the limiting factor
            for itrk in prange(invCholSigmaTN.shape[1]):
                SigmaTNpsProd += invCholSigmaTN[itrj,itrk]*PT_sin[itrk]
                SigmaTNpcProd += invCholSigmaTN[itrj,itrk]*PT_cos[itrk]
                SigmaTNesProd += invCholSigmaTN[itrj,itrk]*ET_sin[itrk]
                SigmaTNecProd += invCholSigmaTN[itrj,itrk]*ET_cos[itrk]

            dotSigmaTNpsr += SigmaTNpsProd*SigmaTNrProd[itrj]
            dotSigmaTNpcr += SigmaTNpcProd*SigmaTNrProd[itrj]

            dotSigmaTNps += SigmaTNpsProd*SigmaTNpsProd
            dotSigmaTNpc += SigmaTNpcProd*SigmaTNpcProd

            dotSigmaTNpses += SigmaTNpsProd*SigmaTNesProd
            dotSigmaTNpces += SigmaTNpcProd*SigmaTNesProd
            dotSigmaTNpsec += SigmaTNpsProd*SigmaTNecProd
            dotSigmaTNpcec += SigmaTNpcProd*SigmaTNecProd
            dotSigmaTNpcps += SigmaTNpcProd*SigmaTNpsProd

            #if not dist_only:
            dotSigmaTNesr += SigmaTNesProd*SigmaTNrProd[itrj]
            dotSigmaTNecr += SigmaTNecProd*SigmaTNrProd[itrj]
            dotSigmaTNes += SigmaTNesProd*SigmaTNesProd
            dotSigmaTNec += SigmaTNecProd*SigmaTNecProd
            dotSigmaTNeces += SigmaTNecProd*SigmaTNesProd

        #get NN
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]

        #if dist_only:
        #    MM[0:2,0:2] = MMs[ii,0:2,0:2]
        #else:
        NN[ii,0] = esNr - dotSigmaTNesr
        NN[ii,1] = ecNr - dotSigmaTNecr

        MM[0,0] = esNes - dotSigmaTNes
        MM[1,1] = ecNec - dotSigmaTNec
        MM[1,0] = ecNes - dotSigmaTNeces
        MM[0,1] = MM[1,0]

        MMs[ii,:,:] = MM

@jitclass([('resres',nb.float64),('logdet',nb.float64),('resres_array',nb.float64[:]),('logdet_array',nb.float64[:]),('logdet_base',nb.float64),('logdet_base_orig',nb.float64),\
           ('pos',nb.float64[:,::1]),('pdist',nb.float64[:,::1]),('toas',nb.types.ListType(nb.types.float64[::1])),('Npsr',nb.int64),('max_toa',nb.float64),\
           ('phiinvs',nb.types.ListType(nb.types.float64[::1])),('dotTNrs',nb.types.ListType(nb.types.float64[::1])),('Nvecs',nb.types.ListType(nb.types.float64[::1])),\
           ('isqrNvecs',nb.types.ListType(nb.types.float64[::1])),('Nrs',nb.types.ListType(nb.types.float64[::1])),\
           ('TNvs',nb.types.ListType(nb.types.float64[::1,:])),('chol_Sigmas',nb.types.ListType(nb.types.float64[::1,:])),\
           ('MMs',nb.float64[:,:,::1]),('NN',nb.float64[:,::1]),\
           ('cos_gwtheta',nb.float64),('gwphi',nb.float64),('log10_fgw',nb.float64),('log10_mc',nb.float64),('cw_p_dists',nb.float64[:]),\
           ('gwb_gamma',nb.float64),('gwb_log10_A',nb.float64),('rn_gammas',nb.float64[:]),('rn_log10_As',nb.float64[:]),
           ('includeCW',nb.boolean),('prior_recovery',nb.boolean)])
class FastLikeInfo:
    """simple jitclass to store the various elements of fast likelihood calculation in a way that can be accessed quickly from a numba environment

    :param logdet_base:     Contibution to logdet without CW signal
    :param pos:             (number of pulsars, 3) array holding 3d unit vector pointing towards each pulsar given by psr.pos
    :param pdist:           (number of pulsars, 2) array holding distance and error of distance for each pulsar in kpc given by psr.pdist
    :param toas:            List of arrays of TOAs for each pulsar
    :param Nvecs:           List of N vectors
    :param Nrs:             Residuals times inverse sqareroot N vectors
    :param max_toa:         Maximum TOA over all pulsars
    :param x0:              CWInfo object
    :param Npsr:            Number of pulsars
    :param isqrNvecs:       Inverse squareroot of N vectors
    :param TNvs:            T vectros times inverse squareroot N vectors
    :param dotTNrs:         Precalculated dot product of Nrs and TNvs
    :param chol_Sigmas:     List of Cholesky decompositions of Sigma matrices
    :param phiinvs:         List of phiinv matrices
    :param includeCW:       Switch if we want to include the contribution of the CW signal or not [True]
    :param prior_recovery:  If True, we return constant likelihood to be used for prior recovery diagnostic test [False]
    """
    def __init__(self,logdet_base,pos,pdist,toas,Nvecs,Nrs,max_toa,x0,Npsr,isqrNvecs,TNvs,dotTNrs,chol_Sigmas,phiinvs,includeCW=True,prior_recovery=False):
        self.resres = 0. #compute internally
        self.logdet = 0.
        self.resres_array = np.zeros(Npsr)
        self.logdet_array = np.zeros(Npsr)
        self.logdet_base = logdet_base
        self.logdet_base_orig = logdet_base
        self.set_resres_logdet(self.resres_array,self.logdet_array,self.logdet_base)

        self.pos = pos
        self.pdist = pdist
        self.toas = toas
        self.Npsr = x0.Npsr
        self.max_toa = max_toa

        self.phiinvs = phiinvs
        #self.invchol_Sigma_Ts = invchol_Sigma_Ts
        #self.invchol_Sigma_TNs = invchol_Sigma_TNs#List([invchol_Sigma_Ts[i]/Nvecs[i] for i in range(self.Npsr)])
        self.dotTNrs = dotTNrs
        self.Nvecs = Nvecs
        self.isqrNvecs = isqrNvecs
        self.Nrs = Nrs
        self.TNvs = TNvs
        self.chol_Sigmas = chol_Sigmas

        self.MMs = np.zeros((Npsr,4,4))
        self.NN = np.zeros((Npsr,4))
        
        self.update_intrinsic_params(x0)
        
        self.includeCW=includeCW
        self.prior_recovery=prior_recovery

    def get_lnlikelihood(self,x0):
        """wrapper to get the log likelihood"""
        #assert isclose(self.cos_gwtheta, x0.cos_gwtheta)
        #assert isclose(self.gwphi, x0.gwphi)
        #assert isclose(self.log10_fgw, x0.log10_fgw)
        #assert isclose(self.log10_mc, x0.log10_mc)

        assert self.cos_gwtheta==x0.cos_gwtheta
        assert self.gwphi==x0.gwphi
        assert self.log10_fgw==x0.log10_fgw
        assert self.log10_mc==x0.log10_mc

        return get_lnlikelihood_helper(x0,self.resres,self.logdet,self.pos,self.pdist,self.NN,self.MMs,includeCW=self.includeCW,prior_recovery=self.prior_recovery)

    def update_pulsar_distance(self,x0,psr_idx):
        """recalculate MM and NN only for the affected pulsar if we only change a single pulsar distance"""
        #this method is just a wrapper for the special case of only 1 pulsar but it won't force recompilation
        #assert self.cos_gwtheta == x0.cos_gwtheta
        #assert self.gwphi == x0.gwphi
        #assert self.log10_fgw == x0.log10_fgw
        #assert self.log10_mc == x0.log10_mc
        #self.rn_gammas = x0.rn_gammas.copy() #hot fix
        #self.rn_log10_As = x0.rn_log10_As.copy()
        #assert np.all(self.rn_gammas==x0.rn_gammas)
        #assert np.all(self.rn_log10_As==x0.rn_log10_As)
        assert self.cos_gwtheta==x0.cos_gwtheta
        assert self.gwphi==x0.gwphi
        assert self.log10_fgw==x0.log10_fgw
        assert self.log10_mc==x0.log10_mc
        assert self.gwb_gamma==x0.gwb_gamma
        assert self.gwb_log10_A==x0.gwb_log10_A
        assert np.all(self.rn_gammas==x0.rn_gammas)
        assert np.all(self.rn_log10_As==x0.rn_log10_As)
        assert np.all(self.cw_p_dists[:psr_idx]==x0.cw_p_dists[:psr_idx])
        assert np.all(self.cw_p_dists[psr_idx:]==x0.cw_p_dists[psr_idx:])
        resres_old = self.resres_array.copy()
        update_intrinsic_params2(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.TNvs,self.chol_Sigmas,np.array([psr_idx]),self.resres_array,self.dotTNrs)
        #protect from incorrectly overwriting
        self.cw_p_dists[psr_idx] = x0.cw_p_dists[psr_idx]
        self.resres_array[:] = resres_old
        self.set_resres_logdet(resres_old,self.logdet_array,self.logdet_base)
        #assert np.all(resres_old==self.resres_array)
    
    def update_pulsar_distances(self,x0,psr_idxs):
        """recalculate MM and NN only for the affected pulsar if  change an arbitrary number of single pulsar distances"""
        #assert self.cos_gwtheta == x0.cos_gwtheta
        #assert self.gwphi == x0.gwphi
        #assert self.log10_fgw == x0.log10_fgw
        #assert self.log10_mc == x0.log10_mc
        #self.rn_gammas = x0.rn_gammas.copy() #hot fix
        #self.rn_log10_As = x0.rn_log10_As.copy()
        #assert np.all(self.rn_gammas==x0.rn_gammas)
        #assert np.all(self.rn_log10_As==x0.rn_log10_As)
        if not self.prior_recovery:
            assert self.cos_gwtheta==x0.cos_gwtheta
            assert self.gwphi==x0.gwphi
            assert self.log10_fgw==x0.log10_fgw
            assert self.log10_mc==x0.log10_mc
            assert self.gwb_gamma==x0.gwb_gamma
            assert self.gwb_log10_A==x0.gwb_log10_A
            assert np.all(self.rn_gammas==x0.rn_gammas)
            assert np.all(self.rn_log10_As==x0.rn_log10_As)
        #leave dist_only in even though it is not currently respected in case it turns out to be faster later
        #resres_temp = self.resres_array.copy()
        resres_old = self.resres_array.copy()
        if not self.prior_recovery:
            update_intrinsic_params2(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.TNvs,self.chol_Sigmas,psr_idxs,self.resres_array,self.dotTNrs)
        #protect from incorrectly overwriting
        self.cw_p_dists[:] = x0.cw_p_dists.copy()
        if not self.prior_recovery:
            self.set_resres_logdet(resres_old,self.logdet_array,self.logdet_base)

        #if not np.all(resres_temp==resres_old):
        #    print(resres_temp)
        #    print(resres_old)
        #    print(np.allclose(resres_temp,resres_old))
        #    print(resres_temp==resres_old)
        #    print(psr_idxs)
        #    print(resres_temp[psr_idxs])
        #    print(resres_old[psr_idxs])
        #    self.update_intrinsic_params(x0)
        #    print(self.resres_array[psr_idxs])
        #    assert np.all(resres_old==self.resres_array)
        #    assert np.all(resres_temp==self.resres_array)
        #assert np.all(resres_temp==self.resres_array)


    #def update_intrinsic_params(self,x0):
    #    """Recalculate filters with updated intrinsic parameters - not quite the same as the setup, since a few things are already stored"""
    #    update_intrinsic_params(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.SigmaTNrProds,self.invchol_Sigma_TNs,np.arange(x0.Npsr),dist_only=False)
    #    #track the intrinsic parameters this was set at so we can throw in error if they are inconsistent with an input x0
    #    self.cos_gwtheta = x0.cos_gwtheta
    #    self.gwphi = x0.gwphi
    #    self.log10_fgw = x0.log10_fgw
    #    self.log10_mc = x0.log10_mc

    def update_intrinsic_params(self,x0):
        """Recalculate filters with updated intrinsic parameters - not quite the same as the setup, since a few things are already stored"""
        #TODO ensure complete consistency with handling of resres
        resres_temp = self.resres_array.copy()
        
        if not self.prior_recovery:
            update_intrinsic_params2(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.TNvs,self.chol_Sigmas,np.arange(x0.Npsr),resres_temp,self.dotTNrs)

            self.set_resres_logdet(resres_temp,self.logdet_array,self.logdet_base)
        #track the intrinsic parameters this was set at so we can throw in error if they are inconsistent with an input x0
        self.gwb_gamma = x0.gwb_gamma
        self.gwb_log10_A = x0.gwb_log10_A
        self.rn_gammas = x0.rn_gammas.copy()
        self.rn_log10_As = x0.rn_log10_As.copy()
        self.cos_gwtheta = x0.cos_gwtheta
        self.gwphi = x0.gwphi
        self.log10_fgw = x0.log10_fgw
        self.log10_mc = x0.log10_mc
        self.cw_p_dists = x0.cw_p_dists.copy()

    def update_red_noise(self,x0,psr_idxs):
        """recalculate MM and NN only for the affected pulsars of red noise update - almost same as update_pulsar_distances but with different asserts and param updates"""
        resres_temp = self.resres_array.copy()

        if not self.prior_recovery:
            update_intrinsic_params2(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.TNvs,self.chol_Sigmas,psr_idxs,resres_temp,self.dotTNrs)

            self.set_resres_logdet(resres_temp,self.logdet_array,self.logdet_base)
        #track the intrinsic parameters this was set at so we can throw in error if they are inconsistent with an input x0
        self.gwb_gamma = x0.gwb_gamma
        self.gwb_log10_A = x0.gwb_log10_A
        self.rn_gammas = x0.rn_gammas.copy()
        self.rn_log10_As = x0.rn_log10_As.copy()
        self.cos_gwtheta = x0.cos_gwtheta
        self.gwphi = x0.gwphi
        self.log10_fgw = x0.log10_fgw
        self.log10_mc = x0.log10_mc
        self.cw_p_dists = x0.cw_p_dists.copy()

        #assert self.cos_gwtheta==x0.cos_gwtheta
        #assert self.gwphi==x0.gwphi
        #assert self.log10_fgw==x0.log10_fgw
        #assert self.log10_mc==x0.log10_mc
        #assert self.gwb_gamma==x0.gwb_gamma
        #assert self.gwb_log10_A==x0.gwb_log10_A
        #assert np.all(self.cw_p_dists==x0.cw_p_dists)
        #
        #update_intrinsic_params2(x0,self.isqrNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.TNvs,self.chol_Sigmas,psr_idxs,self.resres_array,self.dotTNrs)
        #self.rn_gammas[:] = x0.rn_gammas.copy()
        #self.rn_log10_As[:] = x0.rn_log10_As.copy()

    def validate_consistent(self,x0):
        """validate parameters are consistent with input x0"""
        assert self.cos_gwtheta==x0.cos_gwtheta
        assert self.gwphi==x0.gwphi
        assert self.log10_fgw==x0.log10_fgw
        assert self.log10_mc==x0.log10_mc
        assert self.gwb_gamma==x0.gwb_gamma
        assert self.gwb_log10_A==x0.gwb_log10_A
        assert np.all(self.rn_gammas==x0.rn_gammas)
        assert np.all(self.rn_log10_As==x0.rn_log10_As)
        assert np.all(self.cw_p_dists==x0.cw_p_dists)
        assert self.logdet_base_orig==self.logdet_base
        assert self.logdet==self.logdet_base+self.logdet_array.sum()
        assert self.resres==self.resres_array.sum()
        return True

    def set_resres_logdet(self,resres_array,logdet_array,logdet_base):
        """always reset resres and logdet with this method for maximum numerical consistency"""
        self.resres_array[:] = resres_array
        self.logdet_array[:] = logdet_array
        self.logdet_base = logdet_base
        self.resres = np.sum(resres_array)
        self.logdet = self.logdet_base+np.sum(self.logdet_array)

@njit()
def isclose(a,b,rtol=1.e-5,atol=1.e-8):
    """check if close in same way as np.isclose

    :param a:       First float to use in comparison
    :param b:       Second float to use in comparison
    :param rtol:    Realtive tolerance - default:1e-5
    :param atol:    Absolute tolerance - default:1e-8

    :return:        True/False indicating if a and b are close to each other
    """
    return np.abs(a - b) <= (atol + rtol * np.abs(b))
