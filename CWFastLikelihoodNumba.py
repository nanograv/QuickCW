"""C 2021 Matthew Digman and Bence Becsy
numba version of fast likelihood"""
import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List

from enterprise import constants as const
from lapack_wrappers import solve_triangular

import const_mcmc as cm

def get_FastLikeInfo(psrs, pta, params, x0):
    """
    get Class for the fast CW likelihood.
    :param pta: `enterprise` pta object.
    :param params: Dictionary of noise parameters.
    :param x0: CWInfo, which is partially redundant with params but better handled by numba
    """
    Npsr = x0.Npsr

    #put the positions into an array
    pos = np.zeros((Npsr,3))
    pdist = np.zeros((Npsr,2))
    for i,psr in enumerate(psrs):
        pos[i] = psr.pos
        pdist[i] = psr.pdist

    #get the N vects without putting them in a matrix
    Nvecs = List(pta.get_ndiag(params))
    

    #get the part of the determinant that can be computed right now
    logdet = 0.0
    for (l,m) in pta.get_rNr_logdet(params):
        logdet += m
    #self.logdet += np.sum([m for (l,m) in self.pta.get_rNr_logdet(self.params)])

    #get the other pta results
    TNTs = pta.get_TNT(params)
    Ts = pta.get_basis()
    pls_temp = pta.get_phiinv(params, logdet=True, method='partition')

    #invchol_Sigma_Ts = List()
    invchol_Sigma_TNs = List()#List([invchol_Sigma_Ts[i]/Nvecs[i] for i in range(self.Npsr)])
    Nrs = List()
    iNvecs = List()
    #unify types outside numba to avoid slowing down compilation
    #also add more components to logdet

    #put toas and residuals into a numba typed List of arrays, which shouldn't require any actual copies
    toas = List([psr.toas for psr in psrs])
    residuals = List([psr.residuals for psr in psrs])

    max_toa = np.max(toas[0])

    for i in range(Npsr):
        phiinv_loc,logdetphi_loc = pls_temp[i]
        chol_Sigma = np.linalg.cholesky(TNTs[i]+(np.diag(phiinv_loc) if phiinv_loc.ndim == 1 else phiinv_loc))
        #invchol_Sigma_Ts.append(solve_triangular(chol_Sigma,Ts[i].T,lower_a=True,trans_a=False))
        invchol_Sigma_T_loc = solve_triangular(chol_Sigma,Ts[i].T,lower_a=True,trans_a=False)
        iNvecs.append(1/Nvecs[i])
        Nrs.append(residuals[i]/Nvecs[i])
        invchol_Sigma_TNs.append(np.ascontiguousarray(invchol_Sigma_T_loc/Nvecs[i]))

        #find the latest arriving signal to prohibit signals that have already merged
        max_toa = max(max_toa,np.max(toas[i]))

        #add the necessary component to logdet
        logdet += logdetphi_loc+np.sum(2 * np.log(np.diag(chol_Sigma)))

    resres = get_resres(x0,Nvecs,residuals,invchol_Sigma_TNs)

    return FastLikeInfo(resres,logdet,pos,pdist,toas,invchol_Sigma_TNs,Nvecs,Nrs,max_toa,x0,Npsr,iNvecs,residuals)


@jitclass([('Npsr',nb.int64),('cw_p_dists',nb.float64[:]),('cw_p_phases',nb.float64[:]),('cos_gwtheta',nb.float64),\
        ('cos_inc',nb.float64),('gwphi',nb.float64),('log10_fgw',nb.float64),('log10_h',nb.float64),\
        ('log10_mc',nb.float64),('phase0',nb.float64),('psi',nb.float64),\
        ('idx_phases',nb.int64[:]),('idx_dists',nb.int64[:]),('idx_cos_gwtheta',nb.int64),('idx_cos_inc',nb.int64),\
        ('idx_gwphi',nb.int64),('idx_log10_fgw',nb.int64),('idx_log10_mc',nb.int64),('idx_log10_h',nb.int64),('idx_phase0',nb.int64),('idx_psi',nb.int64),
        ('idx_cw_ext',nb.int64[:])])
class CWInfo:
    """simple jitclass to store the various parmeters in a way that can be accessed quickly from a numba environment"""
    def __init__(self,Npsr,params_in,par_names,par_names_cw_ext):
        """parmeters are mostly the same as the params object for the ptas"""
        self.Npsr = Npsr
        self.idx_phases = np.array([par_names.index(par) for par in par_names if "_cw0_p_phase" in par])
        self.idx_dists = np.array([par_names.index(par) for par in par_names if "_cw0_p_dist" in par])

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


@njit()
def get_lnlikelihood_helper(x0,resres,logdet,pos,pdist,NN,MMs):
    """jittable helper for calculating the log likelihood in CWFastLikelihood"""
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

@njit()
def get_hess_fisher_phase_helper(x0,pos,pdist,NN,MMs,psr_idx):
    """jittable helper for calculating the log likelihood in CWFastLikelihood"""
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

    #log_L = -0.5*resres -0.5*logdet

    m_pos = 0.
    n_pos = 0.
    cosMu = 0.
    for j in range(0,3):
        m_pos += m[j]*pos[psr_idx,j]
        n_pos += n[j]*pos[psr_idx,j]
        cosMu -= omhat[j]*pos[psr_idx,j]
    #m_pos = np.dot(m, pos[i])
    #n_pos = np.dot(n, pos[i])
    #cosMu = -np.dot(omhat, pos[i])

    F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
    F_c = (m_pos * n_pos) / (1 - cosMu)

    p_dist = (pdist[psr_idx,0] + pdist[psr_idx,1]*x0.cw_p_dists[psr_idx])*(const.kpc/const.c)

    w0 = np.pi * fgw
    omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

    amp_psr = amp * (w0/omega_p0)**(1.0/3.0)
    phase0_psr = x0.cw_p_phases[psr_idx]

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

    #for j in range(0,4):
    #    log_L += sigma[j]*NN[psr_idx,j]

    #prodMMPart = 0.
    #for j in range(0,4):
    #    for k in range(0,4):
    #        prodMMPart += sigma[j]*MMs[psr_idx,j,k]*sigma[k]
    hess = 4*(NN[psr_idx,2]*sigma[2]+NN[psr_idx,3]*sigma[3])

    hess += -4*((MMs[psr_idx,2,2]-MMs[psr_idx,3,3])*(sigma[2]**2-sigma[3]**2)+4*MMs[psr_idx,2,3]*sigma[2]*sigma[3])
    for j in range(0,2):
        for k in range(2,4):
            hess += -4*sigma[k]*MMs[psr_idx,j,k]*sigma[j]

    #log_L -= prodMMPart/2#np.dot(sigma,np.dot(MMs[i],sigma))/2

    return hess

@njit()
def get_resres(x0,Nvecs,residuals,invchol_Sigma_TNs):
    '''Calculate inner products (res|res)'''

    resres = 0.
    for ii in range(x0.Npsr):
        Nr = residuals[ii]/Nvecs[ii]

        rNr = np.dot(residuals[ii], Nr)

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        invCholSigmaTN = invchol_Sigma_TNs[ii]
        SigmaTNrProd = np.dot(invCholSigmaTN,residuals[ii])

        dotSigmaTNr = np.dot(SigmaTNrProd.T,SigmaTNrProd)

        resres += rNr - dotSigmaTNr

    return resres

@njit(fastmath=True)
def update_MM_NN_new_psr_dists_alt(x0,iNvecs,Nrs,pos,pdist,toas, psr_idxs, NN, MMs,SigmaTNrProds,invchol_Sigma_TNs):
    '''Calculate inner products N=(res|S), M=(S|S) for pulsar with changed distance'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    MM = np.zeros((4,4))#np.copy(MMs[ii, :, :])
    
    for psr_idx in psr_idxs:
        #select pulsar we want to update filters for
        ii = psr_idx


        #set up filters
        toas_loc = toas[ii] - cm.tref

        sin_gwtheta = np.sin(gwtheta)
        cos_gwtheta = np.cos(gwtheta)
        sin_gwphi = np.sin(x0.gwphi)
        cos_gwphi = np.cos(x0.gwphi)

        #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
        #also replace omega with 1/omega**(1/3), which is the quantity we actually need
        omega13 = (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(1/8)
        phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

        omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])
        cosMu = -np.dot(omhat, pos[ii])


        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*(const.kpc/const.c)

        tp = toas_loc - p_dist*(1-cosMu)
        omega_p13 = (1. - 256./5. * mc**(5/3) * w0**(8/3) * tp)**(1/8)
        omega_p013 =(1. + 256./5. * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(1/8)

        phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * (omega_p013**5 - omega_p13**5)

        #get the sin and cosine parts
        PT_sin = np.sin(2*phase_p) * (1/omega_p013)*omega_p13
        PT_cos = np.cos(2*phase_p) * (1/omega_p013)*omega_p13

        ET_sin = np.sin(2*phase) * omega13
        ET_cos = np.cos(2*phase) * omega13

        #divide the signals by N

        Nr = Nrs[ii]#residuals[ii]/Nvecs[ii]
        Nps = PT_sin*iNvecs[ii]
        Npc = PT_cos*iNvecs[ii]

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        #invCholSigmaT = invchol_Sigma_Ts[ii]
        invCholSigmaTN = invchol_Sigma_TNs[ii]
        SigmaTNrProd = SigmaTNrProds[ii]

        SigmaTNesProd = np.dot(invCholSigmaTN,ET_sin)
        SigmaTNecProd = np.dot(invCholSigmaTN,ET_cos)
        SigmaTNpsProd = np.dot(invCholSigmaTN,PT_sin)
        SigmaTNpcProd = np.dot(invCholSigmaTN,PT_cos)

        dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
        dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

        dotSigmaTNps = np.dot(SigmaTNpsProd,SigmaTNpsProd)
        dotSigmaTNpc = np.dot(SigmaTNpcProd,SigmaTNpcProd)

        dotSigmaTNpses = np.dot(SigmaTNpsProd,SigmaTNesProd)
        dotSigmaTNpces = np.dot(SigmaTNpcProd,SigmaTNesProd)
        dotSigmaTNpsec = np.dot(SigmaTNpsProd,SigmaTNecProd)
        dotSigmaTNpcec = np.dot(SigmaTNpcProd,SigmaTNecProd)
        dotSigmaTNpcps = np.dot(SigmaTNpcProd,SigmaTNpsProd)

        #get the results. Note this could be done slightly more efficiently
        #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
        psNr = np.dot(Nr,PT_sin)
        pcNr = np.dot(Nr,PT_cos)

        psNps = np.dot(Nps, PT_sin)
        pcNpc = np.dot(Npc, PT_cos)


        psNes = np.dot(Nps, ET_sin)
        pcNes = np.dot(Npc, ET_sin)
        psNec = np.dot(Nps, ET_cos)
        pcNec = np.dot(Npc, ET_cos)
        pcNps = np.dot(Npc, PT_sin)

        #get NN
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[0,0] = MMs[ii,0,0]
        MM[1,1] = MMs[ii,1,1]

        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[1,0] = MMs[ii,1,0]
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,1] = MMs[ii,0,1]
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]

        MMs[ii,:,:] = MM

@njit(fastmath=True,parallel=True)
def update_MM_NN_new_psr_dists(x0,iNvecs,Nrs,pos,pdist,toas, psr_idxs, NN, MMs, SigmaTNrProds,invchol_Sigma_TNs):
    '''Calculate inner products N=(res|S), M=(S|S) for pulsar with changed distance'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    sin_gwphi = np.sin(x0.gwphi)
    cos_gwphi = np.cos(x0.gwphi)

    omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

    MM = np.zeros((4,4))#np.copy(MMs[ii, :, :])
    
    for psr_idx in psr_idxs:
        #select pulsar we want to update filters for
        ii = psr_idx

        cosMu = -np.dot(omhat, pos[ii])

        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*(const.kpc/const.c)
        omega_p013 = np.sqrt(np.sqrt(np.sqrt((1. + 256./5. * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu)))))

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        invCholSigmaTN = invchol_Sigma_TNs[ii]
        SigmaTNrProd = SigmaTNrProds[ii]

        #SigmaTNesProd = np.zeros(invCholSigmaTN.shape[0])
        #SigmaTNecProd = np.zeros(invCholSigmaTN.shape[0])
        #SigmaTNpsProd = np.zeros(invCholSigmaTN.shape[0])
        #SigmaTNpcProd = np.zeros(invCholSigmaTN.shape[0])

        ET_sin  = np.zeros(invCholSigmaTN.shape[1])
        ET_cos  = np.zeros(invCholSigmaTN.shape[1])

        PT_sin  = np.zeros(invCholSigmaTN.shape[1])
        PT_cos  = np.zeros(invCholSigmaTN.shape[1])

        Nr_loc = Nrs[ii]
        iNvec = iNvecs[ii]

        psNr = 0.
        pcNr = 0.

        psNps = 0.
        pcNpc = 0.

        psNes = 0.
        pcNes = 0.
        psNec = 0.
        pcNec = 0.
        pcNps = 0.


        for itrk in prange(invCholSigmaTN.shape[1]):
            #set up filters
            toa_loc = toas[ii][itrk] - cm.tref
            #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
            #also replace omega with 1/omega**(1/3), which is the quantity we actually need
            omega13 = np.sqrt(np.sqrt(np.sqrt((1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toa_loc))))
            phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

            tp = toa_loc - p_dist*(1-cosMu)
            omega_p13 = np.sqrt(np.sqrt(np.sqrt((1./omega_p013**8 - 256./5. * mc**(5/3) * w0**(8/3)/omega_p013**8 * tp))))

            phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * omega_p013**5 * (1. - omega_p13**5)

            #get the sin and cosine parts
            PT_sin[itrk] = np.sin(2*phase_p) * omega_p13
            PT_cos[itrk] = np.cos(2*phase_p) * omega_p13

            ET_sin[itrk] = np.sin(2*phase) * omega13
            ET_cos[itrk] = np.cos(2*phase) * omega13

            #get the results. Note this could be done slightly more efficiently
            #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
            Nr = Nr_loc[itrk]#residuals[ii]/Nvecs[ii]

            #divide the signals by N
            Nps = PT_sin[itrk]*iNvec[itrk]
            Npc = PT_cos[itrk]*iNvec[itrk]

            psNr += Nr*PT_sin[itrk]
            pcNr += Nr*PT_cos[itrk]

            psNps += Nps*PT_sin[itrk]
            pcNpc += Npc*PT_cos[itrk]

            psNes += Nps*ET_sin[itrk]
            pcNes += Npc*ET_sin[itrk]
            psNec += Nps*ET_cos[itrk]
            pcNec += Npc*ET_cos[itrk]
            pcNps += Npc*PT_sin[itrk]

        dotSigmaTNpsr = 0.
        dotSigmaTNpcr = 0.

        dotSigmaTNps = 0.
        dotSigmaTNpc = 0.

        dotSigmaTNpses = 0.
        dotSigmaTNpces = 0. 
        dotSigmaTNpsec = 0.
        dotSigmaTNpcec = 0.
        dotSigmaTNpcps = 0.

    #        for itrj in prange(invCholSigmaTN.shape[0]):
    #            SigmaTNesProd[itrj] += invCholSigmaTN[itrj,itrk]*ET_sin[itrk]
    #            SigmaTNecProd[itrj] += invCholSigmaTN[itrj,itrk]*ET_cos[itrk]
    #            SigmaTNpsProd[itrj] += invCholSigmaTN[itrj,itrk]*PT_sin[itrk]
    #            SigmaTNpcProd[itrj] += invCholSigmaTN[itrj,itrk]*PT_cos[itrk]
        for itrj in prange(invCholSigmaTN.shape[0]):
            SigmaTNesProd = 0.
            SigmaTNecProd = 0.
            SigmaTNpsProd = 0.
            SigmaTNpcProd = 0.
            for itrk in prange(invCholSigmaTN.shape[1]):
                SigmaTNesProd += invCholSigmaTN[itrj,itrk]*ET_sin[itrk]
                SigmaTNecProd += invCholSigmaTN[itrj,itrk]*ET_cos[itrk]
                SigmaTNpsProd += invCholSigmaTN[itrj,itrk]*PT_sin[itrk]
                SigmaTNpcProd += invCholSigmaTN[itrj,itrk]*PT_cos[itrk]

            #dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd) #remove#
            #dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd) #remove#
            dotSigmaTNpsr += SigmaTNpsProd*SigmaTNrProd[itrj]
            dotSigmaTNpcr += SigmaTNpcProd*SigmaTNrProd[itrj]

            #dotSigmaTNes = np.dot(SigmaTNesProd.T,SigmaTNesProd) #remove#
            #dotSigmaTNec = np.dot(SigmaTNecProd.T,SigmaTNecProd) #remove#
            dotSigmaTNps += SigmaTNpsProd*SigmaTNpsProd
            dotSigmaTNpc += SigmaTNpcProd*SigmaTNpcProd

            #dotSigmaTNeces = np.dot(SigmaTNecProd.T,SigmaTNesProd) #remove#
            dotSigmaTNpses += SigmaTNpsProd*SigmaTNesProd
            dotSigmaTNpces += SigmaTNpcProd*SigmaTNesProd
            dotSigmaTNpsec += SigmaTNpsProd*SigmaTNecProd
            dotSigmaTNpcec += SigmaTNpcProd*SigmaTNecProd
            dotSigmaTNpcps += SigmaTNpcProd*SigmaTNpsProd

        
#        SigmaTNesProd = np.dot(invCholSigmaTN,ET_sin)
#        SigmaTNecProd = np.dot(invCholSigmaTN,ET_cos)
#        SigmaTNpsProd = np.dot(invCholSigmaTN,PT_sin)
#        SigmaTNpcProd = np.dot(invCholSigmaTN,PT_cos)
#
#        #dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd) #remove#
#        #dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd) #remove#
#        dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
#        dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)
#
#        #dotSigmaTNes = np.dot(SigmaTNesProd,SigmaTNesProd) #remove#
#        #dotSigmaTNec = np.dot(SigmaTNecProd,SigmaTNecProd) #remove#
#        dotSigmaTNps = np.dot(SigmaTNpsProd,SigmaTNpsProd)
#        dotSigmaTNpc = np.dot(SigmaTNpcProd,SigmaTNpcProd)
#
#        #dotSigmaTNeces = np.dot(SigmaTNecProd,SigmaTNesProd) #remove#
#        dotSigmaTNpses = np.dot(SigmaTNpsProd,SigmaTNesProd)
#        dotSigmaTNpces = np.dot(SigmaTNpcProd,SigmaTNesProd)
#        dotSigmaTNpsec = np.dot(SigmaTNpsProd,SigmaTNecProd)
#        dotSigmaTNpcec = np.dot(SigmaTNpcProd,SigmaTNecProd)
#        dotSigmaTNpcps = np.dot(SigmaTNpcProd,SigmaTNpsProd)

        #for itrj in prange(invCholSigmaTN.shape[0]):
        #    dotSigmaTNpsr += SigmaTNpsProd[itrj]*SigmaTNrProd[itrj]
        #    dotSigmaTNpcr += SigmaTNpcProd[itrj]*SigmaTNrProd[itrj]

        #    dotSigmaTNps += SigmaTNpsProd[itrj]*SigmaTNpsProd[itrj]
        #    dotSigmaTNpc += SigmaTNpcProd[itrj]*SigmaTNpcProd[itrj]

        #    dotSigmaTNpses += SigmaTNpsProd[itrj]*SigmaTNesProd[itrj]
        #    dotSigmaTNpces += SigmaTNpcProd[itrj]*SigmaTNesProd[itrj]
        #    dotSigmaTNpsec += SigmaTNpsProd[itrj]*SigmaTNecProd[itrj]
        #    dotSigmaTNpcec += SigmaTNpcProd[itrj]*SigmaTNecProd[itrj]
        #    dotSigmaTNpcps += SigmaTNpcProd[itrj]*SigmaTNpsProd[itrj]
        
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[0,0] = MMs[ii,0,0]
        MM[1,1] = MMs[ii,1,1]

        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[1,0] = MMs[ii,1,0]
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,1] = MMs[ii,0,1]
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]

        MMs[ii,:,:] = MM

@njit(fastmath=True)
def update_MM_NN_new_psr_dist(x0,iNvecs,Nrs,pos,pdist,toas, psr_idx, NN, MMs, invchol_Sigma_TNs,SigmaTNrProds):
    '''Calculate inner products N=(res|S), M=(S|S) for pulsar with changed distance'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    #select pulsar we want to update filters for
    ii = psr_idx

    MM = np.zeros((4,4))#np.copy(MMs[ii, :, :])

    #set up filters
    toas_loc = toas[ii] - cm.tref
    #omega = w0 * (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(-3./8.)
    #phase = 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    sin_gwphi = np.sin(x0.gwphi)
    cos_gwphi = np.cos(x0.gwphi)

    #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
    #also replace omega with 1/omega**(1/3), which is the quantity we actually need
    omega13 = (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(1/8)
    phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

    omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])
    cosMu = -np.dot(omhat, pos[ii])


    p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*(const.kpc/const.c)

    tp = toas_loc - p_dist*(1-cosMu)
    omega_p13 = (1. - 256./5. * mc**(5/3) * w0**(8/3) * tp)**(1/8)
    omega_p013 =(1. + 256./5. * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(1/8)

    phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * (omega_p013**5 - omega_p13**5)

    #get the sin and cosine parts
    PT_sin = np.sin(2*phase_p) * (1/omega_p013)*omega_p13
    PT_cos = np.cos(2*phase_p) * (1/omega_p013)*omega_p13

    ET_sin = np.sin(2*phase) * omega13
    ET_cos = np.cos(2*phase) * omega13

    #divide the signals by N

    Nr = Nrs[ii]#residuals[ii]/Nvecs[ii]
    Nes = ET_sin*iNvecs[ii]
    Nec = ET_cos*iNvecs[ii]
    Nps = PT_sin*iNvecs[ii]
    Npc = PT_cos*iNvecs[ii]

    #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
    #invCholSigmaT = invchol_Sigma_Ts[ii]
    invCholSigmaTN = invchol_Sigma_TNs[ii]
    #SigmaTNrProd = np.dot(invCholSigmaT,Nr)
    SigmaTNrProd = SigmaTNrProds[ii]

    SigmaTNesProd = np.dot(invCholSigmaTN,ET_sin)
    SigmaTNecProd = np.dot(invCholSigmaTN,ET_cos)
    SigmaTNpsProd = np.dot(invCholSigmaTN,PT_sin)
    SigmaTNpcProd = np.dot(invCholSigmaTN,PT_cos)

    #dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd) #remove#
    #dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd) #remove#
    dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
    dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

    #dotSigmaTNes = np.dot(SigmaTNesProd.T,SigmaTNesProd) #remove#
    #dotSigmaTNec = np.dot(SigmaTNecProd.T,SigmaTNecProd) #remove#
    dotSigmaTNps = np.dot(SigmaTNpsProd.T,SigmaTNpsProd)
    dotSigmaTNpc = np.dot(SigmaTNpcProd.T,SigmaTNpcProd)

    #dotSigmaTNeces = np.dot(SigmaTNecProd.T,SigmaTNesProd) #remove#
    dotSigmaTNpses = np.dot(SigmaTNpsProd.T,SigmaTNesProd)
    dotSigmaTNpces = np.dot(SigmaTNpcProd.T,SigmaTNesProd)
    dotSigmaTNpsec = np.dot(SigmaTNpsProd.T,SigmaTNecProd)
    dotSigmaTNpcec = np.dot(SigmaTNpcProd.T,SigmaTNecProd)
    dotSigmaTNpcps = np.dot(SigmaTNpcProd.T,SigmaTNpsProd)

    #get the results. Note this could be done slightly more efficiently
    #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
    #esNr = np.dot(ET_sin,Nr) #remove#
    #ecNr = np.dot(ET_cos,Nr) #remove#
    psNr = np.dot(Nr,PT_sin)
    pcNr = np.dot(Nr,PT_cos)

    #esNes = np.dot(ET_sin, Nes) #remove#
    #ecNec = np.dot(ET_cos, Nec) #remove#
    psNps = np.dot(Nps,PT_sin)
    pcNpc = np.dot(Npc,PT_cos)

    #ecNes = np.dot(ET_cos, Nes) #remove#

    psNes = np.dot(Nps, ET_sin)
    pcNes = np.dot(Npc, ET_sin)
    psNec = np.dot(Nps, ET_cos)
    pcNec = np.dot(Npc, ET_cos)
    pcNps = np.dot(Npc, PT_sin)

    #get NN

    #NN[ii,0] = esNr - dotSigmaTNesr #remove#
    #NN[ii,1] = ecNr - dotSigmaTNecr #remove#
    NN[ii,2] = psNr - dotSigmaTNpsr
    NN[ii,3] = pcNr - dotSigmaTNpcr

    #get MM
    #diagonal
    #MM[0,0] = esNes - dotSigmaTNes #remove#
    #MM[1,1] = ecNec - dotSigmaTNec #remove#
    MM[0,0] = MMs[ii,0,0]
    MM[1,1] = MMs[ii,1,1]

    MM[2,2] = psNps - dotSigmaTNps
    MM[3,3] = pcNpc - dotSigmaTNpc
    #lower triangle
    #MM[1,0] = ecNes - dotSigmaTNeces  #remove#
    MM[1,0] = MMs[ii,1,0]
    MM[2,0] = psNes - dotSigmaTNpses
    MM[3,0] = pcNes - dotSigmaTNpces
    MM[2,1] = psNec - dotSigmaTNpsec
    MM[3,1] = pcNec - dotSigmaTNpcec
    MM[3,2] = pcNps - dotSigmaTNpcps
    #upper triangle
    #MM[0,1] = MM[1,0] #remove#
    MM[0,1] = MMs[ii,0,1]
    MM[0,2] = MM[2,0]
    MM[0,3] = MM[3,0]
    MM[1,2] = MM[2,1]
    MM[1,3] = MM[3,1]
    MM[2,3] = MM[3,2]

    MMs[ii,:,:] = MM

@njit(fastmath=True,parallel=True)
def update_intrinsic_params(x0,iNvecs,Nrs,pos,pdist,toas,NN,MMs,SigmaTNrProds,invchol_Sigma_TNs,idxs,dist_only=True):
    '''Calculate inner products N=(res|S), M=(S|S)'''

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
        iNvec = iNvecs[ii]
        toas_in = toas[ii]

        esNr = 0.#np.dot(Nr,ET_sin)
        ecNr = 0.#np.dot(Nr,ET_cos)
        psNr = 0.#np.dot(Nr,PT_sin)
        pcNr = 0.#np.dot(Nr,PT_cos)

        esNes = 0.#np.dot(Nes,ET_sin)
        ecNec = 0.#np.dot(Nec,ET_cos)
        psNps = 0.#np.dot(Nps,PT_sin)
        pcNpc = 0.#np.dot(Npc,PT_cos)

        ecNes = 0.#np.dot(Nec, ET_sin)
        psNes = 0.#np.dot(Nps, ET_sin)
        pcNes = 0.#np.dot(Npc, ET_sin)
        psNec = 0.#np.dot(Nps, ET_cos)
        pcNec = 0.#np.dot(Npc, ET_cos)
        pcNps = 0.#np.dot(Npc, PT_sin)




        #get the sin and cosine parts
        ET_sin  = np.zeros(invCholSigmaTN.shape[1])
        ET_cos  = np.zeros(invCholSigmaTN.shape[1])

        PT_sin  = np.zeros(invCholSigmaTN.shape[1])
        PT_cos  = np.zeros(invCholSigmaTN.shape[1])

        for itrk in prange(invCholSigmaTN.shape[1]):
            #set up filters
            toas_loc = toas_in[itrk] - cm.tref
            #NOTE factored out the common w0 into factor of w0**(-5/3) in phase, cancels in ratios
            #also replace omega with 1/omega**(1/3), which is the quantity we actually need
            omega13 = np.sqrt(np.sqrt(np.sqrt((1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc))))
            phase = 1/32/mc**(5/3) * w0**(-5/3) * (1. - omega13**5)

            tp = toas_loc - p_dist*(1-cosMu)
            omega_p13 = np.sqrt(np.sqrt(np.sqrt((1./omega_p013**8 - 256./5. * mc**(5/3) * w0**(8/3) / omega_p013**8 * tp))))

            phase_p = 1/32*mc**(-5/3) * w0**(-5/3) * omega_p013**5 * (1. - omega_p13**5)

            PT_sin[itrk] = np.sin(2*phase_p) * omega_p13
            PT_cos[itrk] = np.cos(2*phase_p) * omega_p13

            ET_sin[itrk] = np.sin(2*phase) * omega13
            ET_cos[itrk] = np.cos(2*phase) * omega13

            Nps = iNvec[itrk]*PT_sin[itrk]
            Npc = iNvec[itrk]*PT_cos[itrk]

            #get the results

            psNr += Nr[itrk]*PT_sin[itrk]
            pcNr += Nr[itrk]*PT_cos[itrk]

            psNps += Nps*PT_sin[itrk]
            pcNpc += Npc*PT_cos[itrk]

            psNes += Nps*ET_sin[itrk]
            pcNes += Npc*ET_sin[itrk]
            psNec += Nps*ET_cos[itrk]
            pcNec += Npc*ET_cos[itrk]
            pcNps += Npc*PT_sin[itrk]
            
            if not dist_only:
                Nes = iNvec[itrk]*ET_sin[itrk]
                Nec = iNvec[itrk]*ET_cos[itrk]
                esNr += Nr[itrk]*ET_sin[itrk]
                ecNr += Nr[itrk]*ET_cos[itrk]
                esNes += Nes*ET_sin[itrk]
                ecNec += Nec*ET_cos[itrk]
                ecNes += Nec*ET_sin[itrk]

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

    #        for itrj in prange(invCholSigmaTN.shape[0]):
    #            SigmaTNesProd[itrj] += invCholSigmaTN[itrj,itrk]*ET_sin[itrk]
    #            SigmaTNecProd[itrj] += invCholSigmaTN[itrj,itrk]*ET_cos[itrk]
    #            SigmaTNpsProd[itrj] += invCholSigmaTN[itrj,itrk]*PT_sin[itrk]
    #            SigmaTNpcProd[itrj] += invCholSigmaTN[itrj,itrk]*PT_cos[itrk]
        for itrj in prange(invCholSigmaTN.shape[0]):
            SigmaTNesProd = 0.
            SigmaTNecProd = 0.
            SigmaTNpsProd = 0.
            SigmaTNpcProd = 0.
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

            if not dist_only:
                dotSigmaTNesr += SigmaTNesProd*SigmaTNrProd[itrj] #remove#
                dotSigmaTNecr += SigmaTNecProd*SigmaTNrProd[itrj] #remove#
                dotSigmaTNes += SigmaTNesProd*SigmaTNesProd #remove#
                dotSigmaTNec += SigmaTNecProd*SigmaTNecProd #remove#
                dotSigmaTNeces += SigmaTNecProd*SigmaTNesProd #remove#



        #SigmaTNesProd = np.dot(invCholSigmaTN,ET_sin)
        #SigmaTNecProd = np.dot(invCholSigmaTN,ET_cos)
        #SigmaTNpsProd = np.dot(invCholSigmaTN,PT_sin)
        #SigmaTNpcProd = np.dot(invCholSigmaTN,PT_cos)

        #dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd)
        #dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd)
        #dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
        #dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

        #dotSigmaTNes = np.dot(SigmaTNesProd,SigmaTNesProd)
        #dotSigmaTNec = np.dot(SigmaTNecProd,SigmaTNecProd)
        #dotSigmaTNps = np.dot(SigmaTNpsProd,SigmaTNpsProd)
        #dotSigmaTNpc = np.dot(SigmaTNpcProd,SigmaTNpcProd)

        #dotSigmaTNeces = np.dot(SigmaTNecProd,SigmaTNesProd)
        #dotSigmaTNpses = np.dot(SigmaTNpsProd,SigmaTNesProd)
        #dotSigmaTNpces = np.dot(SigmaTNpcProd,SigmaTNesProd)
        #dotSigmaTNpsec = np.dot(SigmaTNpsProd,SigmaTNecProd)
        #dotSigmaTNpcec = np.dot(SigmaTNpcProd,SigmaTNecProd)
        #dotSigmaTNpcps = np.dot(SigmaTNpcProd,SigmaTNpsProd)


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

        if dist_only:
            MM[0:2,0:2] = MMs[ii,0:2,0:2]
        else:
            NN[ii,0] = esNr - dotSigmaTNesr
            NN[ii,1] = ecNr - dotSigmaTNecr

            MM[0,0] = esNes - dotSigmaTNes
            MM[1,1] = ecNec - dotSigmaTNec
            MM[1,0] = ecNes - dotSigmaTNeces
            MM[0,1] = MM[1,0]

        MMs[ii,:,:] = MM

@jitclass([('resres',nb.float64),('logdet',nb.float64),('pos',nb.float64[:,::1]),\
        ('pdist',nb.float64[:,::1]),('NN',nb.float64[:,::1]),('MMs',nb.float64[:,:,::1]),\
        ('toas',nb.types.ListType(nb.types.float64[::1])),('invchol_Sigma_TNs',nb.types.ListType(nb.types.float64[:,::1])),\
        ('Nvecs',nb.types.ListType(nb.types.float64[::1])),('Nrs',nb.types.ListType(nb.types.float64[::1])),\
        ('cos_gwtheta',nb.float64),('gwphi',nb.float64),('log10_fgw',nb.float64),('log10_mc',nb.float64),('max_toa',nb.float64),\
        ('SigmaTNrProds',nb.types.ListType(nb.types.float64[::1])),('Npsr',nb.int64),('iNvecs',nb.types.ListType(nb.types.float64[::1])),('residuals',nb.types.ListType(nb.types.float64[::1]))])
class FastLikeInfo:
    """simple jitclass to store the various elements of fast likelihood calculation in a way that can be accessed quickly from a numba environment"""
    def __init__(self,resres,logdet,pos,pdist,toas,invchol_Sigma_TNs,Nvecs,Nrs,max_toa,x0,Npsr,iNvecs,residuals):
        self.resres = resres
        self.logdet = logdet
        self.pos = pos
        self.pdist = pdist
        self.toas = toas
        self.residuals = residuals
        #self.invchol_Sigma_Ts = invchol_Sigma_Ts
        self.invchol_Sigma_TNs = invchol_Sigma_TNs#List([invchol_Sigma_Ts[i]/Nvecs[i] for i in range(self.Npsr)])

        self.Nvecs = Nvecs
        self.iNvecs = iNvecs
        self.Nrs = Nrs
        self.max_toa = max_toa

        self.Npsr = x0.Npsr

        self.SigmaTNrProds = List([np.dot(self.invchol_Sigma_TNs[i],self.residuals[i]) for i in range(self.Npsr)])

        self.MMs = np.zeros((Npsr,4,4))
        self.NN = np.zeros((Npsr,4))
        self.update_intrinsic_params(x0)


    def get_lnlikelihood(self,x0):
        """wrapper to get the log likelihood"""
        assert self.cos_gwtheta == x0.cos_gwtheta
        assert self.gwphi == x0.gwphi
        assert self.log10_fgw == x0.log10_fgw
        assert self.log10_mc == x0.log10_mc
        return get_lnlikelihood_helper(x0,self.resres,self.logdet,self.pos,self.pdist,self.NN,self.MMs)

    def update_pulsar_distance(self,x0,psr_idx):
        """recalculate MM and NN only for the affected pulsar if we only change a single pulsar distance"""
        assert self.cos_gwtheta == x0.cos_gwtheta
        assert self.gwphi == x0.gwphi
        assert self.log10_fgw == x0.log10_fgw
        assert self.log10_mc == x0.log10_mc
        update_MM_NN_new_psr_dist(x0,self.iNvecs,self.Nrs,self.pos,self.pdist,self.toas,psr_idx, self.NN, self.MMs, self.invchol_Sigma_TNs,self.SigmaTNrProds)

    def update_pulsar_distances(self,x0,psr_idxs):
        """recalculate MM and NN only for the affected pulsar if we only change a single pulsar distance"""
        assert self.cos_gwtheta == x0.cos_gwtheta
        assert self.gwphi == x0.gwphi
        assert self.log10_fgw == x0.log10_fgw
        assert self.log10_mc == x0.log10_mc
        #update_MM_NN_new_psr_dists(x0,self.iNvecs,self.Nrs,self.pos,self.pdist,self.toas,psr_idxs, self.NN, self.MMs, self.SigmaTNrProds,self.invchol_Sigma_TNs)
        update_intrinsic_params(x0,self.iNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.SigmaTNrProds,self.invchol_Sigma_TNs,psr_idxs,dist_only=True)

    def update_pulsar_distances_alt(self,x0,psr_idxs):
        """recalculate MM and NN only for the affected pulsar if we only change a single pulsar distance"""
        assert self.cos_gwtheta == x0.cos_gwtheta
        assert self.gwphi == x0.gwphi
        assert self.log10_fgw == x0.log10_fgw
        assert self.log10_mc == x0.log10_mc
        update_MM_NN_new_psr_dists_alt(x0,self.iNvecs,self.Nrs,self.pos,self.pdist,self.toas,psr_idxs, self.NN, self.MMs, self.SigmaTNrProds,self.invchol_Sigma_TNs)

    def update_intrinsic_params(self,x0):
        """Recalculate filters with updated intrinsic parameters - not quite the same as the setup, since a few things are already stored"""
        update_intrinsic_params(x0,self.iNvecs,self.Nrs,self.pos,self.pdist,self.toas, self.NN, self.MMs,self.SigmaTNrProds,self.invchol_Sigma_TNs,np.arange(x0.Npsr),dist_only=False)
        #track the intrinsic parameters this was set at so we can throw in error if they are inconsistent with an input x0
        self.cos_gwtheta = x0.cos_gwtheta
        self.gwphi = x0.gwphi
        self.log10_fgw = x0.log10_fgw
        self.log10_mc = x0.log10_mc
