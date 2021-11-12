"""C 2021 Matthew Digman and Bence Becsy
numba version of fast likelihood"""
import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List

from enterprise import constants as const
from lapack_wrappers import solve_triangular



class CWFastLikelihood:
    """
    Class for the fats CW likelihood.
    :param pta: `enterprise` pta object.
    :param params: Dictionary of noise parameters.
    :param x0: CWInfo, which is partially redundant with params but better handled by numba
    """
    def __init__(self, psrs, pta, params, x0):
        self.pta = pta
        self.psrs = psrs
        self.Npsr = x0.Npsr
        self.params = params
        self.tref = 53000*86400

        #put the positions into an array
        self.pos = np.zeros((self.Npsr,3))
        self.pdist = np.zeros((self.Npsr,2))
        for i,psr in enumerate(self.psrs):
            self.pos[i] = psr.pos
            self.pdist[i] = psr.pdist

        #get the N vects without putting them in a matrix
        self.Nvecs = List(self.pta.get_ndiag(self.params))

        #get the part of the determinant that can be computed right now
        self.logdet = 0.0
        self.logdet += np.sum([m for (l,m) in self.pta.get_rNr_logdet(self.params)])

        #get the other pta results
        self.TNTs = self.pta.get_TNT(self.params)
        self.Ts = List(self.pta.get_basis())
        pls_temp = self.pta.get_phiinv(self.params, logdet=True, method='partition')

        self.Sigmas = List()
        #unify types outside numba to avoid slowing down compilation
        #also add more components to logdet
        for i in range(self.Npsr):
            phiinv_loc,logdetphi_loc = pls_temp[i]
            self.logdet += logdetphi_loc
            self.Sigmas.append(self.TNTs[i]+(np.diag(phiinv_loc) if phiinv_loc.ndim == 1 else phiinv_loc))

        #put toas and residuals into a numba typed List of arrays, which shouldn't require any actual copies
        self.toas = List([psr.toas for psr in self.psrs])
        self.residuals = List([psr.residuals for psr in self.psrs])

        self.NN, self.resres,logdet_temp, self.MMs, self.MM_chol = get_MM_NN_resres_logdet(x0,self.Sigmas,self.Ts,self.Nvecs,self.pos,self.pdist,self.tref,self.toas,self.residuals)
        #add the final component to logdet:
        self.logdet += logdet_temp


    def get_lnlikelihood(self,x0):
        """wrapper to get the log likelihood"""
        return get_lnlikelihood_helper(x0,self.resres,self.logdet,self.pos,self.pdist,self.NN,self.MM_chol)

    def update_pulsar_distance(self,x0,psr_idx):
        """recalculate MM and NN only for the affected pulsar if we only change a single pulsar distance"""
        update_MM_NN_new_psr_dist(x0,self.Sigmas,self.Ts,self.Nvecs,self.pos,self.pdist,self.tref,self.toas,self.residuals, psr_idx, self.NN, self.MMs, self.MM_chol)

    def update_intrinsic_params(self,x0):
        """Recalculate filters with updated intrinsic parameters - not quite the same as the setup, since a few things are already stored"""
        update_intrinsic_params(x0,self.Sigmas,self.Ts,self.Nvecs,self.pos,self.pdist,self.tref,self.toas,self.residuals, self.NN, self.MM_chol)

@jitclass([('Npsr',nb.int64),('cw_p_dists',nb.float64[:]),('cw_p_phases',nb.float64[:]),('cos_gwtheta',nb.float64),\
        ('cos_inc',nb.float64),('gwphi',nb.float64),('log10_fgw',nb.float64),('log10_h',nb.float64),\
        ('log10_mc',nb.float64),('phase0',nb.float64),('psi',nb.float64)])
class CWInfo:
    """simple jitclass to store the various parmeters in a way that can be accessed quickly from a numba environment"""
    def __init__(self,Npsr,cw_p_phases,cw_p_dists,cos_gwtheta,cos_inc,gwphi,log10_fgw,log10_h,log10_mc,phase0,psi):
        """parmeters are mostly the same as the params object for the ptas"""
        self.Npsr = Npsr
        self.cw_p_phases = cw_p_phases
        self.cw_p_dists = cw_p_dists
        self.cos_gwtheta = cos_gwtheta
        self.cos_inc = cos_inc
        self.gwphi = gwphi
        self.log10_fgw = log10_fgw
        self.log10_h = log10_h
        self.log10_mc = log10_mc
        self.phase0 = phase0
        self.psi = psi

@njit()
def get_lnlikelihood_helper(x0,resres,logdet,pos,pdist,NN,MM_chol):
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
        m_pos = np.dot(m, pos[i])
        n_pos = np.dot(n, pos[i])
        cosMu = -np.dot(omhat, pos[i])
        F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
        F_c = (m_pos * n_pos) / (1 - cosMu)

        p_dist = (pdist[i,0] + pdist[i,1]*x0.cw_p_dists[i])*const.kpc/const.c

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

        #assume lower triangular cholesky decomposition
        prodMM = np.dot(sigma,MM_chol[i])
        log_L += np.dot(sigma, NN[i]) - 0.5*np.dot(prodMM.T,prodMM)

    return log_L

@njit()
def get_MM_NN_resres_logdet(x0,Sigmas,Ts,Nvecs,pos,pdist,tref,toas,residuals):
    '''Calculate inner products (res|res), N=(res|S), M=(S|S), and logdet'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    resres = 0.0
    NN = np.zeros((x0.Npsr, 4))
    MM = np.zeros((4, 4))
    MMs = np.zeros((x0.Npsr, 4, 4))
    MM_chol = np.zeros((x0.Npsr, 4, 4))

    logdet = 0.

    for ii in range(x0.Npsr):
        T = Ts[ii]
        Sigma = Sigmas[ii]

        #set up filters
        toas_loc = toas[ii] - tref
        omega = w0 * (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(-3./8.)
        phase = 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

        sin_gwtheta = np.sin(gwtheta)
        cos_gwtheta = np.cos(gwtheta)
        sin_gwphi = np.sin(x0.gwphi)
        cos_gwphi = np.cos(x0.gwphi)

        omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

        cosMu = -np.dot(omhat, pos[ii])

        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*const.kpc/const.c

        tp = toas[ii] - tref - p_dist*(1-cosMu)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)
        omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

        phase_p = 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))

        #get the sin and cosine parts
        PT_sin = np.sin(2*phase_p) * (omega_p0/omega_p)**(1./3.)
        PT_cos = np.cos(2*phase_p) * (omega_p0/omega_p)**(1./3.)

        ET_sin = np.sin(2*phase) * (w0/omega)**(1./3.)
        ET_cos = np.cos(2*phase) * (w0/omega)**(1./3.)


        #divide the signals by N

        Nr = residuals[ii]/Nvecs[ii]
        Nes = ET_sin/Nvecs[ii]
        Nec = ET_cos/Nvecs[ii]
        Nps = PT_sin/Nvecs[ii]
        Npc = PT_cos/Nvecs[ii]


        rNr = np.dot(residuals[ii], Nr)
        TNr = np.dot(T.T, Nr)

        TNes = np.dot(T.T, Nes)
        TNec = np.dot(T.T, Nec)
        TNps = np.dot(T.T, Nps)
        TNpc = np.dot(T.T, Npc)

        #lower triangular. use the numpy version because it is numba compatible
        cf = np.linalg.cholesky(Sigma)

        #add the necessary component to logdet
        logdet_sigma = np.sum(2 * np.log(np.diag(cf)))
        logdet += logdet_sigma

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        SigmaTNrProd = solve_triangular(cf, TNr,lower_a=True,trans_a=False)

        dotSigmaTNr = np.dot(SigmaTNrProd.T,SigmaTNrProd)

        resres += rNr - dotSigmaTNr

        #get the solutions to Lx=a for M, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        SigmaTNesProd = solve_triangular(cf, TNes,lower_a=True,trans_a=False)
        SigmaTNecProd = solve_triangular(cf, TNec,lower_a=True,trans_a=False)
        SigmaTNpsProd = solve_triangular(cf, TNps,lower_a=True,trans_a=False)
        SigmaTNpcProd = solve_triangular(cf, TNpc,lower_a=True,trans_a=False)

        dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd)
        dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd)
        dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
        dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

        dotSigmaTNes = np.dot(SigmaTNesProd.T,SigmaTNesProd)
        dotSigmaTNec = np.dot(SigmaTNecProd.T,SigmaTNecProd)
        dotSigmaTNps = np.dot(SigmaTNpsProd.T,SigmaTNpsProd)
        dotSigmaTNpc = np.dot(SigmaTNpcProd.T,SigmaTNpcProd)

        dotSigmaTNeces = np.dot(SigmaTNecProd.T,SigmaTNesProd)
        dotSigmaTNpses = np.dot(SigmaTNpsProd.T,SigmaTNesProd)
        dotSigmaTNpces = np.dot(SigmaTNpcProd.T,SigmaTNesProd)
        dotSigmaTNpsec = np.dot(SigmaTNpsProd.T,SigmaTNecProd)
        dotSigmaTNpcec = np.dot(SigmaTNpcProd.T,SigmaTNecProd)
        dotSigmaTNpcps = np.dot(SigmaTNpcProd.T,SigmaTNpsProd)

        #get the results. Note this could be done slightly more efficiently
        #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
        esNr = np.dot(Nes,residuals[ii])
        ecNr = np.dot(Nec,residuals[ii])
        psNr = np.dot(Nps,residuals[ii])
        pcNr = np.dot(Npc,residuals[ii])

        esNes = np.dot(ET_sin, Nes)
        ecNec = np.dot(ET_cos, Nec)
        psNps = np.dot(PT_sin, Nps)
        pcNpc = np.dot(PT_cos, Npc)

        ecNes = np.dot(ET_cos, Nes)
        psNes = np.dot(PT_sin, Nes)
        pcNes = np.dot(PT_cos, Nes)
        psNec = np.dot(PT_sin, Nec)
        pcNec = np.dot(PT_cos, Nec)
        pcNps = np.dot(PT_cos, Nps)

        #get NN
        NN[ii,0] = esNr - dotSigmaTNesr
        NN[ii,1] = ecNr - dotSigmaTNecr
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[0,0] = esNes - dotSigmaTNes
        MM[1,1] = ecNec - dotSigmaTNec
        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[1,0] = ecNes - dotSigmaTNeces
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,1] = MM[1,0]
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]


        MMs[ii,:,:] = np.copy(MM)
        #we only actually need the cholesky decomposition
        MM_chol[ii] = np.linalg.cholesky(MM)

    return NN, resres, logdet, MMs, MM_chol

@njit()
def update_MM_NN_new_psr_dist(x0,Sigmas,Ts,Nvecs,pos,pdist,tref,toas,residuals, psr_idx, NN, MMs, MM_chol):
    '''Calculate inner products N=(res|S), M=(S|S) for pulsar with changed distance'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    #select pulsar we want to update filters for
    ii = psr_idx

    MM = np.copy(MMs[ii, :, :])

    #print(MM)

    T = Ts[ii]
    Sigma = Sigmas[ii]

    #set up filters
    toas_loc = toas[ii] - tref
    omega = w0 * (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(-3./8.)
    phase = 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

    sin_gwtheta = np.sin(gwtheta)
    cos_gwtheta = np.cos(gwtheta)
    sin_gwphi = np.sin(x0.gwphi)
    cos_gwphi = np.cos(x0.gwphi)

    omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

    cosMu = -np.dot(omhat, pos[ii])

    p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*const.kpc/const.c

    tp = toas[ii] - tref - p_dist*(1-cosMu)
    omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)
    omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

    phase_p = 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))

    #get the sin and cosine parts
    PT_sin = np.sin(2*phase_p) * (omega_p0/omega_p)**(1./3.)
    PT_cos = np.cos(2*phase_p) * (omega_p0/omega_p)**(1./3.)

    ET_sin = np.sin(2*phase) * (w0/omega)**(1./3.)
    ET_cos = np.cos(2*phase) * (w0/omega)**(1./3.)


    #divide the signals by N

    Nr = residuals[ii]/Nvecs[ii]
    Nes = ET_sin/Nvecs[ii]
    Nec = ET_cos/Nvecs[ii]
    Nps = PT_sin/Nvecs[ii]
    Npc = PT_cos/Nvecs[ii]


    #rNr = np.dot(residuals[ii], Nr)
    TNr = np.dot(T.T, Nr)

    TNes = np.dot(T.T, Nes)
    TNec = np.dot(T.T, Nec)
    TNps = np.dot(T.T, Nps)
    TNpc = np.dot(T.T, Npc)

    #lower triangular. use the numpy version because it is numba compatible
    cf = np.linalg.cholesky(Sigma)

    #add the necessary component to logdet
    #logdet_sigma = np.sum(2 * np.log(np.diag(cf)))
    #logdet += logdet_sigma

    #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
    SigmaTNrProd = solve_triangular(cf, TNr,lower_a=True,trans_a=False)

    #dotSigmaTNr = np.dot(SigmaTNrProd.T,SigmaTNrProd)

    #resres += rNr - dotSigmaTNr

    #get the solutions to Lx=a for M, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
    SigmaTNesProd = solve_triangular(cf, TNes,lower_a=True,trans_a=False)
    SigmaTNecProd = solve_triangular(cf, TNec,lower_a=True,trans_a=False)
    SigmaTNpsProd = solve_triangular(cf, TNps,lower_a=True,trans_a=False)
    SigmaTNpcProd = solve_triangular(cf, TNpc,lower_a=True,trans_a=False)

    dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd) #remove#
    dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd) #remove#
    dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
    dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

    dotSigmaTNes = np.dot(SigmaTNesProd.T,SigmaTNesProd) #remove#
    dotSigmaTNec = np.dot(SigmaTNecProd.T,SigmaTNecProd) #remove#
    dotSigmaTNps = np.dot(SigmaTNpsProd.T,SigmaTNpsProd)
    dotSigmaTNpc = np.dot(SigmaTNpcProd.T,SigmaTNpcProd)

    dotSigmaTNeces = np.dot(SigmaTNecProd.T,SigmaTNesProd) #remove#
    dotSigmaTNpses = np.dot(SigmaTNpsProd.T,SigmaTNesProd)
    dotSigmaTNpces = np.dot(SigmaTNpcProd.T,SigmaTNesProd)
    dotSigmaTNpsec = np.dot(SigmaTNpsProd.T,SigmaTNecProd)
    dotSigmaTNpcec = np.dot(SigmaTNpcProd.T,SigmaTNecProd)
    dotSigmaTNpcps = np.dot(SigmaTNpcProd.T,SigmaTNpsProd)

    #get the results. Note this could be done slightly more efficiently
    #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
    esNr = np.dot(Nes,residuals[ii]) #remove#
    ecNr = np.dot(Nec,residuals[ii]) #remove#
    psNr = np.dot(Nps,residuals[ii])
    pcNr = np.dot(Npc,residuals[ii])

    esNes = np.dot(ET_sin, Nes) #remove#
    ecNec = np.dot(ET_cos, Nec) #remove#
    psNps = np.dot(PT_sin, Nps)
    pcNpc = np.dot(PT_cos, Npc)

    ecNes = np.dot(ET_cos, Nes) #remove#
    psNes = np.dot(PT_sin, Nes)
    pcNes = np.dot(PT_cos, Nes)
    psNec = np.dot(PT_sin, Nec)
    pcNec = np.dot(PT_cos, Nec)
    pcNps = np.dot(PT_cos, Nps)

    #get NN
    NN[ii,0] = esNr - dotSigmaTNesr #remove#
    NN[ii,1] = ecNr - dotSigmaTNecr #remove#
    NN[ii,2] = psNr - dotSigmaTNpsr
    NN[ii,3] = pcNr - dotSigmaTNpcr

    #get MM
    #diagonal
    MM[0,0] = esNes - dotSigmaTNes #remove#
    MM[1,1] = ecNec - dotSigmaTNec #remove#
    MM[2,2] = psNps - dotSigmaTNps
    MM[3,3] = pcNpc - dotSigmaTNpc
    #lower triangle
    MM[1,0] = ecNes - dotSigmaTNeces  #remove#
    MM[2,0] = psNes - dotSigmaTNpses
    MM[3,0] = pcNes - dotSigmaTNpces
    MM[2,1] = psNec - dotSigmaTNpsec
    MM[3,1] = pcNec - dotSigmaTNpcec
    MM[3,2] = pcNps - dotSigmaTNpcps
    #upper triangle
    MM[0,1] = MM[1,0] #remove#
    MM[0,2] = MM[2,0]
    MM[0,3] = MM[3,0]
    MM[1,2] = MM[2,1]
    MM[1,3] = MM[3,1]
    MM[2,3] = MM[3,2]

    MMs[ii,:,:] = np.copy(MM)
    #we only actually need the cholesky decomposition
    #print(MM)
    MM_chol[ii] = np.linalg.cholesky(MM)


@njit()
def update_intrinsic_params(x0,Sigmas,Ts,Nvecs,pos,pdist,tref,toas,residuals,NN,MM_chol):
    '''Calculate inner products N=(res|S), M=(S|S)'''

    w0 = np.pi * 10.0**x0.log10_fgw
    mc = 10.0**x0.log10_mc * const.Tsun
    gwtheta = np.arccos(x0.cos_gwtheta)

    resres = 0.0
    MM = np.zeros((4, 4))

    for ii in range(x0.Npsr):
        T = Ts[ii]
        Sigma = Sigmas[ii]

        #set up filters
        toas_loc = toas[ii] - tref
        omega = w0 * (1. - 256./5. * mc**(5./3.) * w0**(8./3.) * toas_loc)**(-3./8.)
        phase = 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))

        sin_gwtheta = np.sin(gwtheta)
        cos_gwtheta = np.cos(gwtheta)
        sin_gwphi = np.sin(x0.gwphi)
        cos_gwphi = np.cos(x0.gwphi)

        omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

        cosMu = -np.dot(omhat, pos[ii])

        p_dist = (pdist[ii,0] + pdist[ii,1]*x0.cw_p_dists[ii])*const.kpc/const.c

        tp = toas[ii] - tref - p_dist*(1-cosMu)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)
        omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)

        phase_p = 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))

        #get the sin and cosine parts
        PT_sin = np.sin(2*phase_p) * (omega_p0/omega_p)**(1./3.)
        PT_cos = np.cos(2*phase_p) * (omega_p0/omega_p)**(1./3.)

        ET_sin = np.sin(2*phase) * (w0/omega)**(1./3.)
        ET_cos = np.cos(2*phase) * (w0/omega)**(1./3.)


        #divide the signals by N

        Nr = residuals[ii]/Nvecs[ii]
        Nes = ET_sin/Nvecs[ii]
        Nec = ET_cos/Nvecs[ii]
        Nps = PT_sin/Nvecs[ii]
        Npc = PT_cos/Nvecs[ii]


        rNr = np.dot(residuals[ii], Nr)
        TNr = np.dot(T.T, Nr)

        TNes = np.dot(T.T, Nes)
        TNec = np.dot(T.T, Nec)
        TNps = np.dot(T.T, Nps)
        TNpc = np.dot(T.T, Npc)

        #lower triangular. use the numpy version because it is numba compatible
        cf = np.linalg.cholesky(Sigma)

        #add the necessary component to logdet
        #logdet_sigma = np.sum(2 * np.log(np.diag(cf)))
        #logdet += logdet_sigma

        #get the solution to Lx=a for N, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        SigmaTNrProd = solve_triangular(cf, TNr,lower_a=True,trans_a=False)

        #dotSigmaTNr = np.dot(SigmaTNrProd.T,SigmaTNrProd)

        #resres += rNr - dotSigmaTNr

        #get the solutions to Lx=a for M, note this uses my own numba compatible lapack wrapper but is basically the same as scipy
        SigmaTNesProd = solve_triangular(cf, TNes,lower_a=True,trans_a=False)
        SigmaTNecProd = solve_triangular(cf, TNec,lower_a=True,trans_a=False)
        SigmaTNpsProd = solve_triangular(cf, TNps,lower_a=True,trans_a=False)
        SigmaTNpcProd = solve_triangular(cf, TNpc,lower_a=True,trans_a=False)

        dotSigmaTNesr = np.dot(SigmaTNesProd,SigmaTNrProd)
        dotSigmaTNecr = np.dot(SigmaTNecProd,SigmaTNrProd)
        dotSigmaTNpsr = np.dot(SigmaTNpsProd,SigmaTNrProd)
        dotSigmaTNpcr = np.dot(SigmaTNpcProd,SigmaTNrProd)

        dotSigmaTNes = np.dot(SigmaTNesProd.T,SigmaTNesProd)
        dotSigmaTNec = np.dot(SigmaTNecProd.T,SigmaTNecProd)
        dotSigmaTNps = np.dot(SigmaTNpsProd.T,SigmaTNpsProd)
        dotSigmaTNpc = np.dot(SigmaTNpcProd.T,SigmaTNpcProd)

        dotSigmaTNeces = np.dot(SigmaTNecProd.T,SigmaTNesProd)
        dotSigmaTNpses = np.dot(SigmaTNpsProd.T,SigmaTNesProd)
        dotSigmaTNpces = np.dot(SigmaTNpcProd.T,SigmaTNesProd)
        dotSigmaTNpsec = np.dot(SigmaTNpsProd.T,SigmaTNecProd)
        dotSigmaTNpcec = np.dot(SigmaTNpcProd.T,SigmaTNecProd)
        dotSigmaTNpcps = np.dot(SigmaTNpcProd.T,SigmaTNpsProd)

        #get the results. Note this could be done slightly more efficiently
        #by shifting a 1/sqrt(N) to the right hand side, but it doesn't really make much difference
        esNr = np.dot(Nes,residuals[ii])
        ecNr = np.dot(Nec,residuals[ii])
        psNr = np.dot(Nps,residuals[ii])
        pcNr = np.dot(Npc,residuals[ii])

        esNes = np.dot(ET_sin, Nes)
        ecNec = np.dot(ET_cos, Nec)
        psNps = np.dot(PT_sin, Nps)
        pcNpc = np.dot(PT_cos, Npc)

        ecNes = np.dot(ET_cos, Nes)
        psNes = np.dot(PT_sin, Nes)
        pcNes = np.dot(PT_cos, Nes)
        psNec = np.dot(PT_sin, Nec)
        pcNec = np.dot(PT_cos, Nec)
        pcNps = np.dot(PT_cos, Nps)

        #get NN
        NN[ii,0] = esNr - dotSigmaTNesr
        NN[ii,1] = ecNr - dotSigmaTNecr
        NN[ii,2] = psNr - dotSigmaTNpsr
        NN[ii,3] = pcNr - dotSigmaTNpcr

        #get MM
        #diagonal
        MM[0,0] = esNes - dotSigmaTNes
        MM[1,1] = ecNec - dotSigmaTNec
        MM[2,2] = psNps - dotSigmaTNps
        MM[3,3] = pcNpc - dotSigmaTNpc
        #lower triangle
        MM[1,0] = ecNes - dotSigmaTNeces
        MM[2,0] = psNes - dotSigmaTNpses
        MM[3,0] = pcNes - dotSigmaTNpces
        MM[2,1] = psNec - dotSigmaTNpsec
        MM[3,1] = pcNec - dotSigmaTNpcec
        MM[3,2] = pcNps - dotSigmaTNpcps
        #upper triangle
        MM[0,1] = MM[1,0]
        MM[0,2] = MM[2,0]
        MM[0,3] = MM[3,0]
        MM[1,2] = MM[2,1]
        MM[1,3] = MM[3,1]
        MM[2,3] = MM[3,2]

        #we only actually need the cholesky decomposition
        MM_chol[ii] = np.linalg.cholesky(MM)


