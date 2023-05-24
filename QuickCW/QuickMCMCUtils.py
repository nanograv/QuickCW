"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
Helpers for MCMC; extrinsic blocks and parallel tempering"""

from time import perf_counter
import numpy as np
#np.seterr(all='raise')
#make sure to use the right threading layer

from numba import njit,prange
from numba.typed import List
from numpy.random import uniform

import pickle

import QuickCW.CWFastPrior as CWFastPrior
from QuickCW.QuickCorrectionUtils import check_merged,correct_extrinsic,correct_intrinsic
import QuickCW.const_mcmc as cm
import QuickCW.CWFastLikelihoodNumba as CWFastLikelihoodNumba
from QuickCW.QuickFisherHelpers import get_fishers
from QuickCW.QuickMTHelpers import do_intrinsic_update_mt,add_rn_eig_jump
from QuickCW.OutputUtils import print_acceptance_progress,output_hdf5_loop,output_hdf5_end

################################################################################
#
#UPDATE INTRINSIC PARAMETERS AND RECALCULATE FILTERS
#
################################################################################
#version using multiple try mcmc (based on Table 6 of https://vixra.org/pdf/1712.0244v3.pdf)

###############################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################
@njit(parallel=True)
def do_extrinsic_block(n_chain, samples, itrb, Ts, x0s, FLIs, FPI, n_par_tot, log_likelihood, n_int_block, fisher_diag, a_yes, a_no):
    """do blocks of just the extrinsic parameters, which should be very fast

    :param n_chain:         Number of PT chains
    :param samples:         Array holding posterior samples
    :param itrb:            Index within saved values (as opposed to block index itri or overall index itrn)
    :param Ts:              List of PT temperatures
    :param x0s:             List of CWInfo objects
    :param FLIs:            List of FastLikeInfo objects
    :param FPI:             FastPriorInfo object
    :param n_par_tot:       Number of total parameters
    :param log_likelihood:  Array holding log likelihood values
    :param n_int_block:     Number of iterations to be done in a block
    :param fisher_diag:     Diagonal fisher
    :param a_yes:           Array to hold number of accepted steps
    :param a_no:            Array to hold number of rejected steps
    """
    n_par_ext = x0s[0].idx_cw_ext.size

    #treat all phases where fisher is saturated as 1 parameter for counting purposes in jump scaling
    #saturate_count = np.zeros(n_chain,dtype=np.int64)
    #for j in range(0,n_chain):
    #    if Ts[j]>cm.proj_phase_saturate_temp:
    #        saturate_count[j] = x0s[j].Npsr-1



    #for j in range(0,n_chain):
    #    for ii,idx in enumerate(x0s[j].idx_phases):
    #        if Ts[j]>cm.proj_phase_saturate_temp or np.sqrt(Ts[j])*fisher_diag[j][idx]>=1.9:#cm.sigma_cw0_p_phase_default:
    #            saturate_count[j] += 1
    #    #allow one saturated parameter without reducing jump size
    #    if saturate_count[j] > 1:
    #        saturate_count[j] -= 1

    jump_scale_use = np.zeros(n_chain)
    for j in range(n_chain):
        jump_scale_use[j] = 2.38/np.sqrt(n_par_ext)*np.sqrt(Ts[j])

    for k in range(0,n_int_block,2):
        for j in prange(0,n_chain):
            samples_current = samples[j,itrb+k,:]

            if k%4==0 or j==n_chain-1 or Ts[j]>cm.proj_prior_all_temp:  # every 10th k (so every 5th jump) do a prior draw
                new_point = np.copy(samples_current)
                jump_idx = x0s[j].idx_cw_ext
                for ii, idx in enumerate(jump_idx):
                    new_point[idx] = uniform(FPI.cw_ext_lows[ii], FPI.cw_ext_highs[ii])
            else:
                jump = np.zeros(n_par_tot)
                jump_idx = x0s[j].idx_cw_ext
                jump[jump_idx] = jump_scale_use[j]*fisher_diag[j][jump_idx]*np.random.normal(0.,1.,n_par_ext)
                new_point = samples_current + jump

            new_point = correct_extrinsic(new_point,x0s[j])

            x0s[j].update_params(new_point)

            log_L = FLIs[j].get_lnlikelihood(x0s[j])
            log_acc_ratio = log_L/Ts[j]
            log_acc_ratio += CWFastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                                       FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                                       FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                                       FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                                                       FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                                                       FPI.global_common)
            log_acc_ratio += -log_likelihood[j,itrb+k]/Ts[j]
            log_acc_ratio += -CWFastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                                              FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                                              FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                                                              FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                                                              FPI.global_common)

            acc_decide = np.log(uniform(0.0, 1.0, 1))
            if acc_decide<=log_acc_ratio:
                samples[j,itrb+k+1,:] = new_point
                log_likelihood[j,itrb+k+1] = log_L
                a_yes[cm.idx_full,j] += 1
            else:
                samples[j,itrb+k+1,:] = samples[j,itrb+k,:]
                log_likelihood[j,itrb+k+1] = log_likelihood[j,itrb+k]
                a_no[cm.idx_full,j] += 1

                x0s[j].update_params(samples_current)

        do_pt_swap(n_chain, samples, itrb+k+1, Ts, a_yes, a_no, x0s, FLIs, log_likelihood,fisher_diag)


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
@njit()
def do_pt_swap(n_chain, samples, itrb, Ts, a_yes, a_no, x0s, FLIs, log_likelihood,fisher_diag):
    """do the parallel tempering swap

    :param n_chain:         Number of PT chains
    :param samples:         Array holding posterior samples
    :param itrb:            Index within saved values (as opposed to block index itri or overall index itrn)
    :param Ts:              List of PT temperatures
    :param a_yes:           Array to hold number of accepted steps
    :param a_no:            Array to hold number of rejected steps
    :param x0s:             List of CWInfo objects
    :param FLIs:            List of FastLikeInfo objects
    :param log_likelihood:  Array holding log likelihood values
    :param fisher_diag:     Diagonal fisher
    """
    #print("PT")

    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,itrb])

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    #for swap_chain in reversed(range(n_chain-1)):
    for swap_chain in range(n_chain-2, -1, -1):  # same as reversed(range(n_chain-1)) but supported in numba
        assert swap_map[swap_chain] == swap_chain
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain+1]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] / Ts[swap_chain]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain+1]

        acc_decide = np.log(uniform(0.0, 1.0, 1))
        if acc_decide<=log_acc_ratio:# and do_PT:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[cm.idx_PT,swap_chain] += 1
            #a_yes[0,swap_chain]+=1
        else:
            a_no[cm.idx_PT,swap_chain] += 1
            #a_no[0,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    FLIs_new = []
    x0s_new = []
    fisher_diag_new = np.zeros_like(fisher_diag)
    for j in range(n_chain):
        samples[j,itrb+1,:] = samples[swap_map[j],itrb,:]
        fisher_diag_new[j,:] = fisher_diag[swap_map[j],:]
        log_likelihood[j,itrb+1] = log_likelihood[swap_map[j],itrb]
        FLIs_new.append(FLIs[swap_map[j]])
        x0s_new.append(x0s[swap_map[j]])

    fisher_diag[:] = fisher_diag_new
    FLIs[:] = List(FLIs_new)
    x0s[:] = List(x0s_new)

def add_rn_eig_starting_point(samples,par_names,x0_swap,flm,FLI_swap,chain_params,Npsr,FPI):
    """add a fisher eig jump to the starting point of each chain based only on the fisher matrix at the first point

    :param samples:         Samples to perturb
    :param par_names:       List of parameter names
    :param x0_swap:         CWInfo object
    :param flm:             FastLikeMaster object
    :param FLI_swap:        FastLikeInfo object
    :param chain_params:    ChainParams object
    :param Npsr:            Number of pulsars
    :param FPI:             FastPriorInfo object

    :return samples:        Perturbed samples
    """
    eig_rn0,_,_ = get_fishers(samples[0:1],par_names,x0_swap,flm, FLI_swap,\
                              get_diag=False,get_rn_block=True,get_common=False,get_intrinsic_diag=False)
    scaling = 0.*2.38/np.sqrt(2*Npsr/2)
    for j in range(0,chain_params.n_chain):
        scale_eig0 = scaling*eig_rn0[0,:,0,:]
        scale_eig1 = scaling*eig_rn0[0,:,1,:]
        samples[j,0] = add_rn_eig_jump(scale_eig0,scale_eig1,samples[j,0],samples[j,0,x0_swap.idx_rn],x0_swap.idx_rn,Npsr)
        #correct intrinsic just in case
        samples[j,0] = correct_intrinsic(samples[j,0],x0_swap,chain_params.freq_bounds,FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
    return samples


def initialize_de_buffer(sample0,n_par_tot,par_names,chain_params,x0_swap,FPI,eig_rn):
    """set up differential evolution

    :param sample0:         Parameter values to start the RN from
    :param n_par_tot:       Number of total parameters
    :param par_names:       List of parameter names
    :param chain_params:    ChainParams object
    :param x0_swap:         CWInfo object
    :param FPI:             FastPriorInfo object
    :param eig_rn:          RN eigenvectors

    :return de_history:     Array holding initial differenctial evolution buffer
    """
    de_history = np.zeros((chain_params.n_chain, chain_params.de_history_size, n_par_tot))

    #initialize the rn parameters to the starting point plus a fisher matrix jump
    idx_rn = x0_swap.idx_rn
    scaling = 2.38/np.sqrt(2*x0_swap.Npsr/2)
    rn_base = sample0[idx_rn]

    for j in range(chain_params.n_chain):
        scale_eig0 = scaling*np.sqrt(chain_params.Ts[j])*eig_rn[j,:,0,:]
        scale_eig1 = scaling*np.sqrt(chain_params.Ts[j])*eig_rn[j,:,1,:]

        for i in range(chain_params.de_history_size):
            new_point = CWFastPrior.get_sample_full(len(par_names),FPI)

            #reset the red noise parameters to be a fisher matrix jump off of the starting values
            new_point = add_rn_eig_jump(scale_eig0,scale_eig1,new_point,rn_base,idx_rn,x0_swap.Npsr)

            #do corrections just in case
            x0_swap.update_params(new_point)
            new_point = correct_intrinsic(new_point,x0_swap,chain_params.freq_bounds,FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            new_point = correct_extrinsic(new_point,x0_swap)
            de_history[j,i,:] = new_point
    return de_history

def initialize_sample_helper(chain_params,n_par_tot,Npsr,max_toa,par_names,par_names_cw_ext,par_names_cw_int,FPI,pta,noisedict,rn_emp_dist):
    """initialize starting samples for each chain to a random point

    :param chain_params:        ChainParams object
    :param n_par_tot:           Total number of parameters
    :param Npsr:                Number of pulsars
    :param max_toa:             Latest TOA in any pulsar in the array
    :param par_names:           List of parameter names
    :param par_names_cw_ext:    List of parameter names which are projection parameters (previously called extrinsic parameters)
    :param par_names_cw_int:    List of parameter names which are shape parameters (previously called intrinsic parameters)
    :param FPI:                 FastPriorInfo object
    :param pta:                 enterprise PTA object
    :param noisedict:           Noise dictionary
    :param rn_emp_dist:         RN empirical distributions

    :return samples:            Array to hold posterior samples initialized for the first sample
    """
    samples = np.zeros((chain_params.n_chain, chain_params.save_every_n+1, n_par_tot))
    #samples_load = np.load('samples_final_wde_respace3.npy')
    #samples_load = np.load('samples_final_node18.npy')
    for j in range(chain_params.n_chain):
        #samples[j,0,:] = np.array([par.sample() for par in pta.params])
        #if j<samples_load.shape[0]:
        #samples[j,0,:] = samples_load[j]#samples_load[j%min(samples_load.shape[0],5)]
        #acceptable_initial_samples = True
        #else:
        acceptable_initial_samples = False
        #acceptable_initial_samples = False

        itr_accept = 0
        while not acceptable_initial_samples:
            if itr_accept >= 10:
                raise RuntimeError('failed to find acceptable initial sample')
            #samples[j,0,:] = np.array([CWFastPrior.get_sample_helper(i, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,
            #                                                            FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,
            #                                                            FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs) for i in range(len(par_names))])
            samples[j,0,:] = CWFastPrior.get_sample_full(len(par_names),FPI)
            #do correct intrinsic and correct extrinsic just in case
            if itr_accept == 0 and j==0:
                x0_swap = CWFastLikelihoodNumba.CWInfo(Npsr,samples[j,0],par_names,par_names_cw_ext,par_names_cw_int)
            else:
                x0_swap.update_params(samples[j,0,:])

            for psr in pta.pulsars:
                if chain_params.zero_rn:
                    print("zero_rn=True --> Setting " + psr + "_red_noise_gamma=0.0")
                    samples[j,0,par_names.index(psr + "_red_noise_gamma")] = 0.0
                elif rn_emp_dist is not None:
                    psr_idx = pta.pulsars.index(psr)
                    samples[j,0,par_names.index(psr + "_red_noise_gamma")] = rn_emp_dist[psr_idx].draw()[1]
                    #print(samples[j,0,par_names.index(psr + "_red_noise_gamma")])
                elif (psr + "_red_noise_gamma") in noisedict.keys():
                    samples[j,0,par_names.index(psr + "_red_noise_gamma")] = noisedict[psr + "_red_noise_gamma"]
                else:
                    print("No value found in noisedict for: " + psr + "_red_noise_gamma")
                    print("Using a random draw from the prior as a first sample instead")
                    print(samples[j,0,par_names.index(psr + "_red_noise_gamma")])

                if chain_params.zero_rn:
                    print("zero_rn=True --> Setting " + psr + "_red_noise_log10_A=-20.0")
                    samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = -20.0
                elif rn_emp_dist is not None:
                    psr_idx = pta.pulsars.index(psr)
                    samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = rn_emp_dist[psr_idx].draw()[0]
                    #print(samples[j,0,par_names.index(psr + "_red_noise_log10_A")])
                elif (psr + "_red_noise_log10_A") in noisedict.keys():
                    samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = noisedict[psr + "_red_noise_log10_A"]
                else:
                    print("No value found in noisedict for: " + psr + "_red_noise_log10_A")
                    print("Setting it to a low value of -19 to help convergence of insignificant RN")
                    samples[j,0,par_names.index(psr + "_red_noise_log10_A")] = -19.0

            if chain_params.zero_gwb:
                print("zero_gwb=True --> Setting gwb_gamma=0.0")
                samples[j,0,par_names.index("gwb_gamma")] = 0.0
            elif "gwb_gamma" in noisedict.keys():
                samples[j,0,par_names.index("gwb_gamma")] = noisedict["gwb_gamma"]
            else:
                print("No value found in noisedict for: gwb_gamma")
                print("Using a random draw from the prior as a first sample instead")

            if chain_params.zero_gwb:
                print("zero_gwb=True --> Setting gwb_log10_A=-20.0")
                samples[j,0,par_names.index("gwb_log10_A")] = -20.0
            elif "gwb_log10_A" in noisedict.keys():
                samples[j,0,par_names.index("gwb_log10_A")] = noisedict["gwb_log10_A"]
            else:
                print("No value found in noisedict for: gwb_log10_A")
                print("Setting it to a low value of -19 to help convergence of insignificant RN")
                samples[j,0,par_names.index("gwb_log10_A")] = -19.0

            samples[j,0,:] = correct_intrinsic(samples[j,0,:],x0_swap,chain_params.freq_bounds,FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            samples[j,0,:] = correct_extrinsic(samples[j,0,:],x0_swap)
            #check the maximum toa is not such that the source has already merged, and if so draw new parameters to avoid starting from nan likelihood
            acceptable_initial_samples = not check_merged(samples[j,0,par_names.index("0_log10_fgw")],samples[j,0,par_names.index("0_log10_mc")],max_toa)
            itr_accept += 1

    return samples

def get_param_names(pta):
    """get the name Lists for various parameters

    :param pta:                 enterprise PTA object

    :return par_names:          List of parameter names
    :return par_names_cw:       List of parameter names describing the CW signal
    :return par_names_cw_int:   List of parameter names which are projection parameters (previously called extrinsic parameters)
    :return par_names_cw_ext:   List of parameter names which are shape parameters (previously called intrinsic parameters)
    :return par_names_noise:    List of noise parameter names
    """
    par_names = List(pta.param_names)
    par_names_cw = List(['0_cos_gwtheta', '0_cos_inc', '0_gwphi', '0_log10_fgw', '0_log10_h',
                         '0_log10_mc', '0_phase0', '0_psi'])
    par_names_cw_ext = List(['0_cos_inc', '0_log10_h', '0_phase0', '0_psi'])
    par_names_cw_int = List(['0_cos_gwtheta', '0_gwphi', '0_log10_fgw', '0_log10_mc'])

    par_names_noise = ['gwb_gamma', 'gwb_log10_A']

    for i,psr in enumerate(pta.pulsars):
        par_names_cw.append(psr + "_cw0_p_dist")
        par_names_cw.append(psr + "_cw0_p_phase")
        par_names_cw_ext.append(psr + "_cw0_p_phase")
        par_names_cw_int.append(psr + "_cw0_p_dist")
        par_names_noise.append(psr + "_red_noise_gamma")
        par_names_noise.append(psr + "_red_noise_log10_A")

    return par_names,par_names_cw,par_names_cw_int,par_names_cw_ext,par_names_noise

class ChainParams():
    """store basic parameters the govern the evolution of the mcmc chain

    :param T_max:                   Maximum temperature of PT ladder
    :param n_chain:                 Number of PT chains
    :param n_block_status_update:   Number of blocks between status updates
    :param n_int_block:             Number of iterations in a block [1_000]
    :param n_update_fisher:         Number of iterations between Fisher updates [100_000]
    :param save_every_n:            Number of iterations between saving intermediate results (needs to be intiger multiple of n_int_block) [10_000]
    :param fisher_eig_downsample:   Multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements [10]
    :param T_ladder:                Temperature ladder; if None, geometrically spaced ladder is made with n_chain chains reaching T_max [None]
    :param includeCW:               If False, we are not including the CW in the likelihood (good for testing) [True]
    :param prior_recovery:          If True, likelihood is set to a constant (good for testing the prior recovery of the MCMC) [False]
    :param verbosity:               Parameter indicating how much info to print (higher value means more prints) [1]
    :param freq_bounds:             Lower and upper prior bounds on the GW frequency of the CW; np.nan lower bound is automatically turned into one over the observation time [[np.nan, 1.e-07]] 
    :param gwb_comps:               Number of frequency components to model in the GWB [14]
    :param cos_gwtheta_bounds:      Prior bounds on the cosine of the GW theta sky location parameter (useful e.g. for targeted searches) [[-1,1]]
    :param gwphi_bounds:            Prior bounds on the the GW phi sky location parameter (useful e.g. for targeted searches) [[0,2*np.pi]]
    :param de_history_size:         Size of the differential evolution buffer
    :param thin_de:                 How much to thin samples for the DE buffer
    :param log_fishers:             --
    :param log_mean_likelihoods:    --
    :param savefile:                File name to save the results to, if None, no results are saved [None]
    :param thin:                    How much to thin the samples by for saving [100]
    :param samples_precision:       Precision to use for the saved samples [np.single]
    :param save_first_n_chains:     Number of PT chains to save [1]
    :param prior_draw_prob:         Probability of prior draws [0.1]
    :param de_prob:                 Probability of DE jumps [0.6]
    :param fisher_prob:             Probability of fisher updates [0.3]
    :param rn_emp_dist_file:        Filename with empirical distribution to use for per psr RN, if None, do not do empirical distribution jumps [None]
    :param dist_jump_weight:        Weight if jumps changing pulsar distances [0.2]
    :param rn_jump_weight:          Weight of jumps changing RN parameters [0.3]
    :param gwb_jump_weight:         Weight of jumps changing GWB parameters [0.1]
    :param common_jump_weight:      Weight of jumps changing common CW shape parameters (sky location, frequency, chirp mass) [0.2]
    :param all_jump_weight:         Weight of jumps changing all parameters [0.2]
    :param fix_rn:                  If True, we fix per psr RN parameters to the value it starts at [False]
    :param zero_rn:                 If True, we fix per psr RN amplitude to a very low value effectively turning it off [False]
    :param fix_gwb:                 If True, we fix GWB parameters to the value it starts at [False]
    :param zero_gwb:                If True, we fix GWB amplitude to a very low value effectively turning it off [False]
    """

    def __init__(self, T_max: float, n_chain: int, n_block_status_update: int, n_int_block: int = 1000,
                 n_update_fisher: int = 100_000, save_every_n: int = 10_000,
                 fisher_eig_downsample: int = 10, T_ladder: list = None,
                 includeCW: bool = True, prior_recovery: bool = False, verbosity: int = 1,
                 freq_bounds: np.ndarray = np.array([np.nan, 1e-7], dtype=np.float64), gwb_comps: int = 14,
                 cos_gwtheta_bounds: np.ndarray = np.array([-1,1]), gwphi_bounds: np.ndarray = np.array([0,2*np.pi]),
                 de_history_size: int = 5_000, thin_de: int = 10_000,
                 log_fishers: bool = False, log_mean_likelihoods: bool = True,
                 savefile: str = None, thin: int = 100, samples_precision: type = np.single,
                 save_first_n_chains: int = 1,
                 prior_draw_prob: float = 0.1, de_prob: float = 0.6, fisher_prob: float = 0.3,
                 rn_emp_dist_file: str = None,
                 dist_jump_weight: float = 0.2, rn_jump_weight: float = 0.3, gwb_jump_weight: float = 0.1,
                 common_jump_weight: float = 0.2,
                 all_jump_weight: float = 0.2,
                 fix_rn: bool = False, zero_rn: bool = False, fix_gwb: bool = False, zero_gwb: bool = False):
        assert n_int_block % 2 == 0 and n_int_block >= 4  # need to have n_int block>=4 a multiple of 2
        # in order to always do at least n*(1 extrinsic+1 pt swap)+(1 intrinsic+1 pt swaps)
        assert save_every_n % n_int_block == 0  # or we won't save
        assert n_update_fisher % n_int_block == 0  # or we won't update fisher
        self.n_chain = n_chain
        self.n_int_block = n_int_block
        self.n_update_fisher = n_update_fisher
        self.save_every_n = save_every_n
        #multiplier for how much less to do more expensive updates to fisher eigendirections for red noise and common parameters compared to diagonal elements
        self.fisher_eig_downsample = fisher_eig_downsample
        self.n_update_fisher_eig = self.n_update_fisher*self.fisher_eig_downsample
        self.T_max = T_max
        self.T_ladder = T_ladder
        self.includeCW = includeCW
        self.prior_recovery = prior_recovery
        self.verbosity = verbosity
        self.freq_bounds = freq_bounds
        self.gwb_comps = gwb_comps
        self.cos_gwtheta_bounds=cos_gwtheta_bounds
        self.gwphi_bounds=gwphi_bounds
        self.de_history_size = de_history_size
        self.thin_de = thin_de
        self.log_fishers = log_fishers
        self.log_mean_likelihoods = log_mean_likelihoods
        self.rn_emp_dist_file = rn_emp_dist_file

        if T_ladder is None:
            #using geometric spacing
            c = self.T_max**(1./(self.n_chain-1))
            self.Ts = c**np.arange(self.n_chain)

            print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\nTemperature ladder is:\n".format(self.n_chain,c),self.Ts)
        else:
            self.Ts = np.array(T_ladder)
            self.n_chain = self.Ts.size
            print("Using {0} temperature chains with custom spacing: ".format(self.n_chain),self.Ts)

        #store the set of parameters which are allowed to change between calls to advance_N_blocks
        self.n_block_status_update = n_block_status_update
        self.savefile = savefile
        self.thin = thin
        self.samples_precision = samples_precision
        self.save_first_n_chains = save_first_n_chains

        #jump type probabilities
        self.prior_draw_prob = prior_draw_prob
        self.de_prob = de_prob
        self.fisher_prob = fisher_prob

        #rn switches
        self.zero_rn = zero_rn
        #also fix rn if it's set to be zero
        if self.zero_rn:
            self.fix_rn = True
        else:
            self.fix_rn = fix_rn

        #gwb switches
        self.zero_gwb = zero_gwb
        #also fix gwb if it's set to be zero
        if self.zero_gwb:
            self.fix_gwb = True
        else:
            self.fix_gwb = fix_gwb

        #jump parameter set probabilities
        self.dist_jump_weight = dist_jump_weight
        self.common_jump_weight = common_jump_weight
        if self.fix_rn:
            print("Overwrite rn_jump_weight to 0, due to fix_rn=True.")
            self.rn_jump_weight = 0.0
        else:
            self.rn_jump_weight = rn_jump_weight
        if self.fix_gwb:
            print("Overwrite gwb_jump_weight to 0, due to fix_gwb=True.")
            self.gwb_jump_weight = 0.0
        else:
            self.gwb_jump_weight = gwb_jump_weight
        if self.fix_gwb or self.fix_rn:
            print("Overwrite all_jump_weight to 0, due to either fix_rn or fix_gwb being True.")
            self.all_jump_weight = 0.0
        else:
            self.all_jump_weight = all_jump_weight

        #jump parameters to control number of eigendirections
        #TODO make these actual arguments
        self.n_ext_directions = 32
        self.n_phase_extra = 16

        self.n_dist_extra = 67
        self.n_dist_main = 67

        self.n_noise_emp_dist = 20#5#3#1#30#67#1

        self.big_de_jump_prob = 0.5


class MCMCChain():
    """store the miscellaneous objects needed to manage the mcmc chain

    :param chain_params:    ChainParams object
    :param psrs:            List of enterprise pulsar objects
    :param pta:             enterprise PTA object
    :param max_toa:         Latest TOA in any pulsar in the array
    :param noisedict:       Noise dictionary
    :param ti:              Time after initialization got from time.perf_counter()
    """
    def __init__(self,chain_params,psrs,pta,max_toa,noisedict,ti):
        #set up fast likelihoods
        self.chain_params = chain_params
        self.ti = ti
        self.includeCW = self.chain_params.includeCW
        self.prior_recovery = self.chain_params.prior_recovery
        self.max_toa = max_toa
        self.n_chain = self.chain_params.n_chain
        self.pta = pta
        #gte parameter names
        self.par_names,self.par_names_cw,self.par_names_cw_int,self.par_names_cw_ext,self.par_names_noise = get_param_names(self.pta)
        self.n_par_tot = len(self.par_names)
        self.n_int_block = self.chain_params.n_int_block
        self.n_update_fisher = self.chain_params.n_update_fisher
        self.psrs = psrs
        self.Npsr = len(self.pta.pulsars)
        self.noisedict = noisedict
        self.verbosity = self.chain_params.verbosity
        self.itri = 0

        self.fisher_diag_logs = []
        self.fisher_eig_logs = []
        self.fisher_common_logs = []
        self.mean_likelihoods = []
        self.max_likelihoods = []


        self.FPI = CWFastPrior.get_FastPriorInfo(self.pta,self.psrs,self.par_names_cw_ext)
        if self.verbosity>0:
            print('uniform ',self.FPI.uniform_par_ids)
            print('normal ',self.FPI.normal_par_ids)
            print('lin exp ',self.FPI.lin_exp_par_ids)
            print('dm dist ',self.FPI.dm_par_ids)
            print('px dist ',self.FPI.px_par_ids)

        #assert np.all(FPI.cw_ext_lows==cw_ext_lows)
        #assert np.all(FPI.cw_ext_highs==cw_ext_highs)
        print(self.FPI.cw_ext_lows)
        print(self.FPI.cw_ext_highs)

        #read in RN empirical distribution files if provided
        if self.chain_params.rn_emp_dist_file is not None:
            print("Reading in RN empirical distributions...")
            with open(self.chain_params.rn_emp_dist_file, 'rb') as f:
                self.rn_emp_dist = pickle.load(f)
            #create a temperature adapted empirical distribution
            #self.rn_emp_dist_adapt = []
            #for j,T in enumerate(self.chain_params.Ts):
            #    emp_dist_loc = []
            #    for emp_dist0 in self.rn_emp_dist:
            #        emp_dist_loc.append(TemperatureAdaptedEmpiricalDistribution(emp_dist0,T))

            #    self.rn_emp_dist_adapt.append(emp_dist_loc)


        else:
            self.rn_emp_dist = None
            #self.rn_emp_dist_adapt = None

        #set up samples array
        t1 = perf_counter()
        print("Setting up first sample at %8.3fs..."%(t1-self.ti))
        self.samples = initialize_sample_helper(self.chain_params,self.n_par_tot,self.Npsr,self.max_toa,self.par_names,self.par_names_cw_ext,self.par_names_cw_int,self.FPI,self.pta,self.noisedict, self.rn_emp_dist)


        print("log_prior="+str(CWFastPrior.get_lnprior(self.samples[0,0,:], self.FPI)))

        #set up log_likelihood array
        self.log_likelihood = np.zeros((self.n_chain,self.chain_params.save_every_n+1))

        t1 = perf_counter()
        print("Creating Shared Info Objects at %8.3fs"%(t1-ti))
        self.x0_swap = CWFastLikelihoodNumba.CWInfo(self.Npsr,self.samples[0,0],self.par_names,self.par_names_cw_ext,self.par_names_cw_int)
        #TODO why was this distance zeroing here? Shouldn't change from initialized state
        #self.samples[:,0,self.x0_swap.idx_dists] = 0.

        self.flm = CWFastLikelihoodNumba.FastLikeMaster(self.psrs,self.pta,dict(zip(self.par_names, self.samples[0, 0, :])),self.x0_swap,
                                                        includeCW=self.includeCW,prior_recovery=self.prior_recovery)
        self.FLI_swap = self.flm.get_new_FastLike(self.x0_swap, dict(zip(self.par_names, self.samples[0, 0, :])))

        #add a random fisher eigenvalue jump to the starting point for the j>0 chains to get more diversity in the initial fisher matrices
        self.samples = add_rn_eig_starting_point(self.samples,self.par_names,self.x0_swap,self.flm,self.FLI_swap,self.chain_params,self.Npsr,self.FPI)

        self.x0s = List([])
        self.FLIs  = List([])
        for j in range(self.n_chain):
            self.x0s.append( CWFastLikelihoodNumba.CWInfo(self.Npsr,self.samples[j,0],self.par_names,self.par_names_cw_ext,self.par_names_cw_int))
            self.FLIs.append(self.flm.get_new_FastLike(self.x0s[j], dict(zip(self.par_names, self.samples[j, 0, :]))))

        #make extra x0s to help parallelizing MTMCMC updates
        self.x0_extras = List([])
        for k in range(cm.n_x0_extra):
            self.x0_extras.append(CWFastLikelihoodNumba.CWInfo(len(self.pta.pulsars),self.samples[0,0],self.par_names,self.par_names_cw_ext,self.par_names_cw_int))

        t1 = perf_counter()
        print("Finished Creating Shared Info Objects at %8.3fs"%(t1-self.ti))

        self.dist_prior_sigmas = []
        for dist_idx in self.x0s[0].idx_dists:
            if dist_idx in self.FPI.normal_par_ids:
                self.dist_prior_sigmas.append(self.FPI.normal_sigs[list(self.FPI.normal_par_ids).index(dist_idx)])
            elif dist_idx in self.FPI.dm_par_ids:
                self.dist_prior_sigmas.append(self.FPI.dm_errs[list(self.FPI.dm_par_ids).index(dist_idx)])
            elif dist_idx in self.FPI.px_par_ids:
                self.dist_prior_sigmas.append(self.FPI.px_errs[list(self.FPI.px_par_ids).index(dist_idx)]/self.FPI.px_mus[list(self.FPI.px_par_ids).index(dist_idx)]**2)

        self.dist_prior_sigmas = np.array(self.dist_prior_sigmas)

        print("Distance prior sigmas calculated for fisher correction:")
        print(self.dist_prior_sigmas)

        t1 = perf_counter()
        print("Geting Starting Fishers at %8.3fs"%(t1-self.ti))
        #get only the starting point so we can add a fisher red noise jump to the others
        self.samples_sel = np.zeros((self.n_chain,1,self.samples.shape[2]))
        self.samples_sel[:,0,:] = self.samples[:,0,:]
        #don't bother getting common eigenvectors for the starting positions because they probably aren't at a maximum; will use defaults instead
        self.eig_rn,self.fisher_diag,self.eig_common = get_fishers(self.samples_sel,self.par_names,self.x0_swap, self.flm, self.FLI_swap,\
                                                                   get_diag=True,get_rn_block=True,get_common=False,get_intrinsic_diag=True)

        self.fisher_diag_next = np.zeros_like(self.fisher_diag)
        self.fisher_diag_next2 = np.zeros_like(self.fisher_diag)
        self.eig_rn_next = np.zeros_like(self.eig_rn)
        self.eig_common_next = np.zeros_like(self.eig_common)
        #chose the indices of parameters where fisher matrix diagonals will next be computed so they can be stored before they are erased if n_update_fisher is larger than n_int_block
        self.samples_sel_next = np.zeros((self.n_chain,1,self.samples.shape[2]))
        self.idx_fisher_sel_next = np.random.randint(0,self.n_update_fisher,self.n_chain)#np.zeros(self.n_chain,dtype=np.int64)

        #chose the indices of parameters where fisher matrix eigenvalues will next be computed
        self.samples_sel_next_eig = np.zeros((self.n_chain,1,self.samples.shape[2]))
        self.idx_fisher_sel_next_eig = np.random.randint(0,self.chain_params.n_update_fisher_eig,self.n_chain)#np.zeros(self.n_chain,dtype=np.int64)

        self.fisher_eig_logs.append(self.eig_rn.copy())
        self.fisher_diag_logs.append(self.fisher_diag.copy())
        self.fisher_common_logs.append(self.eig_common.copy())
        t1 = perf_counter()
        print("Finished Getting Starting Fishers at %8.3fs"%(t1-self.ti))


        self.de_history = initialize_de_buffer(self.samples[0,0],self.n_par_tot,self.par_names,self.chain_params,self.x0_swap,self.FPI,self.eig_rn)
        self.x0_swap.update_params(self.samples[0,0])

        t1 = perf_counter()
        print("Finished Setting up Differential Evolution Buffer at %8.3fs"%(t1-self.ti))

        self.a_yes = np.zeros((32,self.n_chain),dtype=np.int64)
        self.a_no = np.zeros((32,self.n_chain),dtype=np.int64)

        with np.errstate(invalid='ignore'):
            self.acc_fraction = self.a_yes/(self.a_no+self.a_yes)

        #printing info about initial parameters
        for j in range(self.n_chain):
            print("chain #"+str(j))
            self.log_likelihood[j,0] = self.FLIs[j].get_lnlikelihood(self.x0s[j])
            print("log_likelihood="+str(self.log_likelihood[j,0]))
            print("log_prior_old="+str(self.pta.get_lnprior(self.samples[j,0,:])))
            print("log_prior_new="+str(CWFastPrior.get_lnprior(self.samples[j,0,:],self.FPI)))
            print("Initial samples:")
            print(self.samples[j,0,:])

        #for j in range(self.n_chain):
        #    print("j="+str(j))
        #    print(self.samples[j,0,:])
        #    #log_likelihood[j,0] = pta.get_lnlikelihood(samples[j,0,:])
        #    self.log_likelihood[j,0] = self.FLIs[j].get_lnlikelihood(self.x0s[j])
        #    print("log_likelihood="+str(self.log_likelihood[j,0]))
        #    print("log_prior="+str(CWFastPrior.get_lnprior(self.samples[j,0,:], self.FPI)))

        self.validate_consistent(0)  # check things were initialized as expected

        self.best_logL = self.log_likelihood[0,0]
        best_logL_idx = np.argmax(self.log_likelihood[:,0])
        self.best_logL_global = self.log_likelihood[best_logL_idx,0]
        self.best_sample_global = self.samples[best_logL_idx,0].copy()

        self.tf_init = perf_counter()
        print("finished initialization steps in %8.3fs"%(self.tf_init-self.ti))
        self.ti_loop = perf_counter()
        self.tf1_loop = perf_counter()
    #@profile
    def advance_block(self):
        """advance the state of the mcmc chain by 1 entire block, updating fisher matrices and differential evolution as necessary"""
        itrn = self.itri*self.n_int_block  # index overall
        itrb = itrn%self.chain_params.save_every_n  # index within the block of saved values
        self.validate_consistent(itrb)  # check FLIs and x0s appear to have internally consistent parameters
        #always do pt steps in extrinsic
        do_extrinsic_block(self.n_chain, self.samples, itrb, self.chain_params.Ts, self.x0s, self.FLIs, self.FPI, self.n_par_tot, self.log_likelihood, self.n_int_block-2, self.fisher_diag, self.a_yes, self.a_no)

        self.update_fishers_partial(itrn,itrn+self.n_int_block-1)

        #update intrinsic parameters once a block
        self.FLI_swap = do_intrinsic_update_mt(self, itrb+self.n_int_block-2)
        self.validate_consistent(itrb+self.n_int_block-1,full_validate=False)  # check FLIs and x0s appear to have internally consistent parameters

        do_pt_swap(self.n_chain, self.samples, itrb+self.n_int_block-1, self.chain_params.Ts, self.a_yes,self.a_no, self.x0s, self.FLIs, self.log_likelihood, self.fisher_diag)
        self.update_fishers_partial(itrn+self.n_int_block-1,itrn+self.n_int_block+1)

        self.update_de_history(itrn) #update de history array

        self.update_fishers(itrn) #do fisher updates as necessary


        #update acceptance rate
        with np.errstate(invalid='ignore'):
            self.acc_fraction = self.a_yes/(self.a_no+self.a_yes)

        if self.itri == 0:
            self.tf1_loop = perf_counter()

        #update best ever found likelihood
        self.best_logL = max(self.best_logL,np.max(self.log_likelihood[self.chain_params.Ts==1.,:]))
        best_idx = np.unravel_index(np.argmax(self.log_likelihood),self.log_likelihood.shape)
        best_logL_global_loc = self.log_likelihood[best_idx]
        if best_logL_global_loc>self.best_logL_global:
            print("new best global sample old logL=%+12.3f new logL=%+12.3f"%(self.best_logL_global,best_logL_global_loc))
            self.best_logL_global = best_logL_global_loc
            self.best_sample_global = self.samples[best_idx].copy()
            print("local index of global best is",best_idx)
            print("new best params",self.best_sample_global)

        self.itri += 1

        if self.chain_params.log_mean_likelihoods:
            self.mean_likelihoods.append(self.log_likelihood[:,itrb:itrb+self.n_int_block].mean(axis=1))
            self.max_likelihoods.append(self.log_likelihood[:,itrb:itrb+self.n_int_block].max(axis=1))
            if itrb==0:
                print("mean likelihoods",self.mean_likelihoods[-1])
                print("best likelihood anywhere in latest block",self.max_likelihoods[-1].max())

        #check that there are no large decreases in log likelihood
        #if itrn>self.chain_params.save_every_n and np.any(np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)<-300.):
        if itrn>self.chain_params.save_every_n and np.any(np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)[:,::2].T<-300.*self.chain_params.Ts):
            print(np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)[:,::2].T)
            print(np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)[:,::2].T.shape)
            print(np.min(np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)[:,::2].T))
            print(np.where( np.diff(self.log_likelihood[:,:itrb+self.n_int_block],axis=1)[:,::2].T<-300.*self.chain_params.Ts ))
            assert False

    def update_de_history(self,itrn):
        """update de history array"""
        for j in range(self.n_chain):
            n_de_update= self.n_int_block//self.chain_params.thin_de
            for itrd in range(0,n_de_update):
                itrbd = itrn%self.chain_params.save_every_n+itrd*self.chain_params.thin_de
                assert not np.all(self.samples[j,itrbd,:]==0.)
                self.de_history[j,(self.itri*n_de_update+itrd)%self.chain_params.de_history_size] = self.samples[j,itrbd,:]

    def update_fishers_partial(self,itrn1,itrn2):
        """handle fisher matrix update logic, itrn1 and itrn2 must be ranges over which no changes in the total set of intrinsic parameters occur
        so that we can skip the intrinsic update"""
        #put each new fisher matrix into action as soon as possible after it is available, also breaks up any special behavior at common reset points
        #note that the random shuffling in update_fishers also makes it so that there is a finite probability any particular fisher matrix is kept for more than 1 block
        itrb1 = itrn1%self.chain_params.save_every_n
        itrb2 = itrb1-1+(itrn2-itrn1)
        assert itrb2<self.samples.shape[1]

        self.validate_consistent(itrb2)

        for eig_sel in range(0,2):
            for j in range(self.n_chain):
                if eig_sel:
                    idx_loc = self.idx_fisher_sel_next_eig[j]
                else:
                    idx_loc = self.idx_fisher_sel_next[j]
                idx_loc_mod = idx_loc%self.chain_params.save_every_n

                #need to handle the specific case where the chosen sample is the very last one in a block to be saved
                #because the samples array actually has size save_every_n+1, not save_every_n
                #alternatively, could just let the next loop around handle this case, but this gets the new fisher 1 block sooner
                if itrn1!=idx_loc and idx_loc_mod==0:
                    idx_loc_mod = self.chain_params.save_every_n

                if itrn1 <= idx_loc < itrn2:
                    #find the posterior sample we need to calculate the fisher matrices at
                    sample_need = self.samples[0:1,idx_loc_mod:idx_loc_mod+1,:]
                    #find the FLI with the right intrinsic parameters
                    found_match = False
                    for j2 in range(self.n_chain):
                        sample_loc =  self.samples[j2:j2+1,itrb2:itrb2+1,:]
                        if itrb2<self.samples.shape[1]-1:
                            assert np.all(self.samples[j2:j2+1,itrb2+1:itrb2+2,:]==0.)

                        if np.all(sample_need[:,:,self.x0_swap.idx_int] == sample_loc[:,:,self.x0_swap.idx_int]):
                            #print("start found",j,j2,itrn1,itrn2,idx_loc,idx_end_mod,eig_sel)
                            #print(sample_loc[0,0,self.x0_swap.idx_cw_int])
                            found_match = True
                            if eig_sel:
                                #the intrinsic and rn all match, so we should be able to evaluate the fisher diagonals from this FLI without a full intrinsic update on FLI_swap
                                self.x0_swap.update_params(sample_need[0,0,:])
                                eig_rn_loc,fisher_diag_loc,eig_common_loc = get_fishers(sample_need,self.par_names, self.x0_swap, self.flm, self.FLIs[j2],\
                                                                                        get_diag=True,get_common=True,get_rn_block=True,get_intrinsic_diag=True,start_safe=True)
                                self.eig_rn[j] = eig_rn_loc[0]
                                self.eig_common[j] = eig_common_loc[0]
                                self.fisher_diag[j] = fisher_diag_loc[0]
                                continue
                            else:
                                #the intrinsic and rn all match, so we should be able to evaluate the fisher diagonals from this FLI without a full intrinsic update on FLI_swap
                                self.x0_swap.update_params(sample_need[0,0,:])
                                _,fisher_diag_loc,_ = get_fishers(sample_need, self.par_names, self.x0_swap, self.flm, self.FLIs[j2],\
                                                                  get_diag=True,get_common=False,get_rn_block=False,get_intrinsic_diag=False,start_safe=True)
                                #assign the extrinsic parameters to the next diagonal
                                self.fisher_diag[j,self.x0_swap.idx_cw_ext] = fisher_diag_loc[0,self.x0_swap.idx_cw_ext]
                                continue
                    assert found_match
        self.validate_consistent(itrb2)


    def validate_consistent(self,itrb,full_validate=False):
        for j3 in range(self.n_chain):
            #print('j2',j3)
            #print(self.FLIs[j3].get_lnlikelihood(self.x0s[j3]),self.log_likelihood[j3,itrb])
            self.FLIs[j3].validate_consistent(self.x0s[j3])
            self.x0s[j3].validate_consistent(self.samples[j3,itrb])
            assert self.FLIs[j3].get_lnlikelihood(self.x0s[j3]) == self.log_likelihood[j3,itrb]
            if full_validate:
                self.x0_swap.update_params(self.samples[j3,itrb])
                self.flm.recompute_FastLike(self.FLI_swap,self.x0_swap,dict(zip(self.par_names, self.samples[j3,itrb])))
                self.FLI_swap.validate_consistent(self.x0s[j3])
                self.FLI_swap.validate_consistent(self.x0_swap)
                self.x0_swap.validate_consistent(self.samples[j3,itrb])
                assert self.FLI_swap.logdet_base == self.FLIs[j3].logdet_base
                assert self.FLI_swap.logdet == self.FLIs[j3].logdet
                assert self.FLI_swap.resres == self.FLIs[j3].resres
                assert np.all(self.FLI_swap.logdet_array == self.FLIs[j3].logdet_array)
                assert np.all(self.FLI_swap.resres_array == self.FLIs[j3].resres_array)
                assert np.all(self.FLI_swap.MMs == self.FLIs[j3].MMs)
                assert np.all(self.FLI_swap.NN == self.FLIs[j3].NN)
                for itrp in range(self.Npsr):
                    assert np.all(self.FLI_swap.chol_Sigmas[itrp] == self.FLIs[j3].chol_Sigmas[itrp])


    def update_fishers(self,itrn):
        """handle fisher matrix update logic"""
        #choose which parameter indexes to get for future fisher matrix evaluation points
        #the actual fisher update logic is done as soon as possible in update_fishers_partial
        for j in range(self.n_chain):
            if itrn <= self.idx_fisher_sel_next[j] < itrn+self.n_int_block:
                self.samples_sel_next[j,0,:] = self.samples[0,self.idx_fisher_sel_next[j]%self.chain_params.save_every_n,:]

            if itrn <= self.idx_fisher_sel_next_eig[j] < itrn+self.n_int_block:
                self.samples_sel_next_eig[j,0,:] = self.samples[0,self.idx_fisher_sel_next_eig[j]%self.chain_params.save_every_n,:]


        if itrn%self.n_update_fisher==0 and self.itri!=0:
            #prevent a terrible red noise fisher matrix from messing up one temperature for a very long time
            #by randomly shuffling the fisher matrices between chains whenever we do diagonal updates
            shuffle_idx = np.random.permutation(np.arange(0,self.n_chain))
            self.eig_rn[:] = self.eig_rn[shuffle_idx]
            self.eig_common[:] = self.eig_common[shuffle_idx]
            self.fisher_diag[:] = self.fisher_diag[shuffle_idx]

            #compute fisher matrix at random recent points in the posterior
            for j in range(self.n_chain):
                #eigenvalue update takes precedence over diagonal update parameters as it happens less frequently
                if itrn%self.chain_params.n_update_fisher_eig == 0:
                    assert not np.all(self.samples_sel_next_eig[j,0,:]==0.)
                    self.samples_sel[j] = self.samples_sel_next_eig[j,0,:]
                    #choose the next point to select
                    self.idx_fisher_sel_next_eig[j] = itrn+self.n_int_block+np.random.randint(0,self.chain_params.n_update_fisher_eig)
                    self.samples_sel_next_eig[j] = 0.
                else:
                    assert not np.all(self.samples_sel_next[j,0,:]==0.)
                    #load in the previous selected point
                    self.samples_sel[j,0,:] = self.samples_sel_next[j,0,:]

                #choose the next point to select
                self.idx_fisher_sel_next[j] = itrn+self.n_int_block+np.random.randint(0,self.n_update_fisher)
                self.samples_sel_next[j,0,:] = 0.

            if self.chain_params.log_fishers:
                #optionally log the old fisher matrices for diagnostic purposes, they dont really take much memory
                if itrn%self.chain_params.n_update_fisher_eig == 0:
                    self.fisher_diag_logs.append(self.fisher_diag.copy())
                    self.fisher_eig_logs.append(self.eig_rn.copy())
                    self.fisher_common_logs.append(self.eig_common.copy())
                else:
                    #update only the extrinsic fisher diagonals in this case
                    self.fisher_diag_logs.append(self.fisher_diag.copy())

        itrb = itrn%self.chain_params.save_every_n+self.n_int_block
        self.validate_consistent(itrb)  # check FLIs and x0s appear to have internally consistent parameters


    def do_status_update(self,itrn,N_blocks):
        """print a status update"""
        t_itr = perf_counter()
        print_acceptance_progress(itrn,N_blocks*self.n_int_block,self.n_int_block,self.a_yes,self.a_no,t_itr,self.ti_loop,self.tf1_loop,self.chain_params.Ts, self.verbosity)
        if len(self.mean_likelihoods)>=1:
            mean_likelihood = self.mean_likelihoods[-1][self.chain_params.Ts==1.].mean()
        else:
            mean_likelihood = self.FLIs[0].get_lnlikelihood(self.x0s[0])
        print("New log_L=%+12.3f Mean T=1 last block=%+12.3f Best T=1 log_L=%+12.3f best overall log_L=%+12.3f"%(self.FLIs[0].get_lnlikelihood(self.x0s[0]),mean_likelihood,self.best_logL,self.best_logL_global))#,FLIs[0].resres,FLIs[0].logdet,FLIs[0].pos,FLIs[0].pdist,FLIs[0].NN,FLIs[0].MMs)))
        #itrb = itrn%self.chain_params.save_every_n #index within the block of saved values
        #print(itrb)
        #print(self.samples[0,itrb,:])
        #print("Old log_L=%+12.3f"%(self.pta.get_lnlikelihood(self.samples[0,itrb,:])))

    def output_and_wrap_state(self,itrn,N_blocks,do_output=True):
        """wrap the samples around to the first element and save the old ones to the hdf5 file"""
        if do_output:
            output_hdf5_loop(itrn,self.chain_params,self.samples,self.log_likelihood,self.acc_fraction,self.fisher_diag,self.par_names,N_blocks*self.n_int_block,self.verbosity)

        #clear out log_likelihood and samples arrays
        samples_now = self.samples[:,-1,:]
        log_likelihood_now = self.log_likelihood[:,-1]
        self.samples = np.zeros((self.n_chain, self.chain_params.save_every_n+1, self.n_par_tot))
        self.log_likelihood = np.zeros((self.n_chain,self.chain_params.save_every_n+1))
        self.samples[:,0,:] = samples_now
        self.log_likelihood[:,0] = log_likelihood_now

    def advance_N_blocks(self,N_blocks):
        """advance the state of the MCMC system by N_blocks of size"""
        assert self.chain_params.save_first_n_chains <= self.n_chain #or we would try to save more chains than we have

        t1 = perf_counter()
        print("Entering Loop Body at %8.3fs"%(t1-self.ti))
        for i in range(N_blocks):
            #itrn = i*self.n_int_block #index overall
            itrn = self.itri*self.n_int_block #index overall
            itrb = itrn%self.chain_params.save_every_n #index within the block of saved values
            if itrb==0 and self.itri!=0:
                self.output_and_wrap_state(itrn,N_blocks)
            if self.itri%self.chain_params.n_block_status_update==0:
                self.do_status_update(itrn,N_blocks)


            #advance the block state
            self.advance_block()

        self.do_status_update(self.itri*self.n_int_block,N_blocks)

        output_hdf5_end(self.chain_params,self.samples,self.log_likelihood,self.acc_fraction,self.fisher_diag,self.par_names,self.verbosity)
        tf = perf_counter()
        print('whole function time = %8.3f s'%(tf-self.ti))
        print('loop time = %8.3f s'%(tf-self.ti_loop))


