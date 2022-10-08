"""store mcmc constants for global reference"""
from numba import config
import numpy as np

tref = 53000*86400

#fisher size parameters
eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3,'red_noise_gamma':1.e-4,'red_noise_log10_A':1.e-4,'gwb_gamma':1.e-2,'gwb_log10_A':1.e-2}
#eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3,'red_noise_gamma':1.e-3,'red_noise_log10_A':1.e-3}
use_default_cw0_p_sigma = False
sigma_cw0_p_phase_default = 2.
sigma_cw0_p_dist_default = 2. #default is big here because informative priors will dominate anyway if constraints are poor
#sigma_log10_fgw_default = 0.005
sigma_log10_fgw_default = 0.01
sigma_log10_h_default = 0.1

eps_rn_diag_gamma_small_mult = 1.
eps_rn_diag_log10_A_small_mult = 1.
eps_rn_offdiag_small_mult = 1.
#eps_rn_diag_gamma_small_mult = 40.
#eps_rn_diag_log10_A_small_mult = 30.
#eps_rn_offdiag_small_mult = 30.
eps_log10_A_small_cut = -18.
eps_rn_offdiag = 1.e-3

use_default_noise_sigma = False
sigma_noise_default = 0.5

use_default_gwb_sigma = True
sigma_gwb_default = 0.1
#sigma_gwb_default = 0.2

###jump parameters to control number of eigendirections
#n_ext_directions = 32
#n_phase_extra = 16
#
#n_dist_extra = 67
#n_dist_main = 67
#
#do_common_def_switch = True#whether to redefine what 'common' parameters above a certain temperature
#common_def_switch_temp = 1.1
#
#do_gwb_common_override_switch = True
#gwb_common_override_switch_temp = 20.
#
#above this temperature, assume all phases have saturated for fisher jump sizing in projection parameters
#proj_phase_saturate_temp = 0.5
proj_phase_saturate_temp = 0.1
#above this temperature only do prior proposals for the projection parameters
proj_prior_all_temp = 1.1
#proj_prior_all_temp = 0.1
#proj_prior_all_temp = 0.55
#
#if do_gwb_common_override_switch:
#    #the gwb override only works if the common parameters have already switched to 6d at the specified temperature
#    assert do_common_def_switch
#    assert gwb_common_override_switch_temp>=common_def_switch_temp
#
#do_dist_prior_switch = True
#dist_prior_switch_temp = 1.3
#
#do_dist_all_override_switch = False
#dist_all_override_switch_temp = 1.5
#
#n_noise_emp_dist = 20#5#3#1#30#67#1

#indexes of summary variables
idx_PT = -2
idx_full = -1

#differential evolution parameters
sigma_de = 0.1

#multiple try MCMC parameters
n_x0_extra = config.NUMBA_NUM_THREADS
n_multi_try = 2000#3_000
#n_multi_try = 1

if n_multi_try < n_x0_extra:
    print("Reset n_x0_extra from "+str(n_x0_extra)+" to "+str(n_multi_try)+" in order to not exceed n_multi_try")
    n_x0_extra = n_multi_try

n_block_try = np.int64(n_multi_try//n_x0_extra)
if n_multi_try%n_x0_extra!=0:
    n_multi_try_old = n_multi_try
    n_block_try = n_block_try+1
    n_multi_try = n_block_try*n_x0_extra
    print("adjusted number multiple tries from "+str(n_multi_try_old)+" to be next even divisor of n_x0_extra="+str(n_x0_extra)+", "+str(n_multi_try))

assert n_multi_try%n_x0_extra == 0
    
