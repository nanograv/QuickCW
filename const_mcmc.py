"""store mcmc constants for global reference"""
from numba import config
import numpy as np

tref = 53000*86400

#fisher size parameters
eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3,'red_noise_gamma':1.e-4,'red_noise_log10_A':1.e-4}
#eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3,'red_noise_gamma':1.e-3,'red_noise_log10_A':1.e-3}
use_default_cw0_p_sigma = False
sigma_cw0_p_phase_default = 0.5
sigma_cw0_p_dist_default = 0.5
sigma_log10_fgw_default = 0.005
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

#jump parameters to control number of eigendirections
n_ext_directions = 32
n_phase_extra = 16

n_dist_extra = 4
n_dist_main = 16

n_noise_main = 10

#indexes of summary variables
idx_PT = -2
idx_full = -1

#jump type probabilities
prior_draw_prob = 0.1
de_prob = 0.6
fisher_prob = 0.3
#prior_draw_prob = 1.
#de_prob = 0.
#fisher_prob = 0.

#jump parameter set probabilities
dist_jump_weight = 0.2
rn_jump_weight = 0.4
common_jump_weight = 0.2
all_jump_weight = 0.2
#dist_jump_weight = 0.
#rn_jump_weight = 0.
#common_jump_weight = 1.
#all_jump_weight = 0.

#differential evolution parameters
sigma_de = 0.1

#multiple try MCMC parameters
n_x0_extra = config.NUMBA_NUM_THREADS
n_multi_try = 2000#3_000

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
    
