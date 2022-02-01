"""store mcmc constants for global reference"""
tref = 53000*86400

#fisher size parameters
eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3,'red_noise_gamma':1.e-4,'red_noise_log10_A':1.e-4}
use_default_cw0_p_sigma = False
sigma_cw0_p_phase_default = 0.5
sigma_cw0_p_dist_default = 0.5

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
de_prob = 0.3
fisher_prob = 0.6

#jump parameter set probabilities
dist_jump_weight = 0.2
rn_jump_weight = 0.6
common_jump_weight = 0.2

#differencial evolution parameters
de_history_size = 1_000
sigma_de = 0.1

#multiple try MCMC parameters
n_multi_try = 3_000
