"""store mcmc constants for global reference"""
tref = 53000*86400

#fisher size parameters
eps = {'0_cos_gwtheta':1.e-4,'0_cos_inc':1.e-4,'0_gwphi':1.e-4,'0_log10_fgw':1.e-5,'0_log10_h':1.e-5,'0_log10_mc':1.e-4,'0_phase0':1.e-4,'0_psi':1.e-4,'cw0_p_phase':1.e-3,'cw0_p_dist':1.e-3}
use_default_cw0_p_sigma = False
sigma_cw0_p_phase_default = 0.1
sigma_cw0_p_dist_default = 0.5


#jump parameters to control number of eigendirections

n_ext_directions = 32
n_phase_extra = 16

n_dist_extra = 4
n_dist_main = 16

#indexes of summary variables
idx_PT = -2
idx_full = -1

