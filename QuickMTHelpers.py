"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)"""
import numpy as np

from numba import njit,prange
from numpy.random import uniform

import CWFastPrior
import const_mcmc as cm
from QuickCorrectionUtils import check_merged,correct_intrinsic,correct_extrinsic_array
from QuickFisherHelpers import safe_reset_swap,get_FLI_mem


################################################################################
#
#UPDATE INTRINSIC PARAMETERS AND RECALCULATE FILTERS
#
################################################################################
#version using multiple try mcmc (based on Table 6 of https://vixra.org/pdf/1712.0244v3.pdf)
#@profile
def do_intrinsic_update_mt(mcc, itrb):
    """do the intrinsic update using the multiple try mcmc algorithm"""
    Npsr = mcc.x0s[0].Npsr
    Ts = mcc.chain_params.Ts
    for j in range(mcc.n_chain):
        assert mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]) == mcc.log_likelihood[j,itrb]
        #print('k',j,mcc.log_likelihood[j,itrb])

    #print("EXT")
    for j in range(mcc.n_chain):
        #print(mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]),mcc.log_likelihood[j,itrb])
        assert mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]) == mcc.log_likelihood[j,itrb]
        mcc.FLIs[j].validate_consistent(mcc.x0s[j])

        #save MMs and NN so we can revert them if the chain is rejected
        FLI_mem_save = get_FLI_mem(mcc.FLIs[j])

        samples_current = np.copy(mcc.samples[j,itrb,:])
        #print('0',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]),mcc.log_likelihood[j,itrb])

        #should already be at this value
        mcc.x0s[j].validate_consistent(samples_current)
        mcc.x0s[j].update_params(samples_current)

        total_weight = (mcc.chain_params.dist_jump_weight + mcc.chain_params.rn_jump_weight + mcc.chain_params.gwb_jump_weight +
                        mcc.chain_params.common_jump_weight + mcc.chain_params.all_jump_weight)
        which_jump = np.random.choice(5, p=[mcc.chain_params.dist_jump_weight/total_weight,
                                            mcc.chain_params.rn_jump_weight/total_weight,
                                            mcc.chain_params.gwb_jump_weight/total_weight,
                                            mcc.chain_params.common_jump_weight/total_weight,
                                            mcc.chain_params.all_jump_weight/total_weight])

        #replace checking which_jump==1 etc with indicator values for desired behavior so that more jump types can be added in the future
        recompute_rn = False
        recompute_gwb = False
        recompute_int = False
        recompute_dist = False
        all_eigs = False

        if which_jump==0:  # update psr distances
            recompute_dist = True
            n_jump_loc = cm.n_dist_main
            idx_choose_psr = np.random.choice(Npsr,cm.n_dist_main,replace=False)
            #idx_choose_psr[0] = pta.pulsars.index(par_names_cw_int[jump_select][:-11])
            idx_choose = mcc.x0s[j].idx_dists[idx_choose_psr]
            scaling = 2.38*np.sqrt(Ts[j])/np.sqrt(n_jump_loc)
            #scaling = 1.0
            #scaling = 0.5
        elif which_jump==1:  # update per psr RN
            #n_jump_loc = cm.n_noise_main*2 #2 parameters per pulsar
            recompute_rn = True
            n_jump_loc = 2*Npsr
            #idx_choose_psr = np.random.randint(0,mcc.x0s[j].Npsr,cm.n_noise_main)
            idx_choose_psr = list(range(Npsr))
            idx_choose = np.concatenate((mcc.x0s[j].idx_rn_gammas,mcc.x0s[j].idx_rn_log10_As))
            scaling = 2.38*np.sqrt(Ts[j])/np.sqrt(n_jump_loc/2)
            #scaling = 1/np.sqrt(n_jump_loc)
        elif which_jump==2:  # update common RN
            recompute_gwb = True
            n_jump_loc = 2
            idx_choose = np.array([mcc.x0s[j].idx_gwb_gamma, mcc.x0s[j].idx_gwb_log10_A])
            scaling = 2.38*np.sqrt(Ts[j])/np.sqrt(n_jump_loc/2)
            #scaling = 1/np.sqrt(n_jump_loc)
        elif which_jump==3:  # update common intrinsic parameters (chirp mass, frequency, sky location[2])
            recompute_int = True
            n_jump_loc = 4 #  2+cm.n_dist_extra
            idx_choose = mcc.x0s[j].idx_cw_int[:4]  # np.array([par_names.index(par_names_cw_int[itrk]) for itrk in range(4)])
            scaling = 2.38*np.sqrt(Ts[j])/np.sqrt(n_jump_loc)
            #scaling = 1.0
            #scaling = 0.5
        elif which_jump==4: # do every possible jump
            #including this ensures any point in parameter space has some finite probability density to be reached in a single jump
            recompute_rn = True
            recompute_gwb = True
            recompute_int = True
            recompute_dist = True
            all_eigs = True
            n_jump_loc = cm.n_dist_main+2*Npsr+4+2 #distance+RN+common_pars+crn
            idx_choose_psr = list(range(Npsr))
            idx_choose = np.concatenate((mcc.x0s[j].idx_cw_int[:4],
                                         mcc.x0s[j].idx_dists[idx_choose_psr],
                                         mcc.x0s[j].idx_rn_gammas, mcc.x0s[j].idx_rn_log10_As,
                                         [mcc.x0s[j].idx_gwb_gamma, mcc.x0s[j].idx_gwb_log10_A]))
            scaling = 2.38*np.sqrt(Ts[j])/np.sqrt(n_jump_loc)
        else:
            raise ValueError('jump index unrecognized',which_jump)

        #decide what kind of jump we do
        if recompute_rn or recompute_gwb:  # RN or GWB jump --> don't do prior draws, only fisher and DE
            total_type_weight = mcc.chain_params.de_prob + mcc.chain_params.fisher_prob
            which_jump_type = np.random.choice(3, p=[0.,
                                                 mcc.chain_params.de_prob/total_type_weight,
                                                 mcc.chain_params.fisher_prob/total_type_weight])
        else:
            if j==(mcc.n_chain-1):  # hottest chain and not RN --> only do prior draws
                which_jump_type = 0
            else:  # not hottest chain and not RN --> choose jump type based in default probabilities of them
                total_type_weight = mcc.chain_params.prior_draw_prob + mcc.chain_params.de_prob + mcc.chain_params.fisher_prob
                which_jump_type = np.random.choice(3, p=[mcc.chain_params.prior_draw_prob/total_type_weight,
                                                         mcc.chain_params.de_prob/total_type_weight,
                                                         mcc.chain_params.fisher_prob/total_type_weight])
        if which_jump_type==0:  # do prior draw
            new_point = CWFastPrior.get_sample_idxs(samples_current.copy(),idx_choose,mcc.FPI)

            log_prior_old = CWFastPrior.get_lnprior(samples_current, mcc.FPI)
            log_prior_new = CWFastPrior.get_lnprior(new_point, mcc.FPI)
            #backwards/forwards proposal ratio not necessarily 1 (e.g. for distances with non-flat priors)
            log_proposal_ratio = log_prior_old - log_prior_new
        elif which_jump_type==1:  # do differential evolution step
            de_indices = np.random.choice(mcc.de_history.shape[1], size=2, replace=False)
            ndim = idx_choose.size
            #alpha0 = 2.38/np.sqrt(2*ndim)
            alpha0 = 1.68/np.sqrt(ndim)*np.sqrt(Ts[j])
            alpha = alpha0*np.random.normal(0.,1.)

            x1 = np.copy(mcc.de_history[j,de_indices[0],idx_choose])
            x2 = np.copy(mcc.de_history[j,de_indices[1],idx_choose])

            new_point = np.copy(samples_current)
            #new_point[idx_choose] += alpha0*(x1-x2)
            #new_point[idx_choose] += alpha0*(1+alpha)*(x1-x2)

            #backwards/forwards proposal ratio is always one for Gaussian jumps
            log_proposal_ratio = 0.0

            big_jump_decide = np.random.uniform(0.0, 1.0)
            if big_jump_decide<0.1: #do big jump
                #new_point[idx_choose] += (1+alpha)*(x1-x2)
                #TODO does this actually need to be scaled by a random amount?
                new_point[idx_choose] += (x1-x2)
            else: #do smaller jump scaled by alpha0
                #new_point[idx_choose] += alpha0*(1+alpha)*(x1-x2)
                new_point[idx_choose] += alpha*(x1-x2)
        elif which_jump_type==2:  # do regular fisher jump
            #jumps don't necessarily need to be mutually exclusive so use the indicator variables
            new_point = samples_current.copy()
            jump = np.zeros(mcc.n_par_tot)

            #backwards/forwards proposal ratio is always one for Gaussian jumps
            log_proposal_ratio = 0.0

            if recompute_rn:  # use RN eigenvectors
                scale_eig0 = scaling*mcc.eig_rn[j,:,0,:]
                scale_eig1 = scaling*mcc.eig_rn[j,:,1,:]
                new_point = add_rn_eig_jump(scale_eig0,scale_eig1,new_point,new_point[mcc.x0s[j].idx_rn],mcc.x0s[j].idx_rn,Npsr,all_eigs=all_eigs)

            if recompute_gwb: # use diagonal fishers
                idx_loc = np.array([mcc.x0s[j].idx_gwb_gamma, mcc.x0s[j].idx_gwb_log10_A])
                fisher_diag_loc = scaling * mcc.fisher_diag[j][idx_loc]
                jump[idx_loc] += fisher_diag_loc*np.random.normal(0.,1.,idx_loc.size)

            if recompute_int:  # use common parameter eigenvectors
                if all_eigs:
                    #allows attempting all of the eigenvalue jumps simultaneously
                    for itrp in range(0,4):
                        jump[mcc.x0s[j].idx_cw_int[:4]] += scaling*mcc.eig_common[j,itrp,:].flatten()*np.random.normal(0., 1.)
                else:
                    which_eig = np.random.choice(4, size=1)
                    jump[mcc.x0s[j].idx_cw_int[:4]] += scaling*mcc.eig_common[j,which_eig,:].flatten()*np.random.normal(0., 1.)

            if recompute_dist:  # use diagonal fishers
                #fisher_diag_loc = scaling * np.sqrt(Ts[j])*fisher_diag[j][idx_choose]
                idx_loc = mcc.x0s[j].idx_dists[idx_choose_psr]
                fisher_diag_loc = scaling * mcc.fisher_diag[j][idx_loc]
                jump[idx_loc] += fisher_diag_loc*np.random.normal(0.,1.,idx_loc.size)

            new_point = new_point + jump

        else:
            raise ValueError('jump type unrecognized',which_jump_type)

        #TODO check wrapping is working right
        new_point = correct_intrinsic(new_point,mcc.x0s[j],mcc.chain_params.freq_bounds,mcc.FPI.cut_par_ids, mcc.FPI.cut_lows, mcc.FPI.cut_highs)

        #more thorough jump types take precedence
        if recompute_rn or recompute_gwb:  # update per psr RN or GWB
            mcc.x0s[j].update_params(new_point)
            mcc.flm.recompute_FastLike(mcc.FLI_swap,mcc.x0s[j],dict(zip(mcc.par_names, new_point)))
            mcc.FLI_swap.validate_consistent(mcc.x0s[j])
        elif recompute_int:  # update common intrinsic parameters (chirp mass, frequency, sky location[2])
            mcc.x0s[j].update_params(new_point)
            mcc.FLIs[j].update_intrinsic_params(mcc.x0s[j])
            mcc.FLIs[j].validate_consistent(mcc.x0s[j])
        elif recompute_dist:  # update psr distances
            mcc.x0s[j].update_params(new_point)
            mcc.FLIs[j].update_pulsar_distances(mcc.x0s[j], idx_choose_psr)
            mcc.FLIs[j].validate_consistent(mcc.x0s[j])
        else:
            raise ValueError('no recompute type selected')

        #save current MM and NN
        FLI_mem_new = get_FLI_mem(mcc.FLIs[j])

        #check_not_merged(mcc.x0s[j].log10_fgw,mcc.x0s[j].log10_mc,FLIs[j].max_toa)
        #w0 = np.pi * 10.0**mcc.x0s[j].log10_fgw
        #mc = 10.0**mcc.x0s[j].log10_mc# * const.Tsun

        #check the maximum toa is not such that the source has already merged, and if so automatically reject the proposal
        if check_merged(mcc.x0s[j].log10_fgw,mcc.x0s[j].log10_mc,mcc.FLIs[j].max_toa):
            #set these so that the step is rejected
            #acc_ratio = -1
            #acc_decide = 0.
            log_acc_ratio = -np.inf
            log_acc_decide = 1.
            log_L_choose = -np.inf
            chosen_trial = -1
            print("Rejected due to too fast evolution.")
            mcc.x0s[j].update_params(samples_current)
            safe_reset_swap(mcc.FLIs[j],mcc.x0s[j],samples_current,FLI_mem_save)
            mcc.x0s[j].validate_consistent(samples_current)
            mcc.FLIs[j].validate_consistent(mcc.x0s[j])
        else:
            log_acc_ratio,chosen_trial,sample_choose,log_L_choose = do_mt_step(mcc,j,itrb,new_point,samples_current,FLI_mem_save,recompute_rn or recompute_gwb,log_proposal_ratio)
            if np.isfinite(log_acc_ratio):
                log_acc_decide = np.log(uniform(1.e-304, 1.0))
            else:
                log_acc_decide = 1.

        if log_acc_decide<=log_acc_ratio:
            #accepted
            mcc.x0s[j].update_params(sample_choose)

            mcc.samples[j,itrb+1,:] = sample_choose

            if recompute_rn or recompute_gwb:
                #swap the temporary FLI for the old one
                FLI_temp = mcc.FLIs[j]
                mcc.FLIs[j] = mcc.FLI_swap
                mcc.FLI_swap = FLI_temp
                mcc.FLIs[j].validate_consistent(mcc.x0s[j])
                mcc.x0s[j].validate_consistent(sample_choose)
                #print('3',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]))
            else:
                #since we reverted to old ones for calculating the reference point likelihoods, revert that
                safe_reset_swap(mcc.FLIs[j],mcc.x0s[j],sample_choose,FLI_mem_new)
                mcc.FLIs[j].validate_consistent(mcc.x0s[j])
                mcc.x0s[j].validate_consistent(sample_choose)
                #print('4',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]))

            mcc.log_likelihood[j,itrb+1] = log_L_choose
            if chosen_trial==0:
                mcc.a_yes[6*which_jump+2*which_jump_type,j] += 1
            else:
                mcc.a_no[6*which_jump+2*which_jump_type,j] += 1
            mcc.a_yes[6*which_jump+2*which_jump_type+1,j] += 1
            #print('1',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]))
        else:
            #rejected
            mcc.samples[j,itrb+1,:] = samples_current

            mcc.log_likelihood[j,itrb+1] = mcc.log_likelihood[j,itrb]

            #Add to both elements of a_no, so we can get acceptance over total jumps w/ and w/o projection perturbation
            if chosen_trial==0 and np.isfinite(log_acc_ratio):
                mcc.a_yes[6*which_jump+2*which_jump_type,j] += 1
            else:
                mcc.a_no[6*which_jump+2*which_jump_type,j] += 1
            mcc.a_no[6*which_jump+2*which_jump_type+1,j] += 1

            mcc.x0s[j].update_params(samples_current)

            #print('2',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]))
            if not recompute_rn and not recompute_gwb:
                #don't needs to do anything if which_jump==1 because we didn't update FLIs[j] at all,
                #and FLI_swap will just be completely overwritten next time it is used

                #revert the changes to FastLs
                safe_reset_swap(mcc.FLIs[j],mcc.x0s[j],samples_current,FLI_mem_save)

            #print('2',mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]))
        #print(which_jump)
        #print(mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]),mcc.log_likelihood[j,itrb+1])
        mcc.FLIs[j].validate_consistent(mcc.x0s[j])
        mcc.x0s[j].validate_consistent(mcc.samples[j,itrb+1,:])
        assert mcc.FLIs[j].get_lnlikelihood(mcc.x0s[j]) == mcc.log_likelihood[j,itrb+1]

    return mcc.FLI_swap


def do_mt_step(mcc,j,itrb,new_point,samples_current,FLI_mem_save,recompute_rn,log_proposal_ratio):
    """compute the multiple tries and chose a sample"""
    Ts = mcc.chain_params.Ts

    log_prior_old = CWFastPrior.get_lnprior(samples_current, mcc.FPI)
    log_posterior_old = mcc.log_likelihood[j,itrb]/Ts[j] + log_prior_old
    assert np.isfinite(log_posterior_old)

    #do multiple try MCMC step with random draws of projection parameters
    #more parameters will be uniform at higher temperatures
    fisher_mask = np.sqrt(Ts[j])*mcc.fisher_diag[j][mcc.x0s[0].idx_cw_ext]<0.5
    random_normals = np.random.normal(0.,1.,(cm.n_multi_try,fisher_mask.sum()))
    jumps = random_normals*np.sqrt(Ts[j])*mcc.fisher_diag[j][mcc.x0s[0].idx_cw_ext][fisher_mask]
    random_draws_from_prior = np.random.uniform(mcc.FPI.cw_ext_lows[~fisher_mask],mcc.FPI.cw_ext_highs[~fisher_mask],(cm.n_multi_try,(~fisher_mask).sum()))

    #itrd = 0
    #for ii,ll in enumerate(mcc.x0s[j].idx_cw_ext):
    #    fisher_loc = mcc.fisher_diag[j][ll]
    #    if fisher_loc<0.5: #not maxed out fisher --> do fisher update
    #        jumps_old[:,ii] = jumps[:,itrj]#np.random.normal(0.,fisher_loc,cm.n_multi_try)
    #        random_normals_old[:,ii] = random_normals[:,itrj]#jumps[:,itrj]/mcc.fisher_diag[j][mcc.x0s[0].idx_cw_ext][fisher_mask][itrj]
    #        itrj += 1
    #    else:
    #        random_draws_from_prior_old[:,ii] = random_draws_from_prior[:,itrd]#np.random.uniform(FPI.cw_ext_lows[ii],FPI.cw_ext_highs[ii],cm.n_multi_try)
    #        itrd += 1

    #make sure the jumps are null for the initial sample
    jumps[0,:] = 0.
    random_draws_from_prior[0,:] = new_point[mcc.x0_swap.idx_cw_ext][~fisher_mask]

    tries = set_params(new_point,jumps,fisher_mask,random_draws_from_prior,mcc.x0_swap)
    tries[0] = new_point  # just to make sure it didn't get reset
    log_prior_news = CWFastPrior.get_lnprior_array(tries, mcc.FPI)

    if recompute_rn:
        FLI_use = mcc.FLI_swap
    else:
        FLI_use = mcc.FLIs[j]
    mt_weights, log_Ls, log_mt_norm_shift = get_mt_weights(mcc.x0_extras, FLI_use, Ts[j],log_posterior_old,tries,log_prior_news)
    #if j==0: print(mt_weights)

    #not sure why but still can get nans here...
    assert np.all(np.isfinite(mt_weights))

    if np.sum(mt_weights)==0.0:
        log_acc_ratio = -np.inf
        chosen_trial = -1
        sample_choose = new_point.copy()
    else:
        chosen_trial = np.random.choice(cm.n_multi_try, p=mt_weights/np.sum(mt_weights))

        if not recompute_rn:  # need to set back FLIs to old state to calculate likelihoods at reference points
            safe_reset_swap(mcc.FLIs[j],mcc.x0s[j],samples_current,FLI_mem_save)
        else:
            mcc.x0s[j].update_params(samples_current)

        mcc.FLIs[j].validate_consistent(mcc.x0s[j])
        mcc.x0s[j].validate_consistent(samples_current)

        sample_ref = samples_current.copy()
        sample_ref[mcc.x0s[j].idx_cw_ext] = tries[chosen_trial,mcc.x0s[j].idx_cw_ext]

        ref_tries = set_params(sample_ref,jumps,fisher_mask,random_draws_from_prior,mcc.x0_swap)
        ref_tries[0] = sample_ref  # fix if it got reset

        log_prior_refs = CWFastPrior.get_lnprior_array(ref_tries, mcc.FPI)

        ref_mt_weights,log_ref_mt_norm_shift = get_ref_mt_weights(mcc.x0_extras, mcc.FLIs[j], Ts[j],log_posterior_old,chosen_trial,ref_tries,log_prior_refs)

        #must undo the normalization shifts; they aren't needed in log space anyway
        log_acc_ratio = np.log(np.sum(mt_weights))-np.log(np.sum(ref_mt_weights))+log_mt_norm_shift-log_ref_mt_norm_shift+log_proposal_ratio

        sample_choose = tries[chosen_trial].copy()
#        if chosen_trial==0:
#            print("selected trial was null?")
#            print(sample_ref)
#            print(sample_choose)
#            print(samples_current)
#            print(tries[1])
#            print(new_point)
#            print(log_ref_mt_norm_shift,log_mt_norm_shift)
#            print(ref_mt_weights)
#            print(mt_weights)
#            print(log_Ls)
#            print(log_prior_refs)
    return log_acc_ratio,chosen_trial,sample_choose,log_Ls[chosen_trial]

@njit(parallel=True)
def get_mt_weights(x0_extras, FLI_use, Ts, log_posterior_old,tries,log_prior_news):
    """Helper function to quickly return multiple tries and their likelihoods fo MTMCMC"""
    #NOTE isfinite does not work with fastmath enabled
    #set up needed arrays
    log_mt_weights = np.zeros(cm.n_multi_try)
    log_Ls = np.zeros(cm.n_multi_try)

    #get mt_weights --------------------------------------------------------------------------------------------------------
    for KK in prange(cm.n_x0_extra):
        for kk in range(cm.n_block_try):
            itrkk = KK*cm.n_block_try+kk

            x0_extras[KK].update_params(tries[itrkk,:])
            #print(x0_extras[KK].cos_gwtheta)

            log_L = FLI_use.get_lnlikelihood(x0_extras[KK])
            log_posterior_new = log_L/Ts + log_prior_news[itrkk]

            if np.isfinite(log_posterior_new):
                log_mt_weights[itrkk] = log_posterior_new - log_posterior_old
            else:
                log_mt_weights[itrkk] = -np.inf
            log_Ls[itrkk] = log_L

    #can apply the same multiplier to shift all the weights, prevents over/underflows in the exponential from breaking the code
    log_mt_norm_shift = np.max(log_mt_weights)
    log_mt_weights -= log_mt_norm_shift

    mt_weights = np.zeros(log_mt_weights.shape)
    #get weights while preventing underflow (values which are <1.e-304 times as likely to be chosen as the most likely value are totally irrelevant)
    mt_weights[log_mt_weights>-700] = np.exp(log_mt_weights[log_mt_weights>-700])

    return mt_weights, log_Ls, log_mt_norm_shift

@njit()
def add_rn_eig_jump(scale_eig0,scale_eig1,new_point,rn_base,idx_rn,Npsr,all_eigs=False):
    """add a fisher eigenvalue jump to the red noise parameters in place"""
    which_eig = np.random.choice(2, size=Npsr)
    jump_sizes = np.random.normal(0., 1.,Npsr)

    jump = np.zeros(2*Npsr)
    for ll in range(Npsr):
        if all_eigs or which_eig[ll] == 0:
            jump[ll] += scale_eig0[ll,0]*jump_sizes[ll]
            jump[ll+Npsr] += scale_eig0[ll,1]*jump_sizes[ll]
        if all_eigs or which_eig[ll] == 1:
            jump[ll] += scale_eig1[ll,0]*jump_sizes[ll]
            jump[ll+Npsr] += scale_eig1[ll,1]*jump_sizes[ll]

    new_point[idx_rn] = rn_base + jump
    return new_point

@njit()
def set_params(sample_set,jumps,fisher_mask,random_draws_from_prior,x0):
    """assign parameters to tries for multiple try mcmc"""
    ref_tries = np.zeros((cm.n_multi_try, sample_set.size))
    #jumps and random_draws_from_prior should give a null jump for the 0th value

    #copy in intrinsic parameters
    #ref_tries[:] = sample_set
    #ref_tries[:,x0.idx_cw_int] = sample_set[x0.idx_cw_int]
    #ref_tries[:,x0.idx_rn] = sample_set[x0.idx_rn]
    #ref_tries[:,x0.idx_gwb] = sample_set[x0.idx_gwb]
    ref_tries[:,x0.idx_int] = sample_set[x0.idx_int]

    ref_tries[:,x0.idx_cw_ext[fisher_mask]] = sample_set[x0.idx_cw_ext[fisher_mask]]+jumps
    ref_tries[:,x0.idx_cw_ext[~fisher_mask]] = random_draws_from_prior

    ref_tries = correct_extrinsic_array(ref_tries,x0)

    return ref_tries



@njit(parallel=True)
def get_ref_mt_weights(x0_extras, FLI_use, Ts, log_posterior_old, chosen_trial,ref_tries,log_prior_refs):
    """Helper function to quickly return multiple tries and their likelihoods fo MTMCMC"""
    #NOTE isfinite does not work with fastmath enabled
    #set up needed arrays
    log_ref_mt_weights = np.zeros(cm.n_multi_try)


    ##get ref_mt_weights ----------------------------------------------------------------------------------------------------
    for KK in prange(cm.n_x0_extra):
        for kk in range(cm.n_block_try):
            itrkk = KK*cm.n_block_try+kk

            x0_extras[KK].update_params(ref_tries[itrkk,:])

            log_L = FLI_use.get_lnlikelihood(x0_extras[KK])

            log_posterior_ref = log_L/Ts + log_prior_refs[itrkk]

            if np.isfinite(log_posterior_ref):
                log_ref_mt_weights[itrkk] = log_posterior_ref - log_posterior_old
            else:
                log_ref_mt_weights[itrkk] = -np.inf

    #can apply the same multiplier to shift all the weights, prevents over/underflows in the exponential from breaking the code
    log_ref_mt_weights[chosen_trial] = 0.  # np.log(1)=0. is the value it should be at the chosen trial pre-shift
    log_ref_mt_norm_shift = np.max(log_ref_mt_weights)
    log_ref_mt_weights -= log_ref_mt_norm_shift

    ref_mt_weights = np.zeros(log_ref_mt_weights.shape)
    #get weights while preventing underflow (values which are <1.e-304 times as likely to be chosen as the most likely value are totally irrelevant)
    ref_mt_weights[log_ref_mt_weights>-700] = np.exp(log_ref_mt_weights[log_ref_mt_weights>-700])

    return ref_mt_weights,log_ref_mt_norm_shift
