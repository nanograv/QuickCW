"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
Helpers to get fisher matrices"""
import numpy as np

import QuickCW.const_mcmc as cm

def get_FLI_mem(FLI_swap):
    """store everything needed to reset FLI for non-red noise updates

    :param FLI_swap:            FastLikelihoodInfo object

    :return MMs0:               list of M matrices
    :return NN0:                list of N vectors
    :return resres_array0:      array with resres values
    :return logdet_array0:      array with logdet contributions
    :return logdet_base_old:    base value of logdet
    """
    MMs0 = FLI_swap.MMs.copy()
    NN0 = FLI_swap.NN.copy()
    resres_array0 = FLI_swap.resres_array.copy()
    logdet_array0 = FLI_swap.logdet_array.copy()
    logdet_base_old = FLI_swap.logdet_base
    return (MMs0,NN0,resres_array0,logdet_array0,logdet_base_old)

def params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,idxs_targ,epsilons,dist_mode=False,phase_mode=False,mask=None):
    """helper to perturb the specified parameters by factors of epsilon en masse

    :param params:      array of parameter values
    :param x0_swap:     CWInfo object
    :param FLI_swap:    FastLikelihoodInfo object
    :param flm:         FastLikeMaster object
    :param par_names:   List of parameter names
    :param idxs_targ:   Indices of parameters to be perturbed
    :param epsilon:     Amount of perturbation
    :param dist_mode:   Specify if pulsar distances are being perturbed [False]
    :param phase_mode:  Specify if phases are being perturbed [False]
    :param mask:        Mask to specify which pulsars need to be updated; if None, all pulsars are updated [None]

    :return paramsPP:   Perturbed parameters
    :return FLI_memp:   FastLikeInfo object for the perturbed parameters
    """
    paramsPP = np.copy(params)
    paramsPP[idxs_targ] += epsilons

    FLI_mem0 = get_FLI_mem(FLI_swap)

    x0_swap.update_params(paramsPP)
    if dist_mode:
        #logdets and resres don't need to be updated for distances
        FLI_swap.update_pulsar_distances(x0_swap, np.arange(0,x0_swap.Npsr))
    elif phase_mode:
        #none of the matrices need to be updated for phases
        pass
    else:
        try:
            flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsPP)),mask=mask)
        except np.linalg.LinAlgError:
            print("failed to perturb parameters for fisher")
            print("params: ",paramsPP)
            #this will probably cause this particular fisher value to be invalid/not useful, but shouldn't be a huge issue in the long run
            safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)


    FLI_memp = get_FLI_mem(FLI_swap)

    safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)

    return paramsPP,FLI_memp#,paramsMM,MMsm,NNm,resres_arraym,logdet_arraym

#@njit()
def fisher_synthetic_FLI_helper(helper_tuple_RR,x0_swap,FLI_swap,params_old,dist_mode=False,phase_mode=False):
    """helper to construct synthetic likelihoods for each pulsar from input MMs and NNs one by one

    :param helper_tuple_RR: --
    :param x0_swap:         CWInfo object
    :param FLI_swap:        FastLikelihoodInfo object
    :param params_old:      Old parameters to set FLI back to once done
    :param dist_mode:       Specify if pulsar distances are being perturbed [False]
    :param phase_mode:      Specify if phases are being perturbed [False]

    :return rrs:            --

    """
    (paramsRR,FLI_memr) = helper_tuple_RR
    (MMsr,NNr,resres_arrayr,logdet_arrayr,_) = FLI_memr
    rrs = np.zeros(x0_swap.Npsr)
    #isolate elements that change for maximum numerical accuracy
    FLI_mem0 = get_FLI_mem(FLI_swap)

    FLI_swap.MMs[:] = 0.
    FLI_swap.NN[:] = 0.
    FLI_swap.resres_array[:] = 0.
    FLI_swap.logdet_array[:] = 0.
    FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

    x0_swap.update_params(paramsRR)

    for ii in range(x0_swap.Npsr):
        FLI_swap.MMs[ii] = MMsr[ii]
        FLI_swap.NN[ii] = NNr[ii]

        FLI_swap.resres_array[ii] = resres_arrayr[ii]
        FLI_swap.logdet_array[ii] = logdet_arrayr[ii]
        FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

        if dist_mode:
            #turn off all elements which do not vary with distance
            FLI_swap.MMs[:,:2,:2] = 0.
            FLI_swap.NN[:,:2] = 0.
            FLI_swap.resres_array[ii] = 0.
            FLI_swap.logdet_array[ii] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
        elif phase_mode:
            #turn off all elements which do not vary with phase
            FLI_swap.MMs[:,:2,:2] = 0.
            FLI_swap.NN[:,:2] = 0.
            FLI_swap.resres_array[ii] = 0.
            FLI_swap.logdet_array[ii] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

        #all other contributions are 0 by construction
        rrs[ii] = FLI_swap.get_lnlikelihood(x0_swap)

        #reset elements to 0
        FLI_swap.MMs[ii] = 0.
        FLI_swap.NN[ii] = 0.
        FLI_swap.resres_array[ii] = 0.
        FLI_swap.logdet_array[ii] = 0.
        FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

    safe_reset_swap(FLI_swap,x0_swap,params_old,FLI_mem0)

    return rrs#,fisher_diag



################################################################################
#
#CALCULATE RN FISHER EIGENVECTORS
#
################################################################################
def get_fishers(samples, par_names, x0_swap, flm, FLI_swap,get_diag=True,get_common=True,get_rn_block=True,get_intrinsic_diag=True,start_safe=False):
    """get all the red noise eigenvectors in a block, and if get_diag is True also get all the diagonal fisher matrix elements

    :param samples:             Array of samples
    :param par_names:           List of parameter names
    :param x0_swap:             CWInfo object
    :param flm:                 FastLikeMaster object
    :param FLI_swap:            FastLikeInfo object
    :param get_diag:            --
    :param get_common:          --
    :param get_rn_block:        --
    :param get_intrinsic_diag:  --
    :param start_safe:          --
    
    :return eig_rn:             Matrix with RN eigenvectors
    :return fisher_diag:        Diagonal fisher
    :return eig_common:         Matrix of common parameter fishers
    """
    #logdet_base is not needed for anything so turn it off, will revert later.
    n_chain = samples.shape[0]
    Npsr = x0_swap.Npsr
    eig_rn = np.zeros((n_chain,Npsr,2,2))
    fisher_diag = np.zeros((n_chain,len(par_names)))
    eig_common = np.zeros((n_chain,4,4))
    logdet_array_in = FLI_swap.logdet_array.copy()

    if start_safe:
        for itrc in range(n_chain):
            params = samples[itrc,0,:].copy()
            x0_swap.update_params(params)
            x0_swap.validate_consistent(params)
            FLI_swap.validate_consistent(x0_swap)

    for itrc in range(n_chain):
        params = samples[itrc,0,:].copy()
        x0_swap.update_params(params)

        if get_intrinsic_diag or get_diag or get_common:
            #as currently structure need diagonal data for get_fisher_eigenvectors_common
            fisher_diag[itrc,:],diagonal_data_loc = get_fisher_diagonal(params, par_names,  x0_swap, flm, FLI_swap,get_intrinsic_diag=get_intrinsic_diag,start_safe=start_safe)
            FLI_swap.validate_consistent(x0_swap)
            if start_safe:
                assert np.all(logdet_array_in==FLI_swap.logdet_array)
            #pp1s,mm1s,nn1s,epsilons,helper_tuple0,pps,mms,nns = diagonal_data_loc
            if get_intrinsic_diag or get_common:
                eig_common[itrc] = get_fisher_eigenvectors_common(params, x0_swap, FLI_swap, diagonal_data_loc,default_all=not get_common)
            FLI_swap.validate_consistent(x0_swap)
            if start_safe:
                assert np.all(logdet_array_in==FLI_swap.logdet_array)
        elif get_rn_block:
            #fisher_diag = np.zeros(len(par_names))
            epsilon_gammas = np.zeros(x0_swap.idx_rn_gammas.size)+2*cm.eps['red_noise_gamma']
            epsilon_log10_As = np.zeros(x0_swap.idx_rn_log10_As.size)+2*cm.eps['red_noise_log10_A']

            #adapt epsilon to be a bit bigger at low amplitude values
            epsilon_gammas[params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_diag_gamma_small_mult
            epsilon_log10_As[params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_diag_log10_A_small_mult

            #don't need gwb because it doesn't affect the eigenvectors
            pp1s,mm1s,nn1s,helper_tuple0,_,_,_ = fisher_rn_mm_pp_diagonal_helper(params,x0_swap,FLI_swap,flm,par_names,epsilon_gammas,epsilon_log10_As,Npsr,get_intrinsic_diag=True,start_safe=start_safe,get_gwb=False)
            FLI_swap.validate_consistent(x0_swap)
            if start_safe:
                assert np.all(logdet_array_in==FLI_swap.logdet_array)

            pps = np.zeros(len(par_names))
            mms = np.zeros(len(par_names))
            nns = np.zeros(len(par_names))
            epsilons = np.zeros(len(par_names))
            epsilons[x0_swap.idx_rn_gammas] = epsilon_gammas
            epsilons[x0_swap.idx_rn_log10_As] = epsilon_log10_As
            diagonal_data_loc = (pp1s,mm1s,nn1s,epsilons,helper_tuple0,pps,mms,nns)

        if get_rn_block:
            eig_rn[itrc] = get_fisher_rn_block_eigenvectors(params, par_names, x0_swap, flm, FLI_swap,diagonal_data_loc)
            FLI_swap.validate_consistent(x0_swap)
            if start_safe:
                assert np.all(logdet_array_in==FLI_swap.logdet_array)

            continue

        if start_safe:
            assert np.all(logdet_array_in==FLI_swap.logdet_array)


    FLI_swap.validate_consistent(x0_swap)

    if start_safe:
        for itrc in range(n_chain):
            params = samples[itrc,0,:].copy()
            x0_swap.update_params(params)
            x0_swap.validate_consistent(params)
            FLI_swap.validate_consistent(x0_swap)

    return eig_rn,fisher_diag,eig_common


def get_fisher_rn_block_eigenvectors(params, par_names, x0_swap, flm, FLI_swap,diagonal_data_loc):
    """get the diagonal elements of the fisher matrix with the needed local stabilizations

    :param params:              --
    :param par_names:           List of parameter names
    :param x0_swap:             CWInfo object
    :param flm:                 FastLikeMaster object
    :param FLI_swap:            FastLikeInfo object
    :param diagonal_data_loc:   --

    :return eig_rn:             Matrix with RN eigenvectors
    """
    Npsr = x0_swap.Npsr
    dim = 2
    #[0.8,0.25] is more reflective of center of distribution for poorly constrained rn parameters
    #but not really necessary to go out that far and it reduces acceptances
    sigma_noise_defaults = np.array([0.5,0.5])
    #eig_limit = 1./np.max(sigma_noise_defaults)**2#1.0#0.25
    small_cut_mult = 1.
    fisher_suppress = 0.9
    eig_limit = 4.

    fisher_prod_lim = 1./(sigma_noise_defaults[0]**2*sigma_noise_defaults[1]**2)
    fisher_cut_lims = small_cut_mult*1./sigma_noise_defaults**2

    determinant_cut = 0.25*fisher_prod_lim

    fisher = np.zeros((Npsr,dim,dim))
    pure = np.full(Npsr,True,dtype=np.bool_)
    badc = np.zeros(Npsr,dtype=np.int64)
    eig_rn = np.zeros((Npsr,2,2))#np.broadcast_to(np.eye(2)*0.5, (n_chain, Npsr, 2, 2) )#.copy()

    #future locations
    pp1s = np.zeros((Npsr,2))
    mm1s = np.zeros((Npsr,2))

    pp2s = np.zeros(Npsr)
    mm2s = np.zeros(Npsr)
    pm2s = np.zeros(Npsr)
    mp2s = np.zeros(Npsr)

    pp1s,mm1s,nn1s,epsilons,helper_tuple0,_,_,_ = diagonal_data_loc
    (_,FLI_mem0) = helper_tuple0

    chol_Sigmas_save = []
    for ii in range(Npsr):
        chol_Sigmas_save.append(FLI_swap.chol_Sigmas[ii].copy())

    epsilon_diags = np.zeros((Npsr,2))
    epsilon_diags[:,0] = epsilons[x0_swap.idx_rn_gammas]
    epsilon_diags[:,1] = epsilons[x0_swap.idx_rn_log10_As]


    for itrs in range(Npsr):
        #calculate off-diagonal elements of the Hessian from a central finite element scheme
        #note the minus sign compared to the regular Hessian
        fisher[itrs] = np.zeros((dim,dim))
        #calculate off-diagonal elements

        for itrp in range(dim):
            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            #factor of 4 in the denominator is absorbed because pp1s and mm1s use 2*epsilon steps
            fisher[itrs,itrp,itrp] = -(pp1s[itrs,itrp] - 2.0*nn1s[itrs] + mm1s[itrs,itrp])/(epsilon_diags[itrs,itrp]*epsilon_diags[itrs,itrp])+1./sigma_noise_defaults[itrp]**2
            #patch to handle bad values
            if ~np.isfinite(fisher[itrs,itrp,itrp]) or fisher[itrs,itrp,itrp] <= fisher_cut_lims[itrp]:
                fisher[itrs,itrp,itrp] = 1./sigma_noise_defaults[itrp]**2
                badc[itrs] += 1
                pure[itrs] = False

        #if one value is bad and the other is small, assume we both were actually bad and default the whole matrix
        if badc[itrs] and (fisher[itrs,0,0]<fisher_cut_lims[0] or fisher[itrs,1,1]<fisher_cut_lims[1]):
            fisher[itrs,0,0] = 1./sigma_noise_defaults[0]**2
            fisher[itrs,1,1] = 1./sigma_noise_defaults[1]**2

        #if both values are small, scale the diagonals up so the product at the minimum to increase the chances of the eigenvectors being usable while mostly preserving the structure
        if fisher[itrs,0,0]*fisher[itrs,1,1]<fisher_prod_lim:
            holdmax = fisher[itrs,0,0]*fisher[itrs,1,1]
            fisher[itrs] *= np.sqrt(fisher_prod_lim/holdmax)
            pure[itrs] = False

    #track ones that defaulted so we can skip evalauation of the off diagonal element completely
    #because many default this can save non-trivial amounts of time
    print("Number of Pulsars with Fisher Eigenvectors in Full Default: ",(badc==2).sum(),"Diagonal Default: ",(badc==1).sum(),"No Default: ",(badc==0).sum())
    #TODO temporary to test if we can recover better fishers
    defaulted = badc == 2
    #defaulted = badc == 2
    #defaulted[:] = False

    #the noise parameters are very expensive to calculate individually so calculate them all en masse
    #get the off diagonal elements of the fisher matrix

    idx_rns = np.hstack([x0_swap.idx_rn_gammas,x0_swap.idx_rn_log10_As])
    epsilon_offdiag = cm.eps_rn_offdiag
    epsilon_drns = np.zeros(idx_rns.size)+epsilon_offdiag
    epsilon_crns = np.zeros(idx_rns.size)-epsilon_offdiag #make idx_rn_log10_As negative and idx_rn_gammax positive
    epsilon_crns[:x0_swap.idx_rn_gammas.size] = epsilon_offdiag

    #adapt epsilon to be a bit bigger at low amplitude values
    epsilon_drns[0:x0_swap.idx_rn_gammas.size][params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_offdiag_small_mult
    epsilon_drns[x0_swap.idx_rn_gammas.size:][params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_offdiag_small_mult

    epsilon_crns[0:x0_swap.idx_rn_gammas.size][params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_offdiag_small_mult
    epsilon_crns[x0_swap.idx_rn_gammas.size:][params[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_offdiag_small_mult

    helper_tuple_drns_PP = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,idx_rns,epsilon_drns,mask=~defaulted)
    helper_tuple_drns_MM = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,idx_rns,-epsilon_drns,mask=~defaulted)

    helper_tuple_crns_PM = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,idx_rns,epsilon_crns,mask=~defaulted)
    helper_tuple_crns_MP = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,idx_rns,-epsilon_crns,mask=~defaulted)

    #the nns from the diagonal method should be derived safely as well as the PP, MM, PM, and MP here
    safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)

    pp2s[:] = fisher_synthetic_FLI_helper(helper_tuple_drns_PP,x0_swap,FLI_swap,params)
    mm2s[:] = fisher_synthetic_FLI_helper(helper_tuple_drns_MM,x0_swap,FLI_swap,params)

    pm2s[:] = fisher_synthetic_FLI_helper(helper_tuple_crns_PM,x0_swap,FLI_swap,params)
    mp2s[:] = fisher_synthetic_FLI_helper(helper_tuple_crns_MP,x0_swap,FLI_swap,params)

    #update FLI_swap and x0_swap to at least a self consistent state
    #copy back in chol_Sigmas for safety
    for ii in range(Npsr):
        FLI_swap.chol_Sigmas[ii][:] = chol_Sigmas_save[ii]

    safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)



    for itrs in range(Npsr):
        if defaulted[itrs]:
            fisher_offdiag = 0.
            pure[itrs] = False
        else:
            fisher_offdiag =  -(pp2s[itrs] - mp2s[itrs] - pm2s[itrs] + mm2s[itrs])/(4.0*epsilon_offdiag*epsilon_offdiag)
            if ~np.isfinite(fisher_offdiag):
                assert False
                fisher_offdiag = 0.

        #do not let the determinant be negative or 0 to ensure matrix is positive definite
        if (fisher[itrs,0,0]*fisher[itrs,1,1]-fisher_offdiag**2)<=determinant_cut:#1./cm.sigma_noise_default**4:
            pure[itrs] = False

            fisher_offdiag = np.sign(fisher_offdiag)*fisher_suppress*np.sqrt(np.abs(fisher[itrs,0,0]*fisher[itrs,1,1]-fisher_prod_lim))

        if ~np.isfinite(fisher_offdiag):
            assert False
            pure[itrs] = False
            fisher_offdiag = 0.

        fisher[itrs,0,1] = fisher_offdiag
        fisher[itrs,1,0] = fisher[itrs,0,1]

        #Filter nans and infs and replace them with 1s
        #this will imply that we will set the eigenvalue to 100 a few lines below
        FISHER = np.where(np.isfinite(fisher[itrs]), fisher[itrs], 1.0)
        if not np.array_equal(FISHER, fisher[itrs]):
            print("Changed some nan elements in the Fisher matrix to 1.0")

        #Find eigenvalues and eigenvectors of the Fisher matrix
        w, v = np.linalg.eigh(FISHER)

        #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
        W = np.where(np.abs(w)>eig_limit, w, eig_limit)

        rn_eigvec = (np.sqrt(1.0/np.abs(W))*v).T

        eig_rn[itrs,:,:] = rn_eigvec[:,:]
    return eig_rn


def fisher_rn_mm_pp_diagonal_helper(params,x0_swap,FLI_swap,flm,par_names,epsilon_gammas,epsilon_log10_As,Npsr,get_intrinsic_diag=True,start_safe=False,get_gwb=True):
    """helper to get the mm and pp values needed to calculate the diagonal fisher eigenvectors for the red noise parameters

    :param params:              --
    :param x0_swap:             CWInfo object
    :param FLI_swap:            FastLikeInfo object
    :param flm:                 FastLikeMaster object
    :param par_names:           List of parameter names
    :param epsilon_gammas:      Perturbation values for gamma parameters
    :param epsilon_log10_As:    Perturbation values for log10 amplitudes
    :param Npsr:                Number of pulsars
    :param get_intrinsic_diag:  --
    :param start_safe:          --
    :param get_gwb:             --    

    :return pp1s:               --
    :return mm1s:               --    
    :return nn1s:               --
    :return helper_tuple0:      --
    :return pps_gwb:            --
    :return mms_gwb:            --
    :return nns_gwb:            --
    """
    if get_intrinsic_diag:
        print("Calculating RN fisher Eigenvectors")
    #future locations
    pp1s = np.zeros((Npsr,2))
    mm1s = np.zeros((Npsr,2))

    nns_gwb = np.zeros(x0_swap.idx_gwb.size)
    pps_gwb = np.zeros(x0_swap.idx_gwb.size)
    mms_gwb = np.zeros(x0_swap.idx_gwb.size)

    x0_swap.update_params(params)

    #put the reset here to avoid having to do it both before and after
    if not start_safe:
        flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, params)))

    FLI_mem0 = get_FLI_mem(FLI_swap)

    helper_tuple0 = (params,FLI_mem0)
    nn1s = fisher_synthetic_FLI_helper(helper_tuple0,x0_swap,FLI_swap,params)

    if get_intrinsic_diag:
        #save chol_Sigmas so everything can be reset to full self consistency
        chol_Sigmas_save = []
        for ii in range(Npsr):
            chol_Sigmas_save.append(FLI_swap.chol_Sigmas[ii].copy())

        #the noise parameters are very expensive to calculate individually so calculate them all en masse
        helper_tuple_gammas_PP = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_rn_gammas,epsilon_gammas)
        helper_tuple_gammas_MM = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_rn_gammas,-epsilon_gammas)

        helper_tuple_log10_As_PP = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_rn_log10_As,epsilon_log10_As)
        helper_tuple_log10_As_MM = params_perturb_helper(params,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_rn_log10_As,-epsilon_log10_As)

        #epsilon = cm.eps['red_noise_gamma']
        pp1s[:,0] = fisher_synthetic_FLI_helper(helper_tuple_gammas_PP,x0_swap,FLI_swap,params)
        mm1s[:,0] = fisher_synthetic_FLI_helper(helper_tuple_gammas_MM,x0_swap,FLI_swap,params)

        #epsilon = cm.eps['red_noise_log10_A']
        pp1s[:,1] = fisher_synthetic_FLI_helper(helper_tuple_log10_As_PP,x0_swap,FLI_swap,params)
        mm1s[:,1] = fisher_synthetic_FLI_helper(helper_tuple_log10_As_MM,x0_swap,FLI_swap,params)

        if get_gwb:
            #double check everything is reset although it shouldn't actually be necessary here
            safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)

            #copy back in chol_Sigmas for safety
            for ii in range(Npsr):
                FLI_swap.chol_Sigmas[ii][:] = chol_Sigmas_save[ii]

            nns_gwb[:] = FLI_swap.get_lnlikelihood(x0_swap)

            #do the gwb parameters
            for itr,i in enumerate(x0_swap.idx_gwb):
                #gwb jump so update everything
                epsilon = cm.eps[par_names[i]]

                paramsPP = np.copy(params)
                paramsMM = np.copy(params)

                paramsPP[i] += 2*epsilon
                paramsMM[i] -= 2*epsilon


                #must be one of the intrinsic parameters
                x0_swap.update_params(paramsPP)

                flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsPP)),mask=None)
                pps_gwb[itr] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

                x0_swap.update_params(paramsMM)

                flm.recompute_FastLike(FLI_swap,x0_swap,dict(zip(par_names, paramsMM)),mask=None)
                mms_gwb[itr] = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

                safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)

                #copy back in chol_Sigmas for safety
                for ii in range(Npsr):
                    FLI_swap.chol_Sigmas[ii][:] = chol_Sigmas_save[ii]

            #copy back in chol_Sigmas for safety
            for ii in range(Npsr):
                FLI_swap.chol_Sigmas[ii][:] = chol_Sigmas_save[ii]

    #double check everything is reset although it shouldn't actually be necessary here
    safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)

    return pp1s,mm1s,nn1s,helper_tuple0,pps_gwb,mms_gwb,nns_gwb

def safe_reset_swap(FLI_swap,x0_swap,params_old,FLI_mem0):
    """safely reset everything back to the initial values as input for self consistency in future calculations

    :param FLI_swap:    FastLikeInfo object to be reset
    :param x0_swap:     CWInfo object
    :param params_old:  Parameters to which we want to reset
    :param FLI_mem0:    Parts of FLI object saved to memory
    """
    MMs0,NN0,resres_array0,logdet_array0,logdet_base_old = FLI_mem0
    x0_swap.update_params(params_old)

    FLI_swap.cos_gwtheta = x0_swap.cos_gwtheta
    FLI_swap.gwphi = x0_swap.gwphi
    FLI_swap.log10_fgw = x0_swap.log10_fgw
    FLI_swap.log10_mc = x0_swap.log10_mc
    FLI_swap.gwb_gamma = x0_swap.gwb_gamma
    FLI_swap.gwb_log10_A = x0_swap.gwb_log10_A
    FLI_swap.rn_gammas = x0_swap.rn_gammas.copy()
    FLI_swap.rn_log10_As = x0_swap.rn_log10_As.copy()
    FLI_swap.cw_p_dists = x0_swap.cw_p_dists.copy()

    FLI_swap.MMs[:] = MMs0
    FLI_swap.NN[:] = NN0

    FLI_swap.set_resres_logdet(resres_array0,logdet_array0,logdet_base_old)

    FLI_swap.validate_consistent(x0_swap)
    x0_swap.validate_consistent(params_old)

################################################################################
#
#CALCULATE FISHER DIAGONAL
#
################################################################################

def get_fisher_diagonal(samples_fisher, par_names, x0_swap, flm, FLI_swap,get_intrinsic_diag=True,start_safe=False):
    """get the diagonal elements of the fisher matrix for all parameters

    :param samples_fisher:          --
    :param par_names:               List of parameter names
    :param x0_swap:                 CWInfo object
    :param flm:                     FastLikeMaster object
    :param FLI_swap:                FastLikeInfo object
    :param get_intrinsic_diag:      --
    :param start_safe:              --

    :return 1/np.sqrt(fisher_diag): --
    :return pp2s:                   --
    :return mm2s:                   --
    :return nn2s:                   --
    :return epsilons:               --
    :return helper_tuple0:          --
    :return pps:                    --
    :return mms:                    --
    :return nns:                    --
    """
    #this print out occurs a bit excessively frequently
    #print("Updating Fisher Diagonals")
    dim = len(par_names)
    fisher_diag = np.zeros(dim)
    #TODO pass directly and fix elsewhere
    Npsr = x0_swap.Npsr#x0_swap.idx_rn_gammas.size

    x0_swap.update_params(samples_fisher)
    #we will update FLI_swap later to prevent having to do it twice

    #future locations
    mms = np.zeros(dim)
    pps = np.zeros(dim)
    nns = np.zeros(dim)
    epsilons = np.zeros(dim)

    sigma_defaults = np.full(dim,1.)
    sigma_defaults[x0_swap.idx_rn_gammas] = cm.sigma_noise_default
    sigma_defaults[x0_swap.idx_rn_log10_As] = cm.sigma_noise_default
    sigma_defaults[x0_swap.idx_dists] = cm.sigma_cw0_p_dist_default
    sigma_defaults[x0_swap.idx_phases] = cm.sigma_cw0_p_phase_default
    sigma_defaults[x0_swap.idx_log10_fgw] = cm.sigma_log10_fgw_default
    sigma_defaults[x0_swap.idx_log10_h] = cm.sigma_log10_h_default
    sigma_defaults[x0_swap.idx_gwb] = cm.sigma_gwb_default

    epsilon_gammas = np.zeros(x0_swap.idx_rn_gammas.size)+2*cm.eps['red_noise_gamma']
    epsilon_log10_As = np.zeros(x0_swap.idx_rn_log10_As.size)+2*cm.eps['red_noise_log10_A']

    #adapt epsilon to be a bit bigger at low amplitude values
    epsilon_gammas[samples_fisher[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_diag_gamma_small_mult
    epsilon_log10_As[samples_fisher[x0_swap.idx_rn_log10_As]<cm.eps_log10_A_small_cut] *= cm.eps_rn_diag_log10_A_small_mult
    epsilon_dists = np.zeros(Npsr)+2*cm.eps['cw0_p_dist']
    epsilons[x0_swap.idx_rn_gammas] = epsilon_gammas
    epsilons[x0_swap.idx_rn_log10_As] = epsilon_log10_As
    epsilons[x0_swap.idx_dists] = epsilon_dists
    epsilons[x0_swap.idx_gwb_gamma] = 2*cm.eps['gwb_gamma']
    epsilons[x0_swap.idx_gwb_log10_A] = 2*cm.eps['gwb_log10_A']

    pp2s,mm2s,nn2s,helper_tuple0,pps_gwb,mms_gwb,nns_gwb = fisher_rn_mm_pp_diagonal_helper(samples_fisher,x0_swap,FLI_swap,flm,\
                                                                   par_names,epsilon_gammas,epsilon_log10_As,Npsr,\
                                                                   get_intrinsic_diag=get_intrinsic_diag,start_safe=start_safe,\
                                                                   get_gwb=(get_intrinsic_diag and not cm.use_default_gwb_sigma))
    (_,FLI_mem0) = helper_tuple0

    if get_intrinsic_diag:
        pps[x0_swap.idx_rn_gammas] = pp2s[:,0]
        mms[x0_swap.idx_rn_gammas] = mm2s[:,0]
        nns[x0_swap.idx_rn_gammas] = nn2s

        pps[x0_swap.idx_rn_log10_As] = pp2s[:,1]
        mms[x0_swap.idx_rn_log10_As] = mm2s[:,1]
        nns[x0_swap.idx_rn_log10_As] = nn2s

        pps[x0_swap.idx_gwb] = pps_gwb[:]
        mms[x0_swap.idx_gwb] = mms_gwb[:]
        nns[x0_swap.idx_gwb] = nns_gwb[:]

        safe_reset_swap(FLI_swap,x0_swap,samples_fisher,FLI_mem0)

        helper_tuple_dists_PP = params_perturb_helper(samples_fisher,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_dists,epsilon_dists,dist_mode=False)
        helper_tuple_dists_MM = params_perturb_helper(samples_fisher,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_dists,-epsilon_dists,dist_mode=False)

        pps[x0_swap.idx_dists] = fisher_synthetic_FLI_helper(helper_tuple_dists_PP,x0_swap,FLI_swap,samples_fisher,dist_mode=True)
        mms[x0_swap.idx_dists] = fisher_synthetic_FLI_helper(helper_tuple_dists_MM,x0_swap,FLI_swap,samples_fisher,dist_mode=True)
        nns[x0_swap.idx_dists] = fisher_synthetic_FLI_helper(helper_tuple0,x0_swap,FLI_swap,samples_fisher,dist_mode=True)

    safe_reset_swap(FLI_swap,x0_swap,samples_fisher,FLI_mem0)

    epsilon_phases = np.zeros(Npsr)+2*cm.eps['cw0_p_phase']
    helper_tuple_phases_PP = params_perturb_helper(samples_fisher,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_phases,epsilon_phases,phase_mode=True)
    helper_tuple_phases_MM = params_perturb_helper(samples_fisher,x0_swap,FLI_swap,flm,par_names,x0_swap.idx_phases,-epsilon_phases,phase_mode=True)

    pps[x0_swap.idx_phases] = fisher_synthetic_FLI_helper(helper_tuple_phases_PP,x0_swap,FLI_swap,samples_fisher,phase_mode=True)
    mms[x0_swap.idx_phases] = fisher_synthetic_FLI_helper(helper_tuple_phases_MM,x0_swap,FLI_swap,samples_fisher,phase_mode=True)
    nns[x0_swap.idx_phases] = fisher_synthetic_FLI_helper(helper_tuple0,x0_swap,FLI_swap,samples_fisher,phase_mode=True)

    epsilons[x0_swap.idx_phases] = epsilon_phases

    assert np.all(fisher_diag>=0.)

    chol_Sigmas_save = []
    for ii in range(Npsr):
        chol_Sigmas_save.append(FLI_swap.chol_Sigmas[ii].copy())

    #calculate diagonal elements
    for i in range(dim):
        paramsPP = np.copy(samples_fisher)
        paramsMM = np.copy(samples_fisher)

        if i in x0_swap.idx_phases:#'_cw0_p_phase' in par_names[i]:
            if cm.use_default_cw0_p_sigma:
                fisher_diag[i] = 1/cm.sigma_cw0_p_phase_default**2
            #otherwise should already have been done

        elif i in x0_swap.idx_cw_ext:#par_names_cw_ext:
            epsilon = cm.eps[par_names[i]]
            epsilons[i] = 2*epsilon

            paramsPP[i] += 2*epsilon
            paramsMM[i] -= 2*epsilon

            FLI_swap.logdet_array[:] = 0.
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

            nns[i] = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            #use fast likelihood
            x0_swap.update_params(paramsPP)

            pps[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            x0_swap.update_params(paramsMM)

            mms[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            #revert changes
            safe_reset_swap(FLI_swap,x0_swap,samples_fisher,FLI_mem0)

        elif i in x0_swap.idx_dists:
            if cm.use_default_cw0_p_sigma or not get_intrinsic_diag:
                fisher_diag[i] = 1/cm.sigma_cw0_p_dist_default**2
            #should already have been done otherwise

        elif (i in x0_swap.idx_rn_gammas) or (i in x0_swap.idx_rn_log10_As):
            #continue
            if cm.use_default_noise_sigma or not get_intrinsic_diag:
                fisher_diag[i] = 1./cm.sigma_noise_default**2
            #already did all of the above otherwise
        elif i in x0_swap.idx_gwb:
            #default gwb indices if requested, otherwise we should already have them
            if cm.use_default_gwb_sigma or not get_intrinsic_diag:
                fisher_diag[i] = 1./cm.sigma_gwb_default**2
        else:
            if not get_intrinsic_diag:
                #don't need this value
                fisher_diag[i] = 1./cm.sigma_noise_default**2
                continue

            epsilon = cm.eps[par_names[i]]
            epsilons[i] = 2*epsilon

            paramsPP[i] += 2*epsilon
            paramsMM[i] -= 2*epsilon

            FLI_swap.logdet_array[:] = 0.
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

            nns[i] = FLI_swap.get_lnlikelihood(x0_swap)


            #must be one of the intrinsic parameters
            x0_swap.update_params(paramsPP)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0. #these are reset to nonzero by calling update_intrinsic, but they do not vary so don't include them in the likelihood
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            pps[i] = FLI_swap.get_lnlikelihood(x0_swap)#FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            x0_swap.update_params(paramsMM)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            mms[i] = FLI_swap.get_lnlikelihood(x0_swap)#,FLI_swap.resres,FLI_swap.logdet,FLI_swap.pos,FLI_swap.pdist,FLI_swap.NN,FLI_swap.MMs)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            #fisher_diag[i] = -(pps[i] - 2*nns[i] + mms[i])/(4*epsilon*epsilon)

            #revert changes
            #x0_swap.update_params(samples_fisher)

            #FLI_swap.cos_gwtheta = x0_swap.cos_gwtheta
            #FLI_swap.gwphi = x0_swap.gwphi
            #FLI_swap.log10_fgw = x0_swap.log10_fgw
            #FLI_swap.log10_mc = x0_swap.log10_mc#

            safe_reset_swap(FLI_swap,x0_swap,samples_fisher,FLI_mem0)

    for ii in range(dim):
        #calculate diagonal elements of the Hessian from a central finite element scheme
        #note the minus sign compared to the regular Hessian
        #defaulted values will already be nonzero so don't overwrite them
        if fisher_diag[ii] == 0.:
            fisher_diag[ii] = -(pps[ii] - 2*nns[ii] + mms[ii])/(epsilons[ii]**2)#+1./sigma_defaults[ii]**2
            if ii in x0_swap.idx_int:
                fisher_diag[ii] += 1./sigma_defaults[ii]**2

        if np.isnan(fisher_diag[ii]) or fisher_diag[ii] <= 0. :
            fisher_diag[ii] = 1/sigma_defaults[ii]**2#1./cm.sigma_noise_default**2#1/cm.sigma_cw0_p_phase_default**2

    #double check FLI_swap and x0_swap are a self consistent state
    safe_reset_swap(FLI_swap,x0_swap,samples_fisher,FLI_mem0)

    #filer out nans and negative values - set them to 1.0 which will result in
    fisher_diag[(~np.isfinite(fisher_diag))|(fisher_diag<0.)] = 1.

    #filter values smaller than 4 and set those to 4 -- Neil's trick -- effectively not allow jump Gaussian stds larger than 0.5=1/sqrt(4)
    eig_limit = 4.0
    #W = np.where(FISHER_diag>eig_limit, FISHER_diag, eig_limit)
    for ii in range(dim):
        if fisher_diag[ii]<eig_limit:
            #use input defaults instead of the eig limit
            if sigma_defaults[ii]>0.:
                fisher_diag[ii] = 1./sigma_defaults[ii]**2
            else:
                fisher_diag[ii] = eig_limit
    #for phases override the eig limit

    return 1/np.sqrt(fisher_diag),(pp2s,mm2s,nn2s,epsilons,helper_tuple0,pps,mms,nns)

def get_fisher_eigenvectors_common(params, x0_swap, FLI_swap, diagonal_data, epsilon=1e-4,default_all=False):
    """update just the 4x4 block of common eigenvectors

    :param params:          Parameters where the fisher is to be calculated
    :param x0_swap:         CWInfo object
    :param FLI_swap:        FastLikeInfo object
    :param diagonal_data:   --
    :param epsilon:         Perturbation values [1e-4]
    :param default_all:     --

    :return:                Matrix with fisher eigenvectors
    """
    print("Updating Common Parameter Fisher Eigenvectors")
    dim = 4
    idx_to_perturb = x0_swap.idx_cw_int[:dim]
    #par_names_to_perturb = par_names_cw_int[:4]
    _,_,_,epsilons_diag,helper_tuple0,pps,mms,nns = diagonal_data
    (_,FLI_mem0) = helper_tuple0

    fisher = np.zeros((dim,dim))
    sigma_defaults = np.array([cm.sigma_noise_default,cm.sigma_noise_default,cm.sigma_log10_fgw_default,cm.sigma_noise_default])
    diag_bad = np.zeros(dim,dtype=np.bool_)

    if default_all:
        #option just make all the fishers their default values for initialization
        for itrp in range(dim):
            fisher[itrp,itrp] = 1./sigma_defaults[itrp]**2
            diag_bad[itrp] = True
    else:
        #calculate diagonal elements
        for itrp in range(dim):
            idx_par = idx_to_perturb[itrp]
            #check not a factor of 4 in the denominator? Maybe is absorbed because epsilon is 2x as large when diagonal elements are computed
            fisher[itrp,itrp] = -(pps[idx_par] - 2.0*nns[idx_par] + mms[idx_par])/(epsilons_diag[idx_par]**2)#+1./sigma_defaults[itrp]**2
            if not np.isfinite(fisher[itrp,itrp]) or fisher[itrp,itrp]<=0.:  # diagonal elements cannot be 0 or negative
                print('bad diagonal',itrp,idx_par,pps[idx_par],nns[idx_par],mms[idx_par],epsilons_diag[idx_par],fisher[itrp,itrp], 1./sigma_defaults[itrp]**2)
                fisher[itrp,itrp] = 1./sigma_defaults[itrp]**2  # TODO pick defaults for diagonals more intelligently
                diag_bad[itrp] = True
        print(np.diag(fisher))
        if np.sum(diag_bad)>=1:
            #several went bad, so just assume they all did and default everything to diagonal defaults
            for itrp in range(dim):
                fisher[itrp,itrp] = 1./sigma_defaults[itrp]**2 
                diag_bad[itrp] = True

    #calculate off-diagonal elements
    for i in range(dim):
        #don't bother calculating the off-diagonals if we didn't get a good diagonal component for either
        if diag_bad[i]:
            continue

        for j in range(i+1,dim):
            #don't bother calculating the off-diagonals if we didn't get a good diagonal component for either
            if diag_bad[j]:
                continue

            #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPM = np.copy(params)
            paramsMP = np.copy(params)

            paramsPP[idx_to_perturb[i]] += epsilon
            paramsPP[idx_to_perturb[j]] += epsilon
            paramsMM[idx_to_perturb[i]] -= epsilon
            paramsMM[idx_to_perturb[j]] -= epsilon
            paramsPM[idx_to_perturb[i]] += epsilon
            paramsPM[idx_to_perturb[j]] -= epsilon
            paramsMP[idx_to_perturb[i]] -= epsilon
            paramsMP[idx_to_perturb[j]] += epsilon

            FLI_swap.logdet_array[:] = 0.
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)

            x0_swap.update_params(paramsPP)

            FLI_swap.update_intrinsic_params(x0_swap)
            #these are reset to nonzero by calling update_intrinsic, but they do not vary so don't include them in the likelihood
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            pp = FLI_swap.get_lnlikelihood(x0_swap)

            x0_swap.update_params(paramsMM)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            mm = FLI_swap.get_lnlikelihood(x0_swap)

            x0_swap.update_params(paramsPM)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            pm = FLI_swap.get_lnlikelihood(x0_swap)

            x0_swap.update_params(paramsMP)

            FLI_swap.update_intrinsic_params(x0_swap)
            FLI_swap.resres_array[:] = 0.
            FLI_swap.set_resres_logdet(FLI_swap.resres_array,FLI_swap.logdet_array,0.)
            mp= FLI_swap.get_lnlikelihood(x0_swap)

            safe_reset_swap(FLI_swap,x0_swap,params,FLI_mem0)


            #calculate off-diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher_loc = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
            if not np.isfinite(fisher_loc):
                fisher_loc = 0.

            fisher[i,j] = fisher_loc  # -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
            fisher[j,i] = fisher_loc  # fisher[i,j]
            #fisher[j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

    #if determinant is too small, rescale all off diagonal elements
    #by a common factor to increase the determinant while preserving as much structure as possible
    diag_prod = np.prod(np.diagonal(fisher))
    det_min_abs = 4.**dim  # minimum value to allow the determinant of the matrix to be
    det_min = max(det_min_abs,1.e-1*diag_prod)  # either use absolute minimum or enforce that matrix must be relatively diagonally dominat
    fisher_det = np.linalg.det(fisher)
    #print(fisher_det)
    itrt = 0
    #off diagonal term may sometimes need to be shrunk multiple times due to numerical precision limits in the multiplier
    while fisher_det < det_min and itrt<20:
        offdiag_prod = diag_prod-fisher_det
        if diag_prod<=det_min:
            #cannot fix by rescaling so turn off off-diagonals and rescale diagonals
            offdiag_mult = 0.
            fisher_scale = 1.1
            det_enhance = 1.1
            for itr1 in range(dim):
                fisher[itr1,itr1] *= fisher_scale*(det_min*det_enhance/diag_prod)**(1/dim)
        else:
            fisher_scale = 0.5
            det_enhance = 2.
            if offdiag_prod>0.:
                offdiag_mult = fisher_scale*(np.abs(diag_prod-det_min*det_enhance)/np.abs(offdiag_prod))**(1/dim)
            else:
                offdiag_mult = fisher_scale*(np.abs(det_min*det_enhance-diag_prod)/np.abs(offdiag_prod))**(1/dim)

        for itr1 in range(dim):
            for itr2 in range(itr1+1,dim):
                fisher[itr1,itr2] *= offdiag_mult
                fisher[itr2,itr1] = fisher[itr1,itr2]

        fisher_det = np.linalg.det(fisher)
        itrt += 1
    #    print(np.linalg.det(fisher),diag_prod,offdiag_prod,det_min,offdiag_mult)
    #    print(np.linalg.det(fisher)-det_min,diag_prod-offdiag_prod*offdiag_mult**dim)
    #assert np.linalg.det(fisher)>=det_min

    #Filter nans and infs and replace them with 1s
    #this will imply that we will set the eigenvalue to 100 a few lines below
    #print('fisher 1')
    #print(fisher)
    #print('fisher det raw ',np.linalg.det(fisher),np.prod(np.diagonal(fisher)))

    FISHER = np.where(np.isfinite(fisher), fisher, 1.0)
    if not np.array_equal(FISHER, fisher):
        print("Changed some nan elements in the Fisher matrix to 1.0")

    #Find eigenvalues and eigenvectors of the Fisher matrix
    w, v = np.linalg.eigh(FISHER)

    #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
    eig_limit = 1.0#1.0#0.25
    print('eig sizes',np.abs(w))

    W = np.where(np.abs(w)>eig_limit, w, eig_limit)

    return (np.sqrt(1.0/np.abs(W))*v).T

def get_fisher_eigenvectors(params, par_names, par_names_to_perturb, pta, epsilon=1e-4):
    """get fisher eigenvectors for a generic set of parameters the slow way

    :param params:                  Parameter values where fisher is to be calculated
    :param par_names:               List of parameter names
    :param par_names_to_perturb:    Subset of par_names for which we want to calculate the fisher
    :param pta:                     enterprise PTA object
    :param epsilon:                 Perturbation values [1e-4]

    :return:                        Matrix with fisher eigenvectors
    """
    try:
        dim = len(par_names_to_perturb)
        fisher = np.zeros((dim,dim))

        #lnlikelihood at specified point
        nn = pta.get_lnlikelihood(params)

        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[par_names.index(par_names_to_perturb[i])] += 2*epsilon
            paramsMM[par_names.index(par_names_to_perturb[i])] -= 2*epsilon

            #lnlikelihood at +-epsilon positions
            pp = pta.get_lnlikelihood(paramsPP)
            mm = pta.get_lnlikelihood(paramsMM)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

            if fisher[i,i] <= 0.:  # diagonal elements must be postive
                fisher[i,i] = 4.

        #calculate off-diagonal elements
        for i in range(dim):
            for j in range(i+1,dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPP[par_names.index(par_names_to_perturb[j])] += epsilon
                paramsMM[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsPM[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[j])] += epsilon

                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)
                pm = pta.get_lnlikelihood(paramsPM)
                mp = pta.get_lnlikelihood(paramsMP)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[j,i] = fisher[i,j]
                #fisher[j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

        #Filter nans and infs and replace them with 1s
        #this will imply that we will set the eigenvalue to 100 a few lines below
        print('fisher 2')
        print(fisher)
        print('fisher determinant',np.linalg.det(fisher),np.prod(np.diagonal(fisher)))
        FISHER = np.where(np.isfinite(fisher), fisher, 1.0)
        if not np.array_equal(FISHER, fisher):
            print("Changed some nan elements in the Fisher matrix to 1.0")

        #Find eigenvalues and eigenvectors of the Fisher matrix
        w, v = np.linalg.eigh(FISHER)

        #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
        eig_limit = 4.0

        W = np.where(np.abs(w)>eig_limit, w, eig_limit)

        return (np.sqrt(1.0/np.abs(W))*v).T

    except np.linalg.LinAlgError:
        print("An Error occured in the eigenvalue calculation")
        print(par_names_to_perturb)
        print(params)
        return np.eye(len(par_names_to_perturb))*0.5
