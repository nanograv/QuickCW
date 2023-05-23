"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
Helpers to print outputs"""
import numpy as np
import h5py

row_labels = ['dist-prior-noproj','dist-prior-proj','dist-DE-noproj','dist-DE-proj','dist-fisher-noproj','dist-fisher-proj',\
              'RN-prior-noproj','RN-prior-proj','RN-DE-noproj','RN-DE-proj','RN-fisher-noproj','RN-fisher-proj',\
              'GWB-prior-noproj','GWB-prior-proj','GWB-DE-noproj','GWB-DE-proj','GWB-fisher-noproj','GWB-fisher-proj',\
              'common-prior-noproj','common-prior-proj','common-DE-noproj','common-DE-proj','common-fisher-noproj','common-fisher-proj',\
              'all-prior-noproj','all-prior-proj','all-DE-noproj','all-DE-proj','all-fisher-noproj','all-fisher-proj',\
              'PT','proj']
display_names ={'dist-prior-noproj':'Dist Prior','dist-DE-noproj':'Dist DE','dist-fisher-noproj':'Dist Fisher',\
                'RN-prior-noproj':'RN Emp Dist','RN-DE-noproj':'RN DE','RN-fisher-noproj':'RN Fisher',\
                'GWB-prior-noproj':'GWB Prior','GWB-DE-noproj':'GWB DE','GWB-fisher-noproj':'GWB Fisher',\
                'common-prior-noproj':'Common Prior','common-DE-noproj':'Common DE','common-fisher-noproj':'Common Fisher',\
                'all-prior-noproj':'All Prior','all-DE-noproj':'All DE','all-fisher-noproj':'All Fisher',\
                'dist-prior-proj':'Dist Prior','dist-DE-proj':'Dist DE','dist-fisher-proj':'Dist Fisher',\
                'RN-prior-proj':'RN Emp Dist','RN-DE-proj':'RN DE','RN-fisher-proj':'RN Fisher',\
                'GWB-prior-proj':'GWB Prior','GWB-DE-proj':'GWB DE','GWB-fisher-proj':'GWB Fisher',\
                'common-prior-proj':'Common Prior','common-DE-proj':'Common DE','common-fisher-proj':'Common Fisher',\
                'all-prior-proj':'All Prior','all-DE-proj':'All DE','all-fisher-proj':'All Fisher',\
                'PT':'PT','proj':'proj'}

def print_acceptance_progress(itrn,N,n_int_block,a_yes,a_no,t_itr,ti_loop,tf1_loop,Ts,verbosity):
    """print the acceptance fraction

    :param itrn:        Overall index (as opposed to the block index itri or the index within saved values itrb)
    :param N:           Total number of iterations we asked for at startup
    :param n_int_block: Number of iterations within a block
    :param a_yes:       Array holding number of accepted steps
    :param a_no:        Array holding number of rejected steps
    :param t_itr:       Time just before calling this function got from time.perf_counter()
    :param ti_loop:     Time after initialization got from time.perf_counter()
    :param tf1_loop:    Time at start of this block got from time.perf_counter()
    :param Ts:          Temperature ladder
    :param verbosity:   Parameter indicating how much info to print (higher value means more prints)
    """
    with np.errstate(invalid='ignore'):
        acc_fraction = a_yes/(a_no+a_yes)
    t_initial = tf1_loop-ti_loop
    t_spent = t_itr-tf1_loop
    if itrn>n_int_block:
        #forecast total time based on average speed ignoring the very first block which may include compile time
        t_forecast = (N-n_int_block)/(itrn-n_int_block)*t_spent+t_initial
    elif itrn>0:
        t_forecast = N/itrn*t_initial
    else:
        t_forecast = np.nan
    print('Progress: {0:2.2f}% '.format(itrn/N*100)+"at t= %8.3fs forecast loop time %8.3fs"%(t_spent,t_forecast))

    #print('Acceptance fraction #columns: chain number; rows: proposal type (for morphological: w/o and w/ projection perturbation) (dist-prior, dist-DE, dist-fisher, RN-prior, RN-DE, RN-fisher, common-prior, common-DE, common-fisher, PT, proj):')
    print('Acceptance fractions for various jumps')
    str_build = "%-13s "%"Jump Name"
    for itrc in range(acc_fraction.shape[1]):
        str_build += " chain %3d "%itrc
    print(str_build)

    str_build = "%-13s "%"Temperature"
    for itrc in range(acc_fraction.shape[1]):
        str_build += " %.3e "%Ts[itrc]
    print(str_build)

    str_proj_rows = []
    str_noproj_rows = []

    for itrp,label in enumerate(row_labels):
        str_build = "%-13s "%display_names[label]
        #skip the one jump that is never attempted, unless it actually has been attempted

        for itrc in range(acc_fraction.shape[1]):
            if np.isnan(acc_fraction[itrp,itrc]):
                str_build += " No Trials "
            elif acc_fraction[itrp,itrc] == 0.:
                #alternate printout to roughly indicate how many have been tried if acceptance is 0
                str_build += "<%8.7f "%(1./(a_yes[itrp,itrc]+a_no[itrp,itrc]))
            else:
                str_build += " %8.7f "%acc_fraction[itrp,itrc]

        if ('RN-prior' in label or 'GWB-prior' in label or 'all-prior' in label) and np.all(np.isnan(acc_fraction[itrp])):
            continue
        if '-noproj' not in label:
            str_proj_rows.append(str_build)
        else:
            str_noproj_rows.append(str_build)

    for str_build in str_proj_rows:
        print(str_build)

    if verbosity>1:
        #the noproj rows have different meaning so print them as a separate channel
        print("")
        print("Fraction of time Multiple Try selected first sample, whether or not trial was accepted:")
        for str_build in str_noproj_rows:
            print(str_build)


def output_hdf5_loop(itrn,chain_params,samples,log_likelihood,acc_fraction,fisher_diag,par_names,N,verbosity):
    """output to hdf5 at loop iteration

    :param itrn:            Overall index (as opposed to the block index itri or the index within saved values itrb)
    :param chain_params:    ChainParams object
    :param samples:         Array with posterior samples
    :param log_likelihood:  Array with log likelihood values corresponding to samples
    :param acc_fraction:    Array with acceptance rates
    :param fisher diag:     Array with the digonal elements of fisher matrix
    :param par_names:       List of parameter names
    :param N:               Total number of samples
    :param verbosity:       Parameter indicating how much info to print (higher value means more prints)
    """
    n_chain = chain_params.n_chain
    save_every_n = chain_params.save_every_n
    Ts = chain_params.Ts
    savefile = chain_params.savefile
    save_first_n_chains = chain_params.save_first_n_chains
    samples_precision = chain_params.samples_precision
    thin = chain_params.thin
    if savefile is not None:
        if itrn>save_every_n:
            if verbosity>1:
                print("Append to HDF5 file...")
            with h5py.File(savefile, 'a') as f:
                f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:save_first_n_chains,:-1:thin,:]
                f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                f['acc_fraction'][...] = np.copy(acc_fraction)
                f['fisher_diag'][...] = np.copy(fisher_diag)
                f['T-ladder'][...] = np.copy(Ts)
                f['samples_freq'].resize((f['samples_freq'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                f['samples_freq'][:,-int((samples.shape[1]-1)/thin):] = samples[:,:-1:thin,par_names.index('0_log10_fgw')]
        else:
            if verbosity>1:
                print("Create HDF5 file...")
            with h5py.File(savefile, 'w') as f:
                f.create_dataset('samples_cold', data=samples[:save_first_n_chains,:-1:thin,:], dtype=samples_precision, compression="gzip", chunks=True, maxshape=(save_first_n_chains,int(N/thin),samples.shape[2]))
                f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape=(samples.shape[0],int(N/thin)))
                f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
                f.create_dataset('acc_fraction', data=acc_fraction)
                f.create_dataset('fisher_diag', data=fisher_diag)
                f.create_dataset('T-ladder', data=Ts)
                f.create_dataset('samples_freq', data=samples[:,:-1:thin,par_names.index('0_log10_fgw')], dtype=samples_precision, compression="gzip", chunks=True, maxshape=(n_chain,int(N/thin)))

def output_hdf5_end(chain_params,samples,log_likelihood,acc_fraction,fisher_diag,par_names,verbosity):
    """output to hdf5 file at end of body of loop

    :param chain_params:    ChainParams object
    :param samples:         Array with posterior samples
    :param log_likelihood:  Array with log likelihood values corresponding to samples
    :param acc_fraction:    Array with acceptance rates
    :param fisher diag:     Array with the digonal elements of fisher matrix
    :param par_names:       List of parameter names
    :param verbosity:       Parameter indicating how much info to print (higher value means more prints)
    """
    Ts = chain_params.Ts
    savefile = chain_params.savefile
    save_first_n_chains = chain_params.save_first_n_chains
    thin = chain_params.thin
    if savefile is not None:
        if verbosity>1:
            print("Append to HDF5 file...")
        with h5py.File(savefile, 'a') as f:
            f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
            f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:save_first_n_chains,:-1:thin,:]
            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
            f['acc_fraction'][...] = np.copy(acc_fraction)
            f['fisher_diag'][...] = np.copy(fisher_diag)
            f['T-ladder'][...] = np.copy(Ts)
            f['samples_freq'].resize((f['samples_freq'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
            f['samples_freq'][:,-int((samples.shape[1]-1)/thin):] = samples[:,:-1:thin,par_names.index('0_log10_fgw')]
