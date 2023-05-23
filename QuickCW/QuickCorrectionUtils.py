"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
utils for correcting parameters to nominal ranges"""
import numpy as np
from numba import njit
import QuickCW.const_mcmc as cm
import enterprise.constants as const

@njit()
def reflect_cosines(cos_in,angle_in,rotfac=np.pi,modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
       which requires also rotating the signal by rotfac each time, then mod the angle by modfac

    :param cos_in:    cosine of polar angle
    :param angle_in:  azimuthal angle
    :param rotfac:    angle to get to the other side of pole [np.pi]
    :param modfac:    angle to change azimuthal angle by before getting back to the same point [2*np.pi]

    :return cos_in:   Reflected cosine of polar angle
    :return angle_in: Reflected azimuthal angle
    """
    if cos_in < -1.:
        cos_in = -1.+(-(cos_in+1.))%4
        angle_in += rotfac
    if cos_in > 1.:
        cos_in = 1.-(cos_in-1.)%4
        angle_in += rotfac
        #if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
        if cos_in < -1.:
            cos_in = -1.+(-(cos_in+1.))%4
            angle_in += rotfac
    angle_in = angle_in%modfac
    return cos_in,angle_in

@njit()
def reflect_cosines_array(cos_ins,angle_ins,rotfac=np.pi,modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
       which requires also rotating the signal by rotfac each time, then mod the angle by modfac
       same as reflect_cosines but handles arrays

    :param cos_in:    cosine of polar angle
    :param angle_in:  azimuthal angle
    :param rotfac:    angle to get to the other side of pole [np.pi]
    :param modfac:    angle to change azimuthal angle by before getting back to the same point [2*np.pi]

    :return cos_in:   Reflected cosine of polar angle
    :return angle_in: Reflected azimuthal angle
    """
    for itrk in range(cos_ins.size):
        if cos_ins[itrk] < -1.:
            cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
            angle_ins[itrk] += rotfac
        if cos_ins[itrk] > 1.:
            cos_ins[itrk] = 1.-(cos_ins[itrk]-1.)%4
            angle_ins[itrk] += rotfac
            #if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
            if cos_ins[itrk] < -1.:
                cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
                angle_ins[itrk] += rotfac
        angle_ins[itrk] = angle_ins[itrk]%modfac
    return cos_ins,angle_ins

@njit()
def reflect_into_range(x, x_low, x_high):
    """reflect an arbitrary parameter into a nominal range

    :param x:       Value of parameter to reflect in range
    :param x_low:   Lower bound of parameter
    :param x_high:  Upper bound of prameter

    :return res:    Value of parameter reflected into range
    """
    #ensure always returns something in range (i.e. do an arbitrary number of reflections) similar to reflect_cosines but does not need to track angles
    x_range = x_high-x_low
    res = x
    if res<x_low:
        res = x_low+(-(res-x_low))%(2*x_range)  # 2*x_low - x
    if res>x_high:
        res = x_high-(res-x_high)%(2*x_range)  # 2*x_high - x
        if res<x_low:
            res = x_low+(-(res-x_low))%(2*x_range)  # 2*x_low - x
    return res

def check_merged(log10_fgw,log10_mc,max_toa):
    """check the maximum toa is not such that the source has already merged, and if so draw new parameters to avoid starting from nan likelihood

    :param log10_fgw:   log base 10 of GW frequency
    :param log10_mc:    log base 10 of chirp mass
    :param max_toa:     Latest TOA in any pulsar in the array

    :return:            True if source already merged, False otherwise
    """
    w0 = np.pi * 10.0**log10_fgw
    mc = 10.0**log10_mc

    if (1. - 256./5. * (mc * const.Tsun)**(5./3.) * w0**(8./3.) * (max_toa - cm.tref)) >= 0:
        return False
    else:
        return True

@njit()
def correct_extrinsic(sample,x0):
    """correct extrinsic parameters for phases and cosines
    
    :param sample:  Array with parameters
    :param x0:      CWInfo object

    :return sample: Corrected sample array
    """
    sample[x0.idx_cos_inc],sample[x0.idx_psi] = reflect_cosines(sample[x0.idx_cos_inc],sample[x0.idx_psi],np.pi/2,np.pi)
    sample[x0.idx_phase0] = sample[x0.idx_phase0]%(2*np.pi)
    sample[x0.idx_phases] = sample[x0.idx_phases]%(2*np.pi)
    return sample

@njit()
def correct_extrinsic_array(samples,x0):
    """correct extrinsic parameters for phases and cosines
       same as correct_extrinsic but handles mutiple samples

    :param samples:     2D array with parameters for multiple samples
    :param x0:          CWInfo object

    :return samples:    Corrected samples array
    """
    samples[:,x0.idx_cos_inc],samples[:,x0.idx_psi] = reflect_cosines_array(samples[:,x0.idx_cos_inc],samples[:,x0.idx_psi],np.pi/2,np.pi)
    samples[:,x0.idx_phase0] %= (2*np.pi)
    samples[:,x0.idx_phases] %= (2*np.pi)
    return samples

@njit()
def correct_intrinsic(sample,x0,freq_bounds,cut_par_ids, cut_lows, cut_highs):
    """correct intrinsic parameters for phases and cosines

    :param sample:      Array with parameters
    :param x0:          CWInfo object
    :param freq_bounds: Lower and upper prior bounds of GW frequency
    :param cut_par_ids: Indices of parameters needing extra check
    :param cut_lows:    Lower bounds of parameters needing extra check
    :param cut_highs:   Upper bounds of parameters needing extra check

    :return sample:     Corrected sample array
    """
    sample[x0.idx_cos_gwtheta],sample[x0.idx_gwphi] = reflect_cosines(sample[x0.idx_cos_gwtheta],sample[x0.idx_gwphi],np.pi,2*np.pi)

    for itr in range(cut_par_ids.size):
        idx = cut_par_ids[itr]
        sample[idx] = reflect_into_range(sample[idx],cut_lows[itr],cut_highs[itr])

    sample[x0.idx_log10_fgw] = reflect_into_range(sample[x0.idx_log10_fgw], np.log10(freq_bounds[0]), np.log10(freq_bounds[1]))

    return sample
