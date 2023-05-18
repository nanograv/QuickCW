import numpy as np
from scipy.stats import norm

from enterprise.signals import parameter


def DMDistPrior(value, dist, err):
    """Prior function for DMDist parameters.

    :param value:   point where we want the prior evaluated
    :param dist:    mean distance
    :param err:     distance error

    :return:        prior value
    """

    boxheight = 1/((dist+err)-(dist-err))
    gaussheight = 1/(np.sqrt(2*np.pi*(0.25*err)**2))

    y = np.where(value<=(dist-err), norm.pdf(value,dist-err, 0.25*err)*boxheight/gaussheight,
                 np.where((value>(dist-err)) & (value<(dist+err)), boxheight,
                          norm.pdf(value,dist+err, 0.25*err)*boxheight/gaussheight))

    area = 1+1*boxheight/gaussheight
    return y/area

def DMDistSampler(dist, err, size=None):
    """Sampling function for DMDist parameters.

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        random draw from prior (float or ndarray with lenght size)
    """

    boxheight = 1/((dist+err)-(dist-err))
    gaussheight = 1/(np.sqrt(2*np.pi*(0.25*err)**2))
    area = 1+1*boxheight/gaussheight

    #probability of being in the uniform part
    boxprob = 1/area

    #decide if we are in the middle or not
    alpha = np.random.uniform()
    if alpha<boxprob:
        return np.random.uniform(dist-err, dist+err)
    else:
        x = np.random.normal(0, 0.25*err, size=size)
        if x>0.0:
            return x+dist+err
        else:
            return x+dist-err

def DMDistParameter(dist=0, err=1, size=None):
    """Class factory for DM distance parameters with a pdf that is
    flat for dist+-err and a half Gaussian beyond that
    
    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        ``DMDist`` parameter class
    """

    class DMDist(parameter.Parameter):
        _size = size
        _prior = parameter.Function(DMDistPrior, dist=dist, err=err)
        _sampler = staticmethod(DMDistSampler)
        _typename = parameter._argrepr("DMDist", dist=dist, err=err)

    return DMDist

def PXDistPrior(value, dist, err):
    """Prior function for PXDist parameters.

    :param value:   point where we want the prior evaluated
    :param dist:    mean distance
    :param err:     distance error

    :return:        prior value
    """
    
    pi = 1/dist
    pi_err = err/dist**2

    return 1/(np.sqrt(2*np.pi)*pi_err*value**2)*np.exp(-(pi-value**(-1))**2/(2*pi_err**2))

def PXDistSampler(dist, err, size=None):
    """Sampling function for PXDist parameters.

    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter

    :return:        random draw from prior (float or ndarray with lenght size)
    """

    pi = 1/dist
    pi_err = err/dist**2

    #just draw parallax from Gaussian with proper mean and std and return its inverse
    return 1/np.random.normal(pi, pi_err)

def PXDistParameter(dist=0, err=1, size=None):
    """Class factory for PX distance parameters with a pdf of inverse Gaussian (since parallax is Gaussian)
    
    :param dist:    mean distance
    :param err:     distance error
    :param size:    length for vector parameter
    
    :return:        ``PXDist`` parameter class
    """

    class PXDist(parameter.Parameter):
        _size = size
        _prior = parameter.Function(PXDistPrior, dist=dist, err=err)
        _sampler = staticmethod(PXDistSampler)
        _typename = parameter._argrepr("PXDist", dist=dist, err=err)

    return PXDist
