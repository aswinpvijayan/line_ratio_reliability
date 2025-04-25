import numpy as np
from unyt import Angstrom

from synthesizer.emission_models.attenuation import Calzetti2000

#Weighted quantiles
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]


    weighted_quantiles = np.nancumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.nansum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)



def binned_weighted_quantile(x,y,weights,bins,quantiles):

    # if ~isinstance(quantiles,list):
    #     quantiles = [quantiles]

    out = np.full((len(bins)-1,len(quantiles)),np.nan)
    for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i,:] = weighted_quantile(y[mask],quantiles,sample_weight=weights[mask])

    return np.squeeze(out)

def calc_line_corr(line_lum, line_lam, Balmer_obs, slope=0):
    
    Halpha_lam  = 6562.80 * Angstrom
    Hbeta_lam   = 4861.32 * Angstrom
    intr_ratio = 2.79
    
    acurve = Calzetti2000(slope=slope)
    k_line = acurve.get_tau_at_lam(lam=line_lam)

    EBV = get_EBV(Balmer_obs, Halpha_lam, Hbeta_lam, intr_ratio, slope)
    
    A_line = EBV * k_line

    y = line_lum * 10**(A_line.value/2.5)
    
    return y


def get_EBV(Balmer_obs, lam1, lam2, intr_ratio, slope=0):
    
    acurve = Calzetti2000(slope=slope)

    k_1 = acurve.get_tau_at_lam(lam1)
    k_2 = acurve.get_tau_at_lam(lam2)

    EBV = (2.5/(k_2-k_1)) * np.log10(Balmer_obs/intr_ratio)

    return EBV

def get_EBV_from_Av(Av, slope=0):
    
    acurve = Calzetti2000(slope=slope)

    k_v = acurve.get_tau_at_lam(5500 * Angstrom)

    EBV = Av/k_v
    
    return EBV

def calc_line_corr_from_Av(line_lum, line_lam, Av, slope=0):
    
    acurve = Calzetti2000(slope=slope)
    k_line = acurve.get_tau_at_lam(lam=line_lam)

    EBV = get_EBV_from_Av(Av, slope=slope)
    
    A_line = EBV * k_line

    y = line_lum * 10**(A_line.value/2.5)
    
    return y

def get_flares_LF(dat, weights, bins, n):

    sims = np.arange(0,len(weights))

    hist = np.zeros(len(bins)-1)
    out = np.zeros(len(bins)-1)
    err = np.zeros(len(bins)-1)

    nsum = np.cumsum(n)
    nsum = np.append([0], nsum)
    nsum = nsum.astype(np.int32)

    for ii, sim in enumerate(sims):
        h, edges = np.histogram(dat[nsum[ii]:nsum[ii+1]], bins = bins)
        hist+=h
        out+=h*weights[ii]
        err+=np.square(np.sqrt(h)*weights[ii])

    return hist, out, np.sqrt(err)
