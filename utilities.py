import numpy as np
from unyt import Angstrom
import cmath
import warnings

from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.exceptions import InconsistentArguments

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

    out = np.full((len(bins)-1,len(quantiles)),np.nan)
    for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i,:] = weighted_quantile(y[mask],quantiles,sample_weight=weights[mask])

    return np.squeeze(out)

def calc_line_corr(line_lum, line_lam, Balmer_obs, slope=0):
    
    """
    Returns dust correction given luminosity, wavelength
    and the Balmer decrement for the Calzetti attenuation
    curve
    """
    
    Halpha_lam  = 6562.80 * Angstrom
    Hbeta_lam   = 4861.32 * Angstrom
    intr_ratio = 2.79
    
    acurve = Calzetti2000(slope=slope)
    k_line = acurve.get_tau_at_lam(lam=line_lam)

    EBV = get_EBV(Balmer_obs, Halpha_lam, Hbeta_lam, intr_ratio, slope)
    
    A_line = EBV * k_line

    y = line_lum * 10**(A_line/2.5)
    
    return y


def get_EBV(Balmer_obs, lam1, lam2, intr_ratio, slope=0):
    
    """
    Returns E(B-V) given the observed Balmer ratios for
    the Calzetti attenuation curve
    """
    
    acurve = Calzetti2000(slope=slope)

    k_1 = acurve.get_tau_at_lam(lam1)
    k_2 = acurve.get_tau_at_lam(lam2)

    EBV = (2.5/(k_2-k_1)) * np.log10(Balmer_obs/intr_ratio)

    return EBV

def get_EBV_from_Av(Av, slope=0):
    
    """
    Returns E(B-V) given the observed Av for
    the Calzetti attenuation curve
    """
    
    acurve = Calzetti2000(slope=slope)

    k_v = acurve.get_tau_at_lam(5500 * Angstrom)

    EBV = Av/k_v
    
    return EBV

def calc_line_corr_from_Av(line_lum, line_lam, Av, slope=0):
    
    """
    Returns dust correction given luminosity, wavelength
    and the Av for the Calzetti attenuation curve
    """
    
    acurve = Calzetti2000(slope=slope)
    k_line = acurve.get_tau_at_lam(lam=line_lam)

    EBV = get_EBV_from_Av(Av, slope=slope)
    
    A_line = EBV * k_line

    y = line_lum * 10**(A_line/2.5)
    
    return y

def get_flares_LF(dat, weights, bins, n):
    
    """
    Helper function to return FLARES dsitribution
    function provided the weight for the different
    regions
    """

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


#Curti+2024 calibrations
def curti_R2(x):
    # x = 12+log(O/H)-8.69
    #sigma_cal = 0.11
    c0 = 0.4326
    c1 = -1.0751
    c2 = -5.1141
    c3 = -5.5321
    c4 = -2.3009
    c5 = -0.2850
    
    return c0 + c1*x + c2*(x**2) + c3*(x**3) + c4*(x**4) + c5*(x**5)

def curti_R3(x):
    # x = 12+log(O/H)-8.69
    #sigma_cal = 0.09
    c0 = -0.2768
    c1 = -3.1422
    c2 = -2.73
    c3 = -0.6003
    
    return c0 + c1*x + c2*(x**2) + c3*(x**3)

def curti_O32(x):
    #sigma_cal = 0.15
    c0 = -0.6915
    c1 = -2.6856
    c2 = -1.1642
    
    return c0 + c1*x + c2*(x**2)

def curti_Ne3O2(x):
    #sigma=0.1 (average, since not given)
    c0 = -1.632
    c1 = -2.0606
    c2 = -0.46088
    
    return c0 + c1*x + c2*(x**2)

def curti_Rhat(x):
    # 0.47 * log10(R2) + 0.88 * log10(R3)
    #sigma_cal = 0.058
    c0 = -0.0478
    c1 = -3.0707
    c2 = -3.4164
    c3 = -1.0034
    c4 = -0.0379
    
    return c0 + c1*x + c2*(x**2) + c3*(x**3) + c4*(x**4)

# Modified code from Lucie Rowlands
# Sanders 2023 calibration functions
def quadratic_solution(a, b, c):
    """Solves a quadratic equation ax^2 + bx + c = 0."""
    discriminant = (b**2) - (4*a*c)
    if discriminant < 0:
        warnings.warn("Complex roots.")
        return None, None
    return (-b - cmath.sqrt(discriminant)) / (2*a), (-b + cmath.sqrt(discriminant)) / (2*a)

def Sanders23_O3(O3, Hbeta):
    O3_ratio = O3 / Hbeta
    sol1, sol2 = quadratic_solution(-0.453, -0.072, 0.834 - np.log10(O3_ratio))
    return (sol1 + 8, sol2 + 8) if sol1 and sol2 else (None, None)

def Sanders23_R23(o3_5007, o3_4959, o2, Hbeta):
    R23_ratio = (o3_5007 + o3_4959 + o2) / Hbeta
    sol1, sol2 = quadratic_solution(-0.331, 0.026, 1.017 - np.log10(R23_ratio))
    return (sol1 + 8, sol2 + 8) if sol1 and sol2 else (None, None)

def Sanders23_Ne3O2(Ne3, O2):
    """Convert Ne3O2 to Z"""
    
    return (np.log10(Ne3 / O2) + 0.386) / -0.998 + 8

def Sanders23_O32(O3, O2):
    """Convert O32 to metallicity"""
    
    return (np.log10(O3 / O2) - 0.723) / -1.153 + 8

def Z_to_O32(Z):
    """Convert metallicity to O32 ratio."""
    if Z is None:
        return None
    return 10**((Z - 8) * (-1.153) + 0.723)

def Z_to_Ne3O2(Z):
    """Convert metallicity to Ne3O2 ratio."""
    if Z is None:
        return None
    return 10**((Z - 8) * (-0.998) - 0.386)


def compute_metallicity_dust_correction(galaxy, Sanders=True, Avdust=False, Balmerdust=True, Av=None, Curti=False):
    """Computes the metallicity for a given galaxy using
        balmer corrected fluxes for line ratios.
        Using OIII]5007, [OIII]4959, [OII]3727,29, Hbeta
    """
    metallicity, notes = None, ""
    
    if (Avdust==True) and (Balmerdust==True):
        InconsistentArguments('Avdust and Balmerdust cannot both be True')     
   
    # R23 = galaxy['R23']
    # R3 = galaxy['R3']
    # Ne3O2 = galaxy['Ne3O2']
    # O32 = galaxy['O32']
    if Avdust==True:
        if Av is None:
            try:
                Av = galaxy['Av']
            except:
                InconsistentArguments('No Av provided anywhere')
                
        galaxy_corr = {}
        galaxy_corr['[OIII]5007'] = calc_line_corr_from_Av(galaxy['[OIII]5007'], 5006.84 * Angstrom, Av, slope=0)
        galaxy_corr['[OIII]4959'] = calc_line_corr_from_Av(galaxy['[OIII]4959'], 4958.91 * Angstrom, Av, slope=0)
        galaxy_corr['Hbeta'] = calc_line_corr_from_Av(galaxy['Hbeta'], 4861.32 * Angstrom, Av, slope=0)
        galaxy_corr['[OII]3727,29'] = calc_line_corr_from_Av(galaxy['[OII]3727,29'], np.mean([3726.03, 3728.81]) * Angstrom, Av, slope=0)
        O32 = galaxy_corr['[OIII]5007']/galaxy_corr['[OII]3727,29']
        galaxy_corr['NeIII3869'] = calc_line_corr_from_Av(galaxy['NeIII3869'], 3868.76 * Angstrom, Av, slope=0)
    elif Balmerdust==True:
        Balmer_obs = galaxy['Halpha']/galaxy['Hbeta']
        
        galaxy_corr = {}
        galaxy_corr['[OIII]5007'] = calc_line_corr(galaxy['[OIII]5007'], 5006.84 * Angstrom, Balmer_obs)
        galaxy_corr['[OIII]4959'] = calc_line_corr(galaxy['[OIII]4959'], 4958.91 * Angstrom, Balmer_obs)
        galaxy_corr['Hbeta'] = calc_line_corr(galaxy['Hbeta'], 4861.32 * Angstrom, Balmer_obs)
        galaxy_corr['[OII]3727,29'] = calc_line_corr(galaxy['[OII]3727,29'], np.mean([3726.03, 3728.81]) * Angstrom, Balmer_obs)
        O32 = galaxy_corr['[OIII]5007']/galaxy_corr['[OII]3727,29']
        galaxy_corr['NeIII3869'] = calc_line_corr(galaxy['NeIII3869'], 3868.76 * Angstrom, Balmer_obs)

    else:       
        galaxy_corr=galaxy
        O32=galaxy['O32']
        
        
    if Sanders:

        #O32 and R23 should only be used for corrected fluxes!!!
        #R3 and R23 give two metallicity solutions, an upper branch and a lower branch solution. We need to choose between them

        Z_high, Z_low = Sanders23_R23(galaxy_corr['[OIII]5007'], galaxy_corr['[OIII]4959'], galaxy_corr['[OII]3727,29'], galaxy_corr['Hbeta'])
        pred_O32_high, pred_O32_low = Z_to_O32(Z_high), Z_to_O32(Z_low)

        #Which metallicity solution, upper or lower branch, should we use?
        #For each metallicity, calculate the corresponding O32 ratio, and then determine which is closest to the observed O32 ratio
        if Z_high is not None and Z_low is not None:
            metallicity = Z_high if abs(np.log10(pred_O32_high) - np.log10(O32)) < abs(np.log10(pred_O32_low) - np.log10(O32)) else Z_low
            notes = 'R23, high branch' if metallicity == Z_high else 'R23, low branch'
            return metallicity.real, notes
        else:
            metallicity = Sanders23_O32(O3=galaxy_corr['[OIII]5007'], O2=galaxy_corr['[OII]3727,29'])
            notes = 'O32'
            return metallicity, notes      
        
    
    else:
        x = np.arange(-1.69, 1, 0.01)
        
        R2 = np.log10(galaxy_corr['[OII]3727,29']/galaxy_corr['Hbeta'])
        R2_sigma = 0.11
        R3 = np.log10(galaxy_corr['[OIII]5007']/galaxy_corr['Hbeta'])
        R3_sigma = 0.09
        O32 = np.log10(galaxy_corr['[OIII]5007']/galaxy_corr['[OII]3727,29'])
        O32_sigma = 0.15
        Ne3O2 = np.log10(galaxy_corr['NeIII3869']/galaxy_corr['[OII]3727,29'])
        Ne3O2_sigma = 0.1 # Not given, average
        Rhat = 0.47 * np.log10(galaxy_corr['[OII]3727,29']/galaxy_corr['Hbeta']) + 0.88*R3
        Rhat_sigma = 0.058
        
        x_err = 0.01
        
        logL = (curti_R2(x)-R2)**2 / (R2_sigma**2 + x_err**2) + (curti_R3(x)-R3)**2 / (R3_sigma**2 + x_err**2)  + (curti_O32(x)-O32)**2 / (O32_sigma**2 + x_err**2) + (curti_Ne3O2(x)-Ne3O2)**2 / (Ne3O2_sigma**2 + x_err**2) + (curti_Rhat(x)-Rhat)**2 / (Rhat_sigma**2 + x_err**2)
        
        arg_sol = np.argmin(logL)
        
        metallicity = x[arg_sol] + 8.69
        notes = 'curti'
        
        return metallicity, notes


def compute_metallicity_nodust_correction(galaxy):
    """Computes the metallicity for a given galaxy.
        This uses line ratios which are close in wavelength
        so that dust effects are considered to be negligible.
        Using R3 and Ne3O2 values
    """
    metallicity, notes = None, ""
    
    Ne3O2 = galaxy['Ne3O2']

    
    #When we don't have H-alpha, use un-corrected catalogue, and use only R3 and Ne3O2, since they are closer in wavelength so attenuation is less of a concern
   
    Z_high, Z_low = Sanders23_O3(galaxy['[OIII]5007'], galaxy['Hbeta'])
    pred_Ne3O2_high, pred_Ne3O2_low = Z_to_Ne3O2(Z_high), Z_to_Ne3O2(Z_low)

    if Z_high is not None and Z_low is not None:

        metallicity = Z_high if abs(np.log10(pred_Ne3O2_high) - np.log10(Ne3O2)) < abs(np.log10(pred_Ne3O2_low) - np.log10(Ne3O2)) else Z_low
        notes = 'R3, high branch' if metallicity == Z_high else 'R3, low branch'
        
        return metallicity.real, notes
      
    else:
        metallicity =  Sanders23_Ne3O2(Ne3=galaxy['NeIII3869'], O2=galaxy['[OII]3727,29'])
        notes = 'Ne3O2'  
        return metallicity, notes
 
        

# Nakajima 2023
def R23_fit(logR23, O32, hbew):

    x = np.arange(-2., 0.5, 10000)

    if 100<hbew<200:
        c0, c1, c2 =  0.986, -0.178, -0.463
        expr = c0 + c1 * x + c2 * x*x
        logR23err = 0.10
        OHerr = 0.14
        high=8.9
    elif hbew<=100:
        c0, c1, c2 = 0.875, -0.313, -0.387
        expr = c0 + c1 * x + c2 * x*x
        logR23err = 0.08
        OHerr = 0.13
        high=8.0
    elif hbew>=200:
        c0, c1, c2 = 0.866, -0.515, -0.463
        expr = c0 + c1 * x + c2 * x*x
        logR23err = 0.05
        OHerr = 0.25
        high=8.1

    sol1, sol2 = np.sort(quadratic_solution(c2, c1, c0 - logR23))[::-1]
    
    pred_O32_high, pred_O32_low = O32_exp(sol1, hbew), O32_exp(sol2, hbew)
    
    sol = sol1 if abs(np.log10(pred_O32_high) - np.log10(O32)) < abs(np.log10(pred_O32_low) - np.log10(O32)) else sol2
    
    metallicity = sol+8.69
    
    if metallicity.imag!=0:
        metallicity = np.max(x)+8.69
        return metallicity
    elif (metallicity.imag==0) and (metallicity<=8.0):
        return metallicity.real    
    else:        
        sols = np.sort(np.roots([-0.274, -1.392, -1.474, 0.515-logR23]))[::-1]
        ok = np.isreal(sols)
        sols = sols[ok]       
        if np.sum(ok)>0:
            pred_O32 = np.array([O32_exp(sol, hbew, all=True) for sol in sols])        
            diff = np.argsort(np.abs(pred_O32 - np.log10(O32)))            
            return sol[diff[0]] + 8.69
        else:            
            return None
                

def O32_exp(x, hbew=100, all=False):

    if all:
        c0, c1, c2 =  -0.693, -2.722, -1.201
        expr = c0 + c1 * x + c2 * x*x
        
        return expr
        

    if 100<hbew<200:
        c0, c1, c2 =  -0.693, -2.722, -1.201
        expr = c0 + c1 * x + c2 * x*x
        logO32err = 0.39
        OHerr = 0.39
        high=8.9
    elif hbew<=100:
        c0, c1, c2 =  0.865, 0.771, 0.243
        expr = c0 + c1 * x + c2 * x*x
        logO32err = 0.31
        OHerr = 0.48
        high=7.9
    elif hbew:
        c0, c1, c2 = -0.080, -2.008, -0.804
        expr = c0 + c1 * x + c2 * x*x
        logO32err = 0.25
        OHerr = 0.38
        high=8.0

    return expr

# Izotov Z calibration
def compute_Izotov_Z(galaxy, dust=True):
    """Computes the metallicity for a given galaxy using
        balmer corrected fluxes for line ratios.
        Using OIII]5007, [OIII]4959, [OII]3727,29, Hbeta
    """
    
    Balmer_obs = galaxy['Halpha']/galaxy['Hbeta']
    
    if dust==True:
        galaxy_corr = {}
        galaxy_corr['[OIII]5007'] = calc_line_corr(galaxy['[OIII]5007'], 5006.84 * Angstrom, Balmer_obs)
        galaxy_corr['[OIII]4959'] = calc_line_corr(galaxy['[OIII]4959'], 4958.91 * Angstrom, Balmer_obs)
        galaxy_corr['Hbeta'] = calc_line_corr(galaxy['Hbeta'], 4861.32 * Angstrom, Balmer_obs)
        galaxy_corr['[OII]3727,29'] = calc_line_corr(galaxy['[OII]3727,29'], np.mean([3726.03, 3728.81]) * Angstrom, Balmer_obs)
        R23 = (galaxy_corr['[OIII]5007'] + galaxy_corr['[OIII]4959'] + galaxy_corr['[OII]3727,29'])/galaxy_corr['Hbeta']
        O32 = galaxy_corr['[OIII]5007']/galaxy_corr['[OII]3727,29']

    else:       
        galaxy_corr=galaxy
        R23 = galaxy['R23']
        O32 = galaxy['O32']
    
    R23 = float(R23)
    O32 = float(O32)
        
    Z =  0.950 * np.log10(R23 - 0.08 * O32) + 6.805
    
    return Z