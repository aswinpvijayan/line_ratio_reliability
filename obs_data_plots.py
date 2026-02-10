"""
This module contains functions to plot the observed data from the literature that we compare to in the paper.
"""

import numpy as np
import pandas as pd
from astropy.io import fits

def plot_met_data(paper, ax, xaxis='mstar', ratio='R23'):
    """Function to plot the observed data from the literature that we compare to in the paper.

    Args:
        paper (str): The name of the paper to plot. Options are 'rebels', 'nakajima', 'heintz', and 'curti'.
        ax (matplotlib axis): The axis to plot on.
        xaxis (str): The x-axis to plot. Options are 'mstar' and 'muv'. Only applies to the rebels data.
        ratio (str): The line ratio diagnostic to plot. Options are 'R23' and 'O32'. Only applies to the rebels data.
        
    Returns:
        ax (matplotlib axis): The axis with the data plotted on it.
    """
       
    if paper=='rebels':
        data = fits.open('./data/REBELS_IFU_forAswin_dustcorr.fits')[1].data
        mstar = data['logMstar']
        mstar_up = data['logMstar_uerr'] - mstar
        mstar_lo = mstar - data['logMstar_lerr'] 
        met = data['metallicity']
        met_err = data['metallicity_err']
        diagnostic = data['Diagnostic']
        muv = np.array([-21.6, -21.8, -22.5, -22.6, -22.6, -22.5, -21.7, -22.3, -21.7, -22.5, -21.9, -22.7])
        muv_err = np.array([0.2, 0.4, 0.3, 0.4, 0.3, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
                
        ok = (diagnostic=='R23') | (diagnostic=='O32')
        if ratio!='R23':
            ok = np.invert(ok)        
        
        if xaxis=='mstar':
            ax.errorbar(mstar[ok], met[ok], xerr=[mstar_lo[ok], mstar_up[ok]], yerr=met_err[ok], marker='d', markersize=5, color='grey', label='Rowland+2025', linestyle='None', alpha=0.3)
        else:
            ax.errorbar(muv[ok], met[ok], xerr=muv_err[ok], yerr=met_err[ok], marker='d', markersize=5, color='grey', label='Rowland+2025', linestyle='None', alpha=0.3)
        
        
    if paper=='nakajima':
        obs_data = pd.read_excel('./data/Nakajima_2023.xlsx')
        obs_z = np.array(obs_data['Redshift'])[1:]
        obs_Muv = np.array(obs_data['Muv'])[1:]
        obs_Muverr = np.array(obs_data['err1_Muv'])[1:]
        obs_Mstar = np.array(obs_data['log_Mstar'])[1:]
        obs_Mstarerr = np.array(obs_data['err1_log_Mstar'])[1:]
        obs_logOH = np.array(obs_data['12+log(O/H)'])[1:]
        obs_logOHerr = np.array(obs_data['err1_log(O/H)'])[1:]
        
        ok = (6.5<obs_z) * (obs_z<7.5)
        
        x, y, xerr, yerr, c = obs_Mstar[ok], obs_logOH[ok], obs_Mstarerr[ok], obs_logOHerr[ok], obs_Muv[ok]
        xerr = xerr.astype(np.float32)
        yerr = yerr.astype(np.float32)
        xuplims, yuplims = np.zeros(len(x)), np.zeros(len(x))
        for ii in range(len(x)):

            if '<' in x[ii]:
                x[ii] = x[ii][1:]
                xuplims[ii] = 1
            
            # if '<' in y[ii]:
            #     y[ii] = y[ii][1:]
            #     yuplims[ii] = 1

            
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # c = c.astype(np.float32)
        
        ok = x>8
        x, y, xerr, yerr, xuplims, yuplims = x[ok], y[ok], xerr[ok], yerr[ok], xuplims[ok], yuplims[ok] 
        
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, xuplims=xuplims, uplims=yuplims, marker='s', markersize=5, label='Nakajima+2023', linestyle='None', color='grey', alpha=0.3)
        
    if paper=='heintz':
        
        obs_Mstar       = np.array([8.84-np.log10(19.2), 8.19-np.log10(2.25), 8.72-np.log10(1.46), 9.05-np.log10(1.33), 8.88-np.log10(2.12), 8.45, 8.33, 8.66, 8.40, 10.0, 8.85, 9.05, 9.07, 9.47, 9.04, 8.64])
        obs_Mstar_up    = np.array([0.02, 0.08, 0.02, 0.06, 0.02, 0.03, 0.05, 0.02, 0.05, 0.01, 0.07, 0.03, 0.01, 0.04, 0.10, 0.05])
        obs_Mstar_low   = np.array([0.03, 0.06, 0.03, 0.05, 0.02, 0.02, 0.03, 0.02, 0.04, 0.01, 0.06, 0.02, 0.01, 0.06, 0.11, 0.05])
        obs_Z           = np.array([7.56, 7.29, 7.68, 7.30, 7.97, 7.82, 7.49, 7.42, 7.86, 8.06, 8.06, 7.55, 7.75, 8.00, 7.62, 7.42])
        obs_Z_up        = np.array([0.16, 0.22, 0.18, 1.00, 0.28, 0.18, 0.16, 0.18, 0.23, 0.17, 1.00, 0.21, 0.17, 0.29, 0.25, 0.19])
        obs_Z_low       = np.array([0.17, 0.28, 0.19, 0.01, 0.30, 0.20, 0.17, 0.18, 0.32, 0.17, 0.01, 0.27, 0.17, 0.34, 0.28, 0.20])
        
        ax.errorbar(obs_Mstar, obs_Z, xerr=[obs_Mstar_low, obs_Mstar_up], yerr=[obs_Z_low, obs_Z_up], marker='^', markersize=5, label='Heintz+2023', linestyle='None', color='grey', alpha=0.3)
        
    if paper=='Curti':
        
        curti_data = np.genfromtxt('./data/curti_2024_met.txt', skip_header=1, dtype=np.float32)
        z, mstar, mstarup, mstarlow, Z, Zup, Zlow = curti_data[:,0], curti_data[:,1], curti_data[:,2], curti_data[:,3], curti_data[:,4], curti_data[:,5], curti_data[:,6]
        
           
        ax.errorbar(mstar, Z, xerr=[mstarlow, mstarup], yerr=[Zlow, Zup], marker='<', markersize=5, label='Curti+2024', linestyle='None', color='grey', alpha=0.3)
        
    return ax

def plot_rebels_data(dataset):
    
    data = fits.open('./data/REBELS_IFU_forAswin_nodustcorr.fits')[1].data
    
    return np.array(data[dataset])