# -*- coding: utf-8 -*-

###############################################################################
## Functions for computing the power spectrum fits and statistical tests,    ##
## and for plotting results on brain maps.                                   ##
## Author: Pablo Martinez Cañada (pablomc@ugr.es)                            ##
## Date: 23/3/2023                                                           ##
###############################################################################

import numpy as np
import sys,os
import scipy
from fooof import FOOOF
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colorbar
from nilearn import plotting
from nilearn import image
from nilearn.image import math_img
from nilearn.image import binarize_img
from nilearn import datasets
from nilearn import surface

# Algorithm for parameterizing neural power spectra
def power_spectrum_fit(fx,PSD,fit_parameters):
    # Parameters
    freq_range_FOOOF_fitting = fit_parameters['freq_range_FOOOF_fitting']
    FOOOF_max_n_peaks = fit_parameters['FOOOF_max_n_peaks']
    FOOOF_peak_width_limits = fit_parameters['FOOOF_peak_width_limits']
    FOOOF_peak_threshold = fit_parameters['FOOOF_peak_threshold']

    # Initialize FOOOF
    fm = FOOOF(max_n_peaks = FOOOF_max_n_peaks,
               peak_width_limits = FOOOF_peak_width_limits,
               peak_threshold = FOOOF_peak_threshold)

    # Run FOOOF model
    fm.fit(fx, PSD, freq_range_FOOOF_fitting)

    # Gather all results
    ap_params, peak_params, r_squared,fit_error, gauss_params = fm.get_results()

    # Power spectrum fit
    if type(ap_params) == np.ndarray:
        # fitting
        ap_fit = np.power(10,ap_params[0])*1./(np.power(fx,ap_params[1]))

        # Periodic+aperiodic fit
        ys = np.zeros_like(fx)
        for ii in gauss_params:
            ctr, hgt, wid = ii
            ys = ys + hgt * np.exp(-(fx-ctr)**2 / (2*wid**2))
        sum_signal = np.power(10,ys)*ap_fit

    return [ap_params,sum_signal]

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# Post hoc Tukey’s Test
def tukeyhsd(x,Vax):
    # Tukey’s Test
    group_x = []
    for n in range(len(x)):
        group_x.append(n * np.ones(len(x[n])))
    # Create dataframe
    df = pd.DataFrame({'score': np.hstack(x),
                       'group': np.hstack(group_x)})
    # Perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'],
                              groups=df['group'],
                              alpha=0.05)

    # display results
    print(tukey)

    # Find the smallest top-3 p-values
    pos_p_values = np.argsort(tukey.pvalues)[0:3]

    # Plot p-values
    index = 0

    # Vax y-limits
    Vax_lims = Vax.get_ylim()

    for i in np.arange(0,len(x)):
        for j in np.arange(i+1,len(x)):
            if index in pos_p_values:
                # y value to place the plot
                height = Vax_lims[1]
                scale = 0.2 * index * (Vax_lims[1] - Vax_lims[0])

                # # Extra alignment
                extra_alignment = 0

                # Label
                pvalue = tukey.pvalues[index]
                dvalue = cohend(x[i],x[j])

                if pvalue >= 0.01:
                    star_label = "p = %.2f d = %.2f" % (pvalue,dvalue)
                elif pvalue >= 0.001 and pvalue < 0.01:
                    star_label = "p < 0.01 d = %.2f" % (dvalue)
                else:
                    star_label = "p < 0.001 d = %.2f" % (dvalue)

                # p-value
                Vax.text(1.,height + scale + 0.1 * (Vax_lims[1] - Vax_lims[0]) +\
                        extra_alignment,star_label, horizontalalignment = 'center',
                        verticalalignment = 'center',fontsize = 8)
                # Vax.text(i+(j-i)/2.,height,'p = '+"{:.1e}".format(tukey.pvalues[index]),
                #         horizontalalignment = 'center', fontdict=font_text_3)
                # line
                Vax.plot([i,j],[height+scale,height+scale],color = 'k',linewidth = 1.0)


            index+=1

    # Reconfigure axis
    Vax.set_ylim([Vax_lims[0],height+scale + 0.2 * (Vax_lims[1] - Vax_lims[0])])

# Map data to brain volumes
def plot_MEG_volumes(diff,figure,axes,hemi,vmax,view, showcb = True):
    # 4D fMRI atlas
    file = 'fMRI_parcellation_ds8mm.nii.gz'
    atlas = image.load_img(file)

    # Map data to brain maps
    for ROI in range(38):
        # Select the volume corresponding to the ROI
        brain_map = image.index_img(atlas,ROI)
        # Scalar multiplication with data + normalization
        brain_map = math_img("img1 * %s" % (diff[ROI] / np.max(brain_map.get_fdata())),
                                            img1 = brain_map)
        # Add scaled volume to final brain volume
        if ROI > 0:
            result_img = math_img("img1 + img2",img1 = result_img,img2 = brain_map)
        else:
            result_img = brain_map

    # Get a cortical mesh
    fsaverage = datasets.fetch_surf_fsaverage()
    # Sample the 3D data around each node of the mesh
    texture_right = surface.vol_to_surf(result_img,fsaverage.pial_right)
    texture_left = surface.vol_to_surf(result_img,fsaverage.pial_left)
    # Plot the surface plot
    if hemi == 'right' and showcb:
        cb = True
    else:
        cb = False
    if hemi == 'right':
        f = plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi = 'right',
                                    figure = figure, axes = axes,threshold = None,
                                    colorbar = cb, cmap = 'bwr', vmax = vmax,view = view,
                                    bg_map = fsaverage.sulc_right,bg_on_data = True,
                                    darkness = 0.25, symmetric_cbar = True)

    else:
        f = plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi = 'left',
                                    figure = figure, axes = axes,threshold = None,
                                    colorbar = cb, cmap = 'bwr', vmax = vmax, view = view,
                                    bg_map = fsaverage.sulc_left,bg_on_data = True,
                                    darkness = 0.25, symmetric_cbar = True)
