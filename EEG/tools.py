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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
import pandas as pd

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

# Compute and plot t-test
def t_test(x,y,Vax,ft):
    [statistic, pvalue] = stats.ttest_ind(x,y,permutations  = 100,
                                              random_state = 83)
    dvalue = cohend(x,y)
    # Print results
    print("T-test. statistic = %s, p-value = %s, Cohen's d: %s" % (statistic,
                                                                   pvalue,
                                                                   dvalue))

    # Label
    if pvalue >= 0.01:
        star_label = "p = %.2f d = %.2f" % (pvalue,dvalue)
    elif pvalue >= 0.001 and pvalue < 0.01:
        star_label = "p < 0.01 d = %.2f" % (dvalue)
    else:
        star_label = "p < 0.001 d = %.2f" % (dvalue)

    ylim = Vax.get_ylim()
    Vax.text(0.5,ylim[0] + (ylim[1] - ylim[0])*1.1,star_label,
            fontsize = ft,horizontalalignment = 'center',
            verticalalignment = 'center')

# Calculate and plot the histogram of levels of statistical significance
# using post hoc tests
def plot_tukey(x,Vax):
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
    pvalues = tukey.pvalues
    Vax.hist(pvalues, bins = [0.001,0.01,0.1,1.01], color = 'slategrey')

# Plot the head model to show EEG data
def plot_simple_head_model(ax, radius, pos):
    circle_npts = 100
    head_x = pos + radius * np.cos(np.linspace(0, 2 * np.pi, circle_npts))
    head_y = radius * np.sin(np.linspace(0, 2 * np.pi, circle_npts))
    patches = []

    right_ear = mpatches.FancyBboxPatch([pos + radius + radius / 20, -radius/10],
                                        radius/50, radius/5,
        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))
    patches.append(right_ear)

    left_ear = mpatches.FancyBboxPatch([pos -radius - radius / 20 - radius / 50,
                                        -radius / 10],radius / 50, radius / 5,
        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))
    patches.append(left_ear)

    collection = PatchCollection(patches, facecolor='none',
                                 edgecolor='k', alpha=1.0, lw=2)
    ax.add_collection(collection)
    ax.plot(head_x, head_y, 'k')
    ax.plot([radius])

    # plot nose
    ax.plot([pos -radius / 10, pos, pos + radius  / 10], [radius,
            radius + radius/10, radius], 'k')

# Plot data on EEG electrodes
def plot_EEG(fig,Vax, data, radius, pos, vmin, vmax):
    # Some parameters
    N = 100             # number of points for interpolation
    xy_center = [pos,0]   # center of the plot

    # Coordinates of the EEG electrodes in the 20 montage
    koord = [[pos-0.25*radius,0.8*radius], # "Fp1"
             [pos-0.6*radius,0.45*radius], # "F7"
             [pos+0.6*radius,0.45*radius], # "F8"
             [pos+0.8*radius,0.0], # "T4"
             [pos+0.6*radius,-0.2], # "T6"
             [pos-0.6*radius,-0.2], # "T5"
             [pos-0.8*radius,0.0], # "T3"
             [pos+0.25*radius,0.8*radius], # "Fp2"
             [pos-0.35*radius,-0.8*radius], # "O1"
             [pos-0.3*radius,-0.4*radius], # "P3"
             [pos,-0.4*radius], # "Pz"
             [pos-0.3*radius,0.35*radius], # "F3"
             [pos,0.35*radius], # "Fz"
             [pos+0.3*radius,0.35*radius], # "F4"
             [pos+0.35*radius,0.], # "C4"
             [pos+0.3*radius,-0.4*radius], # "P4"
             [pos,-0.75*radius], # "POz"
             [pos-0.35*radius,0.0], # "C3"
             [pos,0.], # "Cz"
             [pos+0.35*radius,-0.8*radius]] # "O2"


    # external fake electrodes for completing interpolation
    for xx in np.linspace(pos-radius,pos+radius,50):
        koord.append([xx,np.sqrt(radius**2 - (xx)**2)])
        koord.append([xx,-np.sqrt(radius**2 - (xx)**2)])
        data.append(0)
        data.append(0)

    x,y = [],[]
    for i in koord:
        x.append(i[0])
        y.append(i[1])
    z = data

    xi = np.linspace(-radius, radius, N)
    yi = np.linspace(-radius, radius, N)
    zi = scipy.interpolate.griddata((np.array(x), np.array(y)), z,
                                    (xi[None,:], yi[:,None]), method='cubic')

    # # set points > radius to not-a-number. They will not be plotted.
    # # the dr/2 makes the edges a bit smoother
    # dr = xi[1] - xi[0]
    # for i in range(N):
    #     for j in range(N):
    #         r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
    #         if (r - dr/2) > radius:
    #             zi[j,i] = "nan"

    # Use different number of levels for the fill and the lines
    CS = Vax.contourf(xi, yi, zi, 30, cmap = plt.cm.bwr, zorder = 1,
                      vmin = vmin,vmax = vmax)
    Vax.contour(xi, yi, zi, 5, colors = "grey", zorder = 2,linewidths = 0.5,
                vmin = vmin,vmax = vmax)

    # Make a color bar
    # cbar = fig.colorbar(CS, ax=Vax)
    fig.colorbar(ScalarMappable(norm=CS.norm, cmap=CS.cmap), ax = Vax)

    # Add the EEG electrode positions
    Vax.scatter(x[:20], y[:20], marker = 'o', c = 'k', s = 2, zorder = 3)
