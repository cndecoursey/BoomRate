#!/usr/bin/env python
import os,sys,pdb,scipy,pickle,json
from pylab import *
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime, time
from matplotlib import dates as mdates
import util as u
import cosmocalc
import volume
import glob
from pprint import pprint
from scipy.interpolate import interp1d as scipy_interp1d

import multiprocessing
from functools import partial

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

# -----------------------------------------------------------------------------------------------------

def plot_phase_color_correction(rest_age, color_cor_array, color_cor_raw,
                                 best_rest_filter, redshift, type,
                                 observed_filter, diag_dir):
    """
    Plots the phase-dependent color correction M_X - M_Q as a function
    of rest-frame age, showing both the raw sparse computed values and
    the interpolated/anchored curve used in the control time calculation.

    Parameters
    ----------
    rest_age : numpy.ndarray
        1D array of rest-frame ages in days.
    color_cor_array : numpy.ndarray
        1D array of shape (N_ages,) — the final interpolated and anchored
        color correction used in the control time calculation.
    color_cor_raw : numpy.ndarray
        1D array of shape (N_ages,) — the raw sparse color corrections
        before interpolation, with NaN where no SED data exists.
    best_rest_filter : int
        Central wavelength in nm of the best matching rest-frame SDSS
        filter (X).
    redshift : float
        Redshift of the current bin.
    type : list of str
        SN subtype list (e.g. ['iip']).
    observed_filter : str
        Path to the observed JWST filter file, used for the plot label.
    diag_dir : str
        Path to the diagnostic plots directory.
    """
    fig, (ax1, ax2) = subplots(1, 2, figsize=(14, 6))

    jwst_name = os.path.basename(observed_filter).split('.')[0]

    # --- left panel: raw sparse points and interpolated curve ---
    # raw sparse computed values
    raw_valid = np.isfinite(color_cor_raw)
    ax1.scatter(rest_age[raw_valid], color_cor_raw[raw_valid],
                color='steelblue', s=25, zorder=5, alpha=0.8,
                label='Computed (median across templates)')

    # interpolated/anchored curve
    ax1.plot(rest_age, color_cor_array,
             'r-', lw=2, label='Interpolated + anchored to 0')

    # reference lines
    ax1.axhline(0., color='gray', ls='--', lw=1, alpha=0.5)
    ax1.axvline(0., color='gray', ls='--', lw=1, alpha=0.5,
                label='Peak (age=0)')

    # mark the last valid computed point (where anchoring is applied)
    if np.sum(raw_valid) > 0:
        last_valid_age = rest_age[np.where(raw_valid)[0][-1]]
        ax1.axvline(last_valid_age, color='orange', ls=':', lw=1.5,
                    alpha=0.7, label='Last valid point (anchored to 0)')

    ax1.set_xlabel('Rest-frame age (days)', fontsize=11)
    ax1.set_ylabel('M_X - M_Q color correction (mag)', fontsize=11)
    ax1.set_xlim(-50, 200)
    ax1.legend(fontsize=8)
    ax1.set_title('%s   M_%dnm - M_%s(rest)\nz=%.2f' % (
                  '_'.join(type), best_rest_filter, jwst_name, redshift),
                  fontsize=10)
    ax1.grid(alpha=0.3)

    # --- right panel: histogram of raw computed color corrections ---
    raw_colors = color_cor_raw[raw_valid]
    if len(raw_colors) > 0:
        ax2.hist(raw_colors, bins=max(5, len(raw_colors)//3),
                 color='steelblue', alpha=0.7, edgecolor='k')
        ax2.axvline(np.nanmedian(raw_colors), color='red', ls='--', lw=2,
                    label='Median = %.3f mag' % np.nanmedian(raw_colors))
        ax2.axvline(np.nanmean(raw_colors), color='orange', ls='-.', lw=2,
                    label='Mean = %.3f mag' % np.nanmean(raw_colors))
        ax2.axvline(0., color='gray', ls=':', lw=1, alpha=0.5)
    ax2.set_xlabel('M_X - M_Q color correction (mag)', fontsize=11)
    ax2.set_ylabel('N age steps', fontsize=11)
    ax2.set_title('Distribution of computed color corrections\n'
                  '(before interpolation)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    tight_layout()
    savefig('%s/%s_%s_%dnm_z%.2f_color_correction.png' % (
            diag_dir, type, jwst_name,
            best_rest_filter, redshift),
            bbox_inches='tight')
    clf()

# -----------------------------------------------------------------------------------------------------

def plot_lightcurve_stages(rest_age, observed_frame_lightcurve,
                           Mg_to_Mx_color, Mx_to_MQ_color,
                           mu, da, dm, sens, type, redshift,
                           best_rest_filter, observed_filter, ofilter_cen,
                           diag_dir):
    """
    Plots the representative light curve at five stages of transformation
    following the updated BoomRate workflow:

    (1) Rest-frame g-band absolute magnitude + dm (Richardson+2014 anchor)
    (2) After Mg->Mx color correction (rest-frame X-band absolute magnitude)
    (3) After phase-dependent Mx->MQ color correction (rest-frame Q-band)
    (4) After host galaxy extinction (rest-frame Q-band + extinction)
    (5) After distance modulus + K-correction bandwidth term
        (final observed JWST apparent magnitude)

    Parameters
    ----------
    rest_age : numpy.ndarray
        1D array of rest-frame ages in days.
    observed_frame_lightcurve : numpy.ndarray
        2D array of shape (N_ages, 5) — the anchored representative
        light curve from mean_pop(), column 0 is the median g-band
        absolute magnitude anchored to Richardson+2014.
    Mg_to_Mx_color : float
        Peak color correction from g-band to best-matching SDSS rest-frame
        filter X (from peak_color_corrections.txt). Applied as a constant
        scalar offset across all phases.
    Mx_to_MQ_color : numpy.ndarray
        1D array of shape (N_ages,) — phase-dependent color correction
        from rest-frame X-band to rest-frame Q-band (JWST filter blueshifted
        to rest frame). Computed from SED templates, interpolated, and
        anchored to 0 at late phases.
    mu : float
        Distance modulus at this redshift.
    da : float
        Host galaxy extinction in Q-band magnitudes (AQ).
    dm : float
        Luminosity function offset in magnitudes.
    sens : float
        Survey magnitude limit (50% completeness magnitude).
    type : str
        SN subtype label (e.g. 'iip').
    redshift : float
        Redshift of the current bin.
    best_rest_filter : int
        Central wavelength in nm of the best matching rest-frame SDSS
        filter X.
    observed_filter : str
        Path to the observed JWST filter file, used for the plot label.
    ofilter_cen : float
        Central wavelength corresponding to your observed JWST filter
    diag_dir : str
        Path to the diagnostic plots directory.
    """
    jwst_name    = os.path.basename(observed_filter).split('.')[0]
    peak_age_idx = np.argmin(np.abs(rest_age))

    fig, axes = subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    # base light curve — g-band absolute magnitude
    lc    = observed_frame_lightcurve[:, 0]
    valid = lc < 50.
    sig   = observed_frame_lightcurve[:, 1]

    # stage 1: g-band absolute magnitude + dm
    stage1 = lc + dm
    # stage 2: after Mg->Mx peak color correction (X-band absolute)
    stage2 = lc + dm + Mg_to_Mx_color
    # stage 3: after Mx->MQ phase-dependent color correction (Q-band absolute)
    stage3 = lc + dm + Mg_to_Mx_color + Mx_to_MQ_color
    # stage 4: after host galaxy extinction (Q-band absolute + extinction)
    stage4 = lc + dm + Mg_to_Mx_color + Mx_to_MQ_color + da
    # stage 5: after distance modulus + K-correction bandwidth term
    stage5 = lc + dm + Mg_to_Mx_color + Mx_to_MQ_color + da + mu - 2.5*np.log10(1+redshift)

    stages  = [stage1, stage2, stage3, stage4, stage5]

    # rest-frame central wavelength of the JWST filter at this redshift
    jwst_restframe_cen = ofilter_cen / (1 + redshift)
    
    titles = [
        '(1) g-band absolute magnitude + dm',
        '(2) After Mg\u2192Mx color correction\n(g \u2192 %d nm SDSS)' % best_rest_filter,
        '(3) After Mx\u2192MQ color correction\n(%d nm SDSS \u2192 %s rest-frame (%.0f nm))' % (
            best_rest_filter, jwst_name, jwst_restframe_cen),
        '(4) After host galaxy extinction',
        '(5) After distance modulus + K-correction\n(final %s apparent magnitude)' % jwst_name,
    ]
    ylabels = [
        'Absolute magnitude (g-band)',
        'Absolute magnitude (X-band)',
        'Absolute magnitude (Q-band)',
        'Absolute magnitude (Q-band + extinction)',
        'Apparent magnitude',
    ]

    panel_params = [
        # panel 1: g-band + dm
        [
            'SN type: %s' % type,
            'z = %.3f' % redshift,
            'dm = %+.2f mag' % dm,
            'Mg_peak = %.2f mag' % (min(lc[valid]) + dm),
        ],
        # panel 2: Mg->Mx color correction
        [
            'Mg\u2192Mx = %+.3f mag' % Mg_to_Mx_color,
            'dm = %+.2f mag' % dm,
            'Mx_peak = %.2f mag' % (min(lc[valid]) + dm + Mg_to_Mx_color),
            '(constant peak color offset)',
        ],
        # panel 3: Mx->MQ phase-dependent
        [
            'Mx\u2192MQ at peak = %+.3f mag' % Mx_to_MQ_color[peak_age_idx],
            'Mx\u2192MQ range = [%.3f, %.3f] mag' % (
                np.nanmin(Mx_to_MQ_color[valid]),
                np.nanmax(Mx_to_MQ_color[valid])),
            '(phase-dependent correction)',
        ],
        # panel 4: host galaxy extinction
        [
            'da (AQ) = %.2f mag' % da,
            'MQ_peak = %.2f mag' % (min(stage3[valid & np.isfinite(stage3)])),
            'MQ_ext_peak = %.2f mag' % (min(stage4[valid & np.isfinite(stage4)])),
        ],
        # panel 5: distance modulus + K-correction
        [
            'mu = %.2f mag' % mu,
            '-2.5*log10(1+z) = %.3f mag' % (-2.5*np.log10(1+redshift)),
            'z = %.3f' % redshift,
            'survey limit = %.1f mag' % sens,
            'peak apparent mag = %.2f mag' % (
                min(stage5[valid & np.isfinite(stage5)])),
        ],
    ]

    for ax, stage, title, ylabel, params in zip(
            axes[:5], stages, titles, ylabels, panel_params):

        valid_stage = valid & np.isfinite(stage)

        # plot median light curve
        ax.plot(rest_age[valid_stage], stage[valid_stage],
                'k-', lw=2, label='Median light curve')

        # 1-sigma shading
        ax.fill_between(rest_age[valid_stage],
                        stage[valid_stage] - sig[valid_stage],
                        stage[valid_stage] + sig[valid_stage],
                        alpha=0.2, color='blue', label='1-sigma spread')

        # survey limit and detectable region for final apparent magnitude panel only
        if 'apparent' in ylabel.lower():
            ax.axhline(sens, color='red', ls='--', lw=1.5,
                       label='Survey limit = %.1f mag' % sens)
            detectable = valid_stage & (stage < sens)
            if sum(detectable) > 0:
                ax.fill_between(rest_age, sens, stage,
                                where=detectable,
                                alpha=0.15, color='green',
                                label='Detectable region')

        # reference line at age=0
        ax.axvline(0., color='gray', ls=':', lw=1, alpha=0.5)

        # parameter text box
        param_text = '\n'.join(params)
        ax.text(0.97, 0.97, param_text,
                transform=ax.transAxes,
                fontsize=7.5,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Rest-frame age (days)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-50, 200)
        ax.invert_yaxis()
        if 'apparent' in ylabel.lower():
            ax.set_ylim(32, 24)
        else:
            ax.set_ylim(-14, -21)
        ax.legend(fontsize=7, loc='lower right')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)

    # hide the unused 6th panel
    axes[5].set_visible(False)

    fig.suptitle('%s at z=%.2f  (dm=%.2f, da=%.2f)' % (
                 type, redshift, dm, da),
                 fontsize=12, y=1.01)
    tight_layout()
    savefig('%s/%s_%s_z%.2f_lc_stages.png' % (
            diag_dir, type, jwst_name, redshift),
            bbox_inches='tight')
    clf()

# -----------------------------------------------------------------------------------------------------

def plot_restframe_lightcurves(rflc, rest_age, models_used, type, diag_dir):
    
    # Diagnostic plots showing used_model rest-frame light curves in all available rest-frame filters
    for k, v in rflc.items():
        clf()
        fig, ax = subplots(1, 1, figsize=(12, 7))
        v = array(v)
        colors = [cm.viridis(i/max(len(v)-1, 1)) for i in range(len(v))]
        for ii in range(len(v)):
            ax.plot(rest_age, v[ii], color=colors[ii], lw=1.0, alpha=0.7,
                    label='%s' % models_used[ii])
        ax.set_xlim(-50, 200)
        # set y limits based on physically meaningful values
        physical = v[v < 50.]
        if len(physical) > 0:
            ax.set_ylim(nanmax(physical)+2, nanmin(physical)-2)
        else:
            ax.set_ylim(-10, -20)
        ax.set_xlabel('Rest-frame age (days)')
        ax.set_ylabel('Absolute magnitude')
        ax.set_title('%s individual template light curves, filter = %d nm' % (type, k))
        ax.legend(fontsize=10, ncol=2, loc='lower right',
                  framealpha=0.5, bbox_to_anchor=(1.0, 0.0))
        tight_layout()
        savefig('%s/%s_%dnm_individual_templates.png' % (diag_dir, type, k))
        clf()

# -----------------------------------------------------------------------------------------------------

def plot_anchoring_diagnostic(lc_smoothed, lc_normalized, lc_anchored, rest_age, type, diag_dir, absmags):
    """
    Plots the representative light curve at three stages of the anchoring
    process: after smoothing, after normalization to peak=0, and after
    anchoring to the Richardson 2014 mean peak absolute magnitude.

    Parameters
    ----------
    lc_smoothed: numpy.ndarray
        2D array of shape (N_ages, 5) after Gaussian smoothing but before
        any magnitude anchoring.
    lc_normalized: numpy.ndarray
        2D array of shape (N_ages, 5) after subtracting template_peak,
        so that the peak magnitude is at 0.0.
    lc_anchored: numpy.ndarray
        2D array of shape (N_ages, 5) after adding absmags[type][0],
        anchored to the Richardson 2014 mean peak absolute magnitude.
    rest_age : numpy.ndarray
        1D array of rest-frame ages in days.
    type : str
        SN subtype label used for plot title and filename (e.g. 'iip').
    diag_dir : str
        'diagnostic_plots/<run_name>/', used for output filename.
    absmags : dict
        Dictionary of mean peak absolute magnitudes, keyed by SN subtype.
    """
    fig, ax = subplots(1, 1, figsize=(10, 14))

    # plot all three light curves, column 0 only
    ax.plot(rest_age, lc_smoothed[:,0], 'b-', lw=1.5, alpha=0.7,
            label='Before anchoring')
    ax.plot(rest_age, lc_normalized[:,0], 'g-', lw=1.5, alpha=0.7,
            label='Normalized to peak = 0.0 mag')
    ax.plot(rest_age, lc_anchored[:,0], 'r-', lw=2,
            label='Anchored to peak = %.2f mag' % absmags[type][0])

    # reference lines
    ax.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.5, label='0.0 mag reference')
    ax.axhline(absmags[type][0], color='red', ls=':', lw=1, alpha=0.5,
               label='peak = %.2f mag' % absmags[type][0])

    ax.set_xlabel('Rest-frame age (days)')
    ax.set_ylabel('Absolute magnitude')
    ax.set_xlim(-50, 200)

    # set y limits to accommodate all three light curves including peak=0
    all_physical = concatenate([
        lc_smoothed[:,0][lc_smoothed[:,0] < 50.],
        lc_normalized[:,0][lc_normalized[:,0] < 50.],
        lc_anchored[:,0][lc_anchored[:,0] < 50.]
    ])
    if len(all_physical) > 0:
        ax.set_ylim(nanmax(all_physical)+2, nanmin(all_physical)-2)

    ax.legend(fontsize=9)
    ax.set_title('%s light curve anchoring diagnostic' % type)
    tight_layout()
    savefig('%s/%s_anchoring_diagnostic.png' % (diag_dir, type))
    clf()

# -----------------------------------------------------------------------------------------------------

def plot_smoothing_diagnostic(observed_frame_lightcurve_unsmoothed, observed_frame_lightcurve_smoothed, rest_age, type, diag_dir, dstep):
    """
    Plots the representative light curve before and after Gaussian smoothing

    Parameters
    ----------
    observed_frame_lightcurve_unsmoothed: numpy.ndarray
        2D array of shape (N_ages, 5) before smoothing is applied.
    observed_frame_lightcurve_smoothed: numpy.ndarray
        2D array of shape (N_ages, 5) after smoothing is applied.
    rest_age: numpy.ndarray
        1D array of rest-frame ages in days.
    type: str
        SN subtype label used for plot title and filename.
    diag_dir: str
        '/diagnostic_plots/<run_name>/', used for output filename.
    dstep: float
        Step size in days, used as the Gaussian kernel standard deviation.
    """
    fig, ax = subplots(1, 1, figsize=(10, 7))

    ax.plot(rest_age, observed_frame_lightcurve_unsmoothed[:,0], 
            'b-', lw=1.5, alpha=0.7, label='Before smoothing')
    ax.plot(rest_age, observed_frame_lightcurve_smoothed[:,0],
            'r-', lw=2, label='After smoothing (Gaussian, sigma=%d days)' %(dstep*dstep))

    ax.set_xlabel('Rest-frame age (days)')
    ax.set_ylabel('Absolute magnitude')
    ax.set_xlim(-50, 200)

    # Set y limits based on physically meaningful values
    physical = observed_frame_lightcurve_unsmoothed[:,0]
    physical = physical[physical < 50.]
    if len(physical) > 0:
        ax.set_ylim(nanmax(physical)+2, nanmin(physical)-2)

    ax.legend(fontsize=9)
    ax.set_title('%s light curve: before vs after Gaussian smoothing' % type)
    tight_layout()
    savefig('%s/%s_smoothing_diagnostic.png' % (diag_dir, type))
    clf()

# -----------------------------------------------------------------------------------------------------

def plot_redshift_dist(redshift_tmp, rv, dzza, redshifts, ng, sntypes, diag_dir):
    """
    Plots the smoothed redshift probability distribution and cumulative distribution
    with redshift bin boundaries and observed SN counts per bin.
    
    Parameters
    ----------
    redshift_tmp: array 
        Redshift grid from 0 to 10 in steps of dzza
    rv: array
        Summed redshift probability distribution across all SNe
    dzza: float
        Redshift grid step size
    redshifts: array
        Bin edges
    ng: array
        Observed number of SNe in each bin
    sntypes: list
        List of SN types being considered
    run_name: str
        Name of the current run, used for the output filename
    """

    smooth_rv = convolve(rv*dzza, Gaussian1DKernel(10), boundary='extend')

    fig, (ax1, ax2) = subplots(1, 2, figsize=(14, 5))

    # Left panel: redshift distribution 
    ax1.plot(redshift_tmp, smooth_rv, 'k-')
    ax1.set_xlabel('Redshift')
    ax1.set_xlim(0, 6)
    if 'ia' in sntypes:
        ax1.set_ylabel('Number of SNeIa')
    else:
        ax1.set_ylabel('Number of CCSNe')

    # Shade each redshift bin and label with observed counts
    colors = ['#d0e8f0', '#f0d8c0', '#d0f0d0', '#f0d0e8', '#e8f0d0']
    for i in range(len(redshifts)-1):
        ax1.axvspan(redshifts[i], redshifts[i+1], alpha=0.3, color=colors[i % len(colors)],
                    label='z=%.2f-%.2f, N=%.1f' %(redshifts[i], redshifts[i+1], ng[i]))
    ax1.legend(loc=1, fontsize=9, title='Total N=%.1f' %sum(ng), title_fontsize=10)

    # Right panel: cumulative distribution 
    ax2.plot(redshift_tmp, cumsum(rv)*dzza, 'k-', label='%.1f Total' %(cumsum(rv)[-1]*dzza))
    ax2.set_xlabel('Redshift')
    ax2.set_xlim(0, 6)
    if 'ia' in sntypes:
        ax2.set_ylabel('Cumulative number of SNeIa')
    else:
        ax2.set_ylabel('Cumulative number of CCSNe')

    # Show bin boundaries as vertical lines
    for z in redshifts:
        ax2.axvline(z, color='gray', ls='--', lw=1)
    ax2.legend(loc=2, fontsize=9)

    tight_layout()
    savefig('%s/redshift_dist.png' % diag_dir)
    clf()

# -----------------------------------------------------------------------------------------------------

def plot_tc_per_type(tc_per_type, z_low, z_high, diag_dir, mag=None):
    """
    Per-z-bin diagnostic: stacked single bar showing each subtype's control time
    """
    if not tc_per_type:
        return
    types = list(tc_per_type.keys())
    contributions = array([tc_per_type[t] for t in types]) * 365.25  # rest-frame days
    total = contributions.sum()
    palette = cm.tab10(linspace(0, 1, max(len(types), 3)))

    fig, ax = subplots(figsize=(5, 6))
    bottom = 0.0
    for t_idx, t in enumerate(types):
        h = contributions[t_idx]
        ax.bar(0, h, 0.6, bottom=bottom,
               color=palette[t_idx % len(palette)], edgecolor='white', linewidth=0.5,
               label='%s  %.2f d' % (t.upper(), h))
        bottom += h

    ax.set_xticks([0])
    ax.set_xticklabels(['z=%.2f-%.2f' % (z_low, z_high)])
    ax.set_ylabel('Control time (rest-frame days)')
    title = 'tc_tot = %.2f d' % total
    if mag is not None:
        title += '  | m50 %.1f' % mag
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='Subtype contribution', frameon=False)
    ax.margins(x=0.4)
    tight_layout()

    if not os.path.isdir('diagnostic_plots'):
        os.makedirs('diagnostic_plots')
    savefig('%s/tc_per_type_z%.2f-%.2f.png' % (diag_dir, z_low, z_high),
            bbox_inches='tight')
    clf()
    close('all')