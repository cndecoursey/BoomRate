#!/usr/bin/env python
"""
compute_peak_colors.py

Computes the synthetic color between SDSS g-band (AB) and each other SDSS
filter (u, r, i, z) at peak for each SN subtype, using the same SED pkl
files used in control_time.run(). These color corrections are needed to
anchor the non-g-band rest-frame light curves to the same absolute magnitude
scale as the g-band.

The color correction for each filter X is defined as:
    color_cor[X] = M_g(AB) - M_X(AB)

So that:
    M_X(AB) = M_g(AB) - color_cor[X]

Usage:
    python compute_peak_colors.py

Output:
    Prints updated color_cor_gen dictionary for use in control_time.py
    Saves diagnostic plots to diagnostic_plots/
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------
# CONFIGURATION -- update these paths for your machine
# -----------------------------------------------------------------------
sndata_root = '/Users/christadecoursey/Documents/SNANA/SNANA_2025'
base_root   = '/Users/christadecoursey/Documents/JADES/SN_Classification_and_Rates_Paper/classification/BoomRate'
pkl_dir     = base_root   # directory containing SEDs_*.pkl files
diag_dir    = base_root + '/diagnostic_plots'
# -----------------------------------------------------------------------

# SN type to pkl file mapping
pkl_files = {
    'iip': 'SEDs_iip.pkl',
    'iil': 'SEDs_iil.pkl',
    'iin': 'SEDs_iin.pkl',
    'ib':  'SEDs_ib.pkl',
    'ic':  'SEDs_ic.pkl',
    'ia':  'SEDs_ia.pkl',
}

# SDSS filters to compute colors for
sdss_filter_names = ['u', 'g', 'r', 'i', 'z']

# Age tolerance for "near peak" in days
peak_age_tolerance = 3.0

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def load_filter(filter_path):
    """Load a filter transmission file. Returns (wavelength_AA, transmission)."""
    data = np.loadtxt(filter_path)
    return data[:, 0], data[:, 1]


def synthetic_mag_AB(wave_sed, flux_sed, wave_filt, trans_filt):
    """
    Computes a synthetic AB magnitude by integrating an SED through a filter.

    m_AB = -2.5*log10( int(F_lambda * T * dlambda) /
                        int(T * c/lambda^2 * dlambda) ) - 48.6

    Parameters
    ----------
    wave_sed : array
        SED wavelengths in Angstroms (must be > 100 AA, sorted ascending).
    flux_sed : array
        SED f_lambda flux.
    wave_filt : array
        Filter wavelengths in Angstroms.
    trans_filt : array
        Filter transmission (0 to 1).

    Returns
    -------
    mag : float
        Synthetic AB magnitude, or NaN if integration fails.
    """
    # interpolate filter onto SED wavelength grid
    f_interp = interp1d(wave_filt, trans_filt, bounds_error=False, fill_value=0.0)
    trans_on_sed = f_interp(wave_sed)
    trans_on_sed = np.clip(trans_on_sed, 0., None)

    dlambda_sed = np.gradient(wave_sed)
    c = 2.998e18  # speed of light in Angstroms/s

    # numerator: int(F_lambda * T * dlambda)
    numerator = np.sum(flux_sed * trans_on_sed * dlambda_sed)

    # denominator: int(T * c/lambda^2 * dlambda)
    denominator = np.sum(trans_on_sed * c / wave_sed**2 * dlambda_sed)

    if numerator <= 0. or denominator <= 0.:
        return np.nan

    return -2.5 * np.log10(numerator / denominator) - 48.6


def get_peak_spectra(pkl_path, peak_age_tolerance=3.0):
    """
    Loads a SED pkl file and extracts all spectra near peak (age ~ 0).

    Parameters
    ----------
    pkl_path : str
        Path to the SED pkl file.
    peak_age_tolerance : float
        Maximum age offset from 0 to be considered near peak, in days.

    Returns
    -------
    peak_spectra : list of (wave, flux) tuples
        List of (wavelength, flux) arrays for each model near peak.
        Wavelengths are sorted ascending and cleaned of unphysical values.
    model_names : list of str
        Names of models contributing peak spectra.
    """
    with open(pkl_path, 'rb') as f:
        models_dict = pickle.load(f, encoding='latin1')

    peak_spectra = []
    model_names  = []

    for model_name, spec in models_dict.items():
        # find rows near peak age
        idx = np.where(np.abs(spec[:, 0]) <= peak_age_tolerance)[0]
        if len(idx) == 0:
            continue
        if np.sum(spec[idx, 2]) == 0.:
            continue

        wave = spec[idx, 1]
        flux = spec[idx, 2]

        # if multiple age steps within tolerance, use the one closest to 0
        if len(np.unique(spec[idx, 0])) > 1:
            unique_ages = np.unique(spec[idx, 0])
            best_age = unique_ages[np.argmin(np.abs(unique_ages))]
            idx  = np.where(spec[:, 0] == best_age)[0]
            wave = spec[idx, 1]
            flux = spec[idx, 2]

        # sort by wavelength ascending
        sort_idx = np.argsort(wave)
        wave = wave[sort_idx]
        flux = flux[sort_idx]

        # remove unphysical zero or near-zero wavelengths
        valid_wave = wave > 100.
        wave = wave[valid_wave]
        flux = flux[valid_wave]

        if len(wave) < 2 or np.sum(flux) <= 0.:
            continue

        peak_spectra.append((wave, flux))
        model_names.append(model_name)

    return peak_spectra, model_names


def plot_color_distribution(sntype, filter_names, all_colors, central_waves, diag_dir):
    """
    Plots the distribution of (g - X) colors across templates for each
    filter, for a given SN subtype.

    Parameters
    ----------
    sntype : str
        SN subtype label.
    filter_names : list of str
        Filter names (e.g. ['u', 'r', 'i', 'z']).
    all_colors : dict
        Dictionary mapping filter name to array of (g-X) colors across
        templates.
    central_waves : dict
        Dictionary mapping filter name to central wavelength in nm.
    diag_dir : str
        Path to diagnostic plots directory.
    """
    n_filters = len(filter_names)
    fig, axes = plt.subplots(1, n_filters, figsize=(5*n_filters, 6))
    if n_filters == 1:
        axes = [axes]

    for ax, fname in zip(axes, filter_names):
        colors = np.array(all_colors[fname])
        valid  = np.isfinite(colors)

        if np.sum(valid) == 0:
            ax.text(0.5, 0.5, 'No valid measurements',
                    transform=ax.transAxes, ha='center', va='center')
        elif np.sum(valid) == 1:
            ax.axvline(colors[valid][0], color='steelblue', lw=3,
                       label='Single template\n= %.3f mag' % colors[valid][0])
        else:
            ax.hist(colors[valid], bins=max(3, np.sum(valid)//3),
                    color='steelblue', alpha=0.7, edgecolor='k')
            ax.axvline(np.median(colors[valid]), color='red', ls='--', lw=2,
                       label='Median = %.3f mag' % np.median(colors[valid]))
            ax.axvline(np.mean(colors[valid]), color='orange', ls='-.', lw=2,
                       label='Mean = %.3f mag' % np.mean(colors[valid]))

        ax.set_xlabel('(g - %s) AB color at peak (mag)' % fname, fontsize=11)
        ax.set_ylabel('N templates', fontsize=11)
        ax.set_title('%s: g - %s\n(%d nm, N=%d templates)' % (
                     sntype.upper(), fname,
                     central_waves.get(fname, 0),
                     np.sum(valid)), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('%s peak colors relative to SDSS g-band (AB)' % sntype.upper(),
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('%s/peak_color_diagnostic/%s_peak_colors_diagnostic.png' % (diag_dir, sntype),
                bbox_inches='tight')
    plt.clf()
    print('  Diagnostic plot saved to %s/peak_color_diagnostic/%s_peak_colors_diagnostic.png' % (
          diag_dir, sntype))

def write_color_corrections(all_color_corrections, central_waves, 
                            sdss_filter_names, output_path):
    """
    Writes per-subtype peak color corrections relative to SDSS g-band (AB)
    to a text file for use in BoomRate config files.

    File format:
        # comment lines starting with #
        sntype  filter_central_wave_nm  color_correction_mag
        e.g.:
        iip  356  0.45
        iip  472  0.00
        iip  619  -0.31
        ...

    Parameters
    ----------
    all_color_corrections : dict
        Dictionary mapping sntype to dict of {filter_name: color_correction}.
    central_waves : dict
        Dictionary mapping filter name to central wavelength in nm.
    sdss_filter_names : list of str
        List of SDSS filter names in order.
    output_path : str
        Path to the output text file.
    """
    with open(output_path, 'w') as f:
        f.write('# Peak color corrections relative to SDSS g-band (AB)\n')
        f.write('# Generated by compute_peak_colors.py\n')
        f.write('# Convention: color_cor = M_X(AB) - M_g(AB)\n')
        f.write('# Usage: M_X(AB) = M_g(AB) + color_cor\n')
        f.write('# g-band is the reference filter with color_cor = 0.0\n')
        f.write('# Positive value means X is fainter than g (bluer source)\n')
        f.write('# Negative value means X is brighter than g (redder source)\n')
        f.write('#\n')

        for sntype, corrections in all_color_corrections.items():
            f.write('# --- %s ---\n' % sntype.upper())
            for fname in sdss_filter_names:
                cen_wave = central_waves[fname]
                if fname == 'g':
                    f.write('%s  %d  0.000\n' % (sntype, cen_wave))
                else:
                    color = corrections.get(fname, np.nan)
                    if np.isfinite(color):
                        f.write('%s  %d  %.3f\n' % (sntype, cen_wave, color))
                    else:
                        f.write('%s  %d  nan\n' % (sntype, cen_wave))
            f.write('\n')

    print('\nColor corrections written to %s' % output_path)


# -----------------------------------------------------------------------
# LOAD SDSS FILTERS
# -----------------------------------------------------------------------

print('Loading SDSS filters...')
sdss_filters = {}
central_waves = {}
for fname in sdss_filter_names:
    fpath = '%s/filters/SDSS/SDSS_web2001/%s.dat' % (sndata_root, fname)
    wave_f, trans_f = load_filter(fpath)
    sdss_filters[fname] = (wave_f, trans_f)
    # compute central wavelength in nm
    cen = np.sum(wave_f * trans_f * wave_f) / np.sum(trans_f * wave_f) / 10.
    central_waves[fname] = int(cen + 0.5)
    print('  SDSS %s: central wavelength = %d nm' % (fname, central_waves[fname]))

# -----------------------------------------------------------------------
# MAIN COMPUTATION
# -----------------------------------------------------------------------

# store color corrections per subtype: color_cor[sntype][filter] = median (g-X) color
all_color_corrections = {}

print('\n' + '='*60)
print('Computing peak colors relative to SDSS g-band (AB)...')
print('='*60)

for sntype, pkl_file in pkl_files.items():
    pkl_path = os.path.join(pkl_dir, pkl_file)

    if not os.path.isfile(pkl_path):
        print('\nWARNING: %s not found, skipping %s' % (pkl_path, sntype))
        continue

    print('\nProcessing %s...' % sntype.upper())

    # load peak spectra from pkl
    peak_spectra, model_names = get_peak_spectra(pkl_path, peak_age_tolerance)
    print('  Found %d templates with spectra near peak' % len(peak_spectra))

    if len(peak_spectra) == 0:
        print('  WARNING: no peak spectra found, skipping')
        continue

    # compute synthetic AB magnitude in each SDSS filter for each template
    all_mags = {fname: [] for fname in sdss_filter_names}

    for i, (wave, flux) in enumerate(peak_spectra):
        for fname in sdss_filter_names:
            wave_f, trans_f = sdss_filters[fname]
            mag = synthetic_mag_AB(wave, flux, wave_f, trans_f)
            all_mags[fname].append(mag)

    # convert to arrays
    for fname in sdss_filter_names:
        all_mags[fname] = np.array(all_mags[fname])

    # compute (g - X) color for each non-g filter
    all_colors = {}
    color_corrections = {}

    for fname in sdss_filter_names:
        if fname == 'g':
            continue
        # (g - X) = M_g - M_X
        g_mags = all_mags['g']
        x_mags = all_mags[fname]
        colors = x_mags - g_mags  # X - g
        all_colors[fname] = colors

        valid = np.isfinite(colors)
        if np.sum(valid) == 0:
            print('  WARNING: no valid (g-%s) colors' % fname)
            color_corrections[fname] = np.nan
        elif np.sum(valid) == 1:
            print('  WARNING: only 1 template for (g-%s) color' % fname)
            color_corrections[fname] = colors[valid][0]
        else:
            median_color = np.median(colors[valid])
            std_color    = np.std(colors[valid])
            color_corrections[fname] = median_color
            print('  (g - %s) = %.3f +/- %.3f mag (N=%d templates)' % (
                  fname, median_color, std_color, np.sum(valid)))

    all_color_corrections[sntype] = color_corrections

    # diagnostic plot for non-g filters
    non_g_filters = [f for f in sdss_filter_names if f != 'g']
    plot_color_distribution(sntype, non_g_filters, all_colors,
                            central_waves, diag_dir)

# -----------------------------------------------------------------------
# PRINT FINAL RESULTS
# -----------------------------------------------------------------------

print('\n' + '='*60)
print('RESULTS: peak color corrections relative to SDSS g-band (AB)')
print('='*60)

print('\nPer subtype color corrections (g - X) at peak:')
print('(Add these to M_g(AB) to get M_X(AB) for each filter)\n')

# print per-subtype results
for sntype, corrections in all_color_corrections.items():
    print('%s:' % sntype.upper())
    for fname, color in corrections.items():
        print('  g - %s = %+.3f mag  (central wave: %d nm)' % (
              fname, color, central_waves[fname]))

# compute mean color correction across all subtypes
print('\nMean color corrections across all CC subtypes:')
cc_types = ['iip', 'iil', 'iin', 'ib', 'ic']
mean_corrections = {}
for fname in sdss_filter_names:
    if fname == 'g':
        continue
    colors = [all_color_corrections[t][fname]
              for t in cc_types
              if t in all_color_corrections and
              np.isfinite(all_color_corrections[t][fname])]
    if len(colors) > 0:
        mean_corrections[fname] = np.mean(colors)
        print('  g - %s = %+.3f mag' % (fname, mean_corrections[fname]))

# print color_cor_gen dictionary for direct use in control_time.py
print('\nUpdated color_cor_gen dictionary for control_time.py:')
print('color_cor_gen = {')
for fname in sdss_filter_names:
    cen_wave = central_waves[fname]
    if fname == 'g':
        # g-band is the reference, color correction = 0.0
        print("    %d: 0.0,   # SDSS g (reference)" % cen_wave)
    else:
        # average across CC subtypes
        colors = [all_color_corrections[t][fname]
                  for t in cc_types
                  if t in all_color_corrections and
                  np.isfinite(all_color_corrections[t][fname])]
        if len(colors) > 0:
            mean_color = np.mean(colors)
            print("    %d: %.2f,  # SDSS %s (g - %s)" % (
                  cen_wave, mean_color, fname, fname))
        else:
            print("    %d: nan,  # SDSS %s (no valid templates)" % (
                  cen_wave, fname))
print('}')

print('\nPer-subtype color_cor dictionaries:')
for sntype, corrections in all_color_corrections.items():
    print('\ncolor_cor_%s = {' % sntype)
    print("    %d: 0.0," % central_waves['g'])
    for fname, color in corrections.items():
        print("    %d: %.2f,  # g - %s" % (central_waves[fname], color, fname))
    print('}')

# -----------------------------------------------------------------------
# WRITE OUTPUT FILE
# -----------------------------------------------------------------------
output_path = os.path.join(base_root, 'JADES/peak_color_corrections.txt')
write_color_corrections(all_color_corrections, central_waves,
                        sdss_filter_names, output_path)