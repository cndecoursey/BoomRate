#!/usr/bin/env python
"""
plot_peak_sed_colors.py

Plots the peak SED for each SN subtype with the SDSS g and z filter
transmission curves overlaid, to visually verify that the color corrections
in peak_color_corrections.txt are physically reasonable.

For each subtype, shows:
- Left panel: peak SED with g and z filter transmission overlaid
- Right panel: synthetic flux through each SDSS filter at peak,
  showing the color corrections as magnitude differences

Usage:
    python plot_peak_sed_colors.py
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
pkl_dir = base_root
diag_dir    = base_root + '/diagnostic_plots'
# -----------------------------------------------------------------------

# SN type to pkl file mapping
pkl_files = {
    'iip': 'SEDs_iip.pkl',
    'iil': 'SEDs_iil.pkl',
    'iin': 'SEDs_iin.pkl',
    'ib':  'SEDs_ib.pkl',
    'ic':  'SEDs_ic.pkl',
}

# SDSS filter central wavelengths for labeling
filter_names   = ['u', 'g', 'r', 'i', 'z']
filter_colors  = {'u': 'purple', 'g': 'green', 'r': 'red',
                  'i': 'darkorange', 'z': 'brown'}
peak_age_tolerance = 3.0

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def load_filter(filter_path):
    data = np.loadtxt(filter_path)
    return data[:, 0], data[:, 1]


def synthetic_mag_AB(wave_sed, flux_sed, wave_filt, trans_filt):
    """Compute synthetic AB magnitude."""
    f_interp    = interp1d(wave_filt, trans_filt,
                           bounds_error=False, fill_value=0.0)
    trans       = np.clip(f_interp(wave_sed), 0., None)
    dlambda     = np.gradient(wave_sed)
    c           = 2.998e18  # AA/s
    numerator   = np.sum(flux_sed * trans * dlambda)
    denominator = np.sum(trans * c / wave_sed**2 * dlambda)
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


# -----------------------------------------------------------------------
# LOAD SDSS FILTERS
# -----------------------------------------------------------------------

print('Loading SDSS filters...')
sdss_filters   = {}
central_waves  = {}
for fname in filter_names:
    fpath = '%s/filters/SDSS/SDSS_web2001/%s.dat' % (sndata_root, fname)
    wave_f, trans_f    = load_filter(fpath)
    sdss_filters[fname] = (wave_f, trans_f)
    cen = np.sum(wave_f * trans_f * wave_f) / np.sum(trans_f * wave_f)
    central_waves[fname] = cen
    print('  SDSS %s: central wavelength = %.0f AA' % (fname, cen))

# -----------------------------------------------------------------------
# MAIN PLOT
# -----------------------------------------------------------------------

n_types = len(pkl_files)
fig, axes = plt.subplots(n_types, 2, figsize=(16, 4*n_types))

for row, (sntype, pkl_file) in enumerate(pkl_files.items()):
    pkl_path = os.path.join(pkl_dir, pkl_file)
    ax_sed   = axes[row, 0]
    ax_color = axes[row, 1]

    if not os.path.isfile(pkl_path):
        ax_sed.text(0.5, 0.5, 'pkl not found', transform=ax_sed.transAxes,
                    ha='center', va='center')
        continue

    print('\nProcessing %s...' % sntype.upper())
    # load peak spectra using raw flux (matching compute_peak_colors.py)
    peak_spectra, model_names = get_peak_spectra(pkl_path, peak_age_tolerance)
    print('  Found %d templates with spectra near peak' % len(peak_spectra))

    if len(peak_spectra) == 0:
        ax_sed.text(0.5, 0.5, 'No peak spectra found',
                    transform=ax_sed.transAxes, ha='center', va='center')
        ax_color.text(0.5, 0.5, 'No peak spectra found',
                      transform=ax_color.transAxes, ha='center', va='center')
        continue

    # --- left panel: SED with filter transmission overlaid ---
    # normalize for display only -- raw flux used for color computation
    for wave, flux in peak_spectra:
        peak_val = np.nanmax(flux)
        if peak_val > 0:
            flux_norm = flux / peak_val
        else:
            flux_norm = flux
        # interpolate onto common grid for display
        wave_min = 2500.
        wave_max = 10000.
        wave_display = np.arange(wave_min, wave_max, 10.)
        f_interp = interp1d(wave, flux_norm, bounds_error=False, fill_value=0.0)
        flux_display = f_interp(wave_display)
        ax_sed.plot(wave_display, flux_display,
                    color='gray', lw=0.5, alpha=0.4)

    # compute and plot median normalized SED for display
    all_flux_display = []
    for wave, flux in peak_spectra:
        peak_val = np.nanmax(flux)
        if peak_val > 0:
            flux_norm = flux / peak_val
        else:
            flux_norm = flux
        f_interp = interp1d(wave, flux_norm, bounds_error=False, fill_value=0.0)
        all_flux_display.append(f_interp(wave_display))
    median_flux_display = np.nanmedian(all_flux_display, axis=0)
    ax_sed.plot(wave_display, median_flux_display, 'k-', lw=2, label='Median SED')

    # overlay filter transmission curves (scaled for visibility)
    ax_twin = ax_sed.twinx()
    for fname in filter_names:
        wave_f, trans_f = sdss_filters[fname]
        ax_twin.plot(wave_f, trans_f,
                     color=filter_colors[fname], lw=1.5, ls='--',
                     alpha=0.7, label='SDSS %s' % fname)
        # mark central wavelength
        ax_twin.axvline(central_waves[fname], color=filter_colors[fname],
                        lw=0.5, alpha=0.4)

    ax_sed.set_xlabel('Wavelength (Angstroms)', fontsize=10)
    ax_sed.set_ylabel('Normalized flux (peak=1)', fontsize=10)
    ax_twin.set_ylabel('Filter transmission', fontsize=10)
    ax_sed.set_xlim(2500, 10000)
    ax_sed.set_ylim(-0.05, 1.2)
    ax_twin.set_ylim(0, 1.5)
    ax_sed.set_title('%s peak SED (N=%d templates)' % (
                     sntype.upper(), len(peak_spectra)), fontsize=11)

    # combine legends
    lines1, labels1 = ax_sed.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_sed.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper right')

    # --- right panel: synthetic AB magnitudes through each filter ---
    # Use raw unnormalized flux per template (matching compute_peak_colors.py)
    # compute synthetic mag in each filter for each template individually
    filter_mags = {fname: [] for fname in filter_names}
    for wave, flux in peak_spectra:
        for fname in filter_names:
            wave_f, trans_f = sdss_filters[fname]
            mag = synthetic_mag_AB(wave, flux, wave_f, trans_f)
            filter_mags[fname].append(mag)

    # convert to arrays
    for fname in filter_names:
        filter_mags[fname] = np.array(filter_mags[fname])

    # compute X - g color per template then take median (matching compute_peak_colors.py)
    g_mags = filter_mags['g']
    color_corrections = {}
    for fname in filter_names:
        if fname == 'g':
            color_corrections[fname] = 0.0
            continue
        x_mags = filter_mags[fname]
        colors = x_mags - g_mags  # X - g convention
        valid  = np.isfinite(colors)
        if np.sum(valid) == 0:
            color_corrections[fname] = np.nan
        else:
            color_corrections[fname] = np.nanmedian(colors[valid])

    # histogram diagnostic
    print('\n%s per-template z-g colors:' % sntype)
    z_g_colors = []
    for i, (wave, flux) in enumerate(peak_spectra):
        wave_g, trans_g = sdss_filters['g']
        wave_z, trans_z = sdss_filters['z']
        g_mag = synthetic_mag_AB(wave, flux, wave_g, trans_g)
        z_mag = synthetic_mag_AB(wave, flux, wave_z, trans_z)
        z_g = z_mag - g_mag
        z_g_colors.append(z_g)
        print('  Template %d (%s): g=%.3f, z=%.3f, z-g=%.3f' % (
              i, model_names[i], g_mag, z_mag, z_g))
    print('  Median z-g: %.3f' % color_corrections['z'])

    # plot histogram of z-g colors
    z_g_colors = np.array(z_g_colors)
    valid = np.isfinite(z_g_colors)

    fig_diag, ax_diag = plt.subplots(1, 1, figsize=(8, 6))
    ax_diag.bar(range(len(model_names)),
                z_g_colors,
                color=['steelblue' if c < 0 else 'salmon' for c in z_g_colors],
                alpha=0.7, edgecolor='k')
    ax_diag.axhline(np.nanmedian(z_g_colors[valid]), color='red', ls='--', lw=2,
                    label='Median = %.3f mag' % np.nanmedian(z_g_colors[valid]))
    ax_diag.axhline(np.nanmean(z_g_colors[valid]), color='orange', ls='-.', lw=2,
                    label='Mean = %.3f mag' % np.nanmean(z_g_colors[valid]))
    ax_diag.axhline(0., color='k', ls=':', lw=1)
    ax_diag.set_xticks(range(len(model_names)))
    ax_diag.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax_diag.set_ylabel('z - g color (mag)\n(positive = bluer than g, negative = redder than g)',
                       fontsize=10)
    ax_diag.set_title('%s per-template z-g colors at peak\n'
                      'blue bars = bluer in z than g, red bars = redder in z than g' % sntype,
                      fontsize=11)
    ax_diag.legend(fontsize=10)
    ax_diag.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('%s/peak_color_diagnostic/%s_zg_color_distribution.png' % (diag_dir,sntype),
                bbox_inches='tight', dpi=150)
    plt.close(fig_diag)
    print('  %s z-g diagnostic plot saved.' % sntype)

    # plot color corrections as bar chart
    x_pos       = np.array([central_waves[f] for f in filter_names])
    colors_vals = [color_corrections[f] for f in filter_names]
    bar_colors  = [filter_colors[f] for f in filter_names]

    bars = ax_color.bar(x_pos, colors_vals, width=200,
                        color=bar_colors, alpha=0.7, edgecolor='k')
    ax_color.axhline(0., color='k', ls='--', lw=1)

    # label each bar with its value
    for bar, val, fname in zip(bars, colors_vals, filter_names):
        if np.isfinite(val):
            offset = 0.01 if val >= 0 else -0.05
            va     = 'bottom' if val >= 0 else 'top'
            ax_color.text(bar.get_x() + bar.get_width()/2.,
                          bar.get_height() + offset,
                          '%.3f' % val,
                          ha='center', va=va, fontsize=8)

    ax_color.set_xlabel('Filter central wavelength (Angstroms)', fontsize=10)
    ax_color.set_ylabel('X - g color (mag)\n(positive = fainter than g, negative = brighter than g)',
                        fontsize=10)
    ax_color.set_title('%s synthetic X-g colors at peak\n(N=%d templates)' % (
                       sntype.upper(), len(peak_spectra)), fontsize=11)
    ax_color.set_xticks(x_pos)
    ax_color.set_xticklabels(['%s\n(%.0f AA)' % (f, central_waves[f])
                               for f in filter_names], fontsize=8)
    ax_color.grid(alpha=0.3, axis='y')

    print('  Color corrections (X-g) at peak:')
    for fname in filter_names:
        print('    %s - g = %+.3f mag' % (fname, color_corrections.get(fname, np.nan)))

plt.suptitle('Peak SED and color corrections by SN subtype', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('%s/peak_color_diagnostic/peak_sed_color_verification.png' % diag_dir, bbox_inches='tight', dpi=150)
plt.show()
print('\nPlot saved to %s/peak_color_diagnostics/peak_sed_color_verification.png' % diag_dir)