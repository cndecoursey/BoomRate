#!/usr/bin/env python
"""
compute_absmags_sdss_g.py

Converts Richardson+2014 peak absolute magnitudes from Bessell B-band (Vega)
to SDSS g-band (AB) using synthetic photometry from the same SED pkl files
used in control_time.run(). This ensures full consistency between the
luminosity function and the K-correction calculation in BoomRate.

Method:
    1. Load SED pkl files (same as used in control_time.run())
    2. For each SN subtype, find spectra near peak (age ~ 0 days)
    3. Integrate each spectrum through Bessell B (Vega) and SDSS g (AB)
    4. Compute synthetic (B-g) color from templates
    5. Convert: M_g(AB) = M_B(Vega) + 0.09 + (g-B)_synthetic

Usage:
    python compute_absmags_sdss_g.py

Output:
    Prints updated absmags dictionary for use in control_time.py
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

# Richardson+2014 values in Bessell B-band (Vega): [mean, sigma, scatter]
absmags_richardson_2014_B_Vega = {
    'iip': [-16.80, 0.97, 0.37],
    'iil': [-17.98, 0.90, 0.34],
    'iin': [-18.62, 1.48, 0.32],
    'ib':  [-17.54, 0.94, 0.33],
    'ic':  [-17.67, 1.04, 0.40],
    'ia':  [-19.26, 0.51, 0.20],
}

# Vega-to-AB offset for Bessell B-band (Fukugita et al. 1995)
B_vega_to_AB = 0.09

# SN type to pkl file mapping
pkl_files = {
    'iip': 'SEDs_iip.pkl',
    'iil': 'SEDs_iil.pkl',
    'iin': 'SEDs_iin.pkl',
    'ib':  'SEDs_ib.pkl',
    'ic':  'SEDs_ic.pkl',
    'ia':  'SEDs_ia.pkl',
}

# Age tolerance for "near peak" in days
peak_age_tolerance = 3.0

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def load_filter(filter_path):
    """Load a filter transmission file. Returns (wavelength_AA, transmission)."""
    data = np.loadtxt(filter_path)
    return data[:, 0], data[:, 1]


def synthetic_mag_vega(wave_sed, flux_sed, wave_filt, trans_filt, wave_vega, flux_vega):
    """
    Computes a synthetic Vega magnitude by integrating an SED through a filter.

    m_Vega = -2.5 * log10( int(F_SN * T * lambda * dlambda) /
                            int(F_Vega * T * lambda * dlambda) )

    Parameters
    ----------
    wave_sed : array
        SED wavelengths in Angstroms.
    flux_sed : array
        SED f_lambda flux.
    wave_filt : array
        Filter wavelengths in Angstroms.
    trans_filt : array
        Filter transmission (0 to 1).
    wave_vega : array
        Vega spectrum wavelengths in Angstroms.
    flux_vega : array
        Vega spectrum f_lambda flux.

    Returns
    -------
    mag : float
        Synthetic Vega magnitude, or NaN if integration fails.
    """
    # Interpolate the filter transmission curve onto the SN SED wavelength grid
    # Ensure that SED wavelengths outside the filter's wavelength range get zero 
    # transmission rather than raising an error
    f_interp = interp1d(wave_filt, trans_filt, bounds_error=False, fill_value=0.0)
    trans_on_sed = f_interp(wave_sed)
    # Remove any tiny negative values that can arise from interpolation near zero.
    trans_on_sed = np.clip(trans_on_sed, 0., None)

    # integrate SED through filter: sum(F * T * lambda * dlambda)
    # Compute the wavelength step size at each point in the SED grid
    dlambda_sed = np.gradient(wave_sed)

    # Integrates the SN SED through the filter using the photon-counting flux formula:
    # integral(F_lambda * T(lambda) * lambda * dlambda)
    flux_sn_through_filter = np.sum(flux_sed * trans_on_sed * wave_sed * dlambda_sed)

    # Get the synthetic flux of Vega through the filter — the reference against which the SN flux is measured.
    trans_on_vega = f_interp(wave_vega)
    trans_on_vega = np.clip(trans_on_vega, 0., None)
    dlambda_vega = np.gradient(wave_vega)
    flux_vega_through_filter = np.sum(flux_vega * trans_on_vega * wave_vega * dlambda_vega)

    # Guard against logarithm of zero or negative values in the magnitude calculation below
    # This can happen if the SED has no flux in the filter bandpass (e.g. at very early ages 
    # before the SN has risen) or if the filter and SED wavelength ranges don't overlap.
    if flux_sn_through_filter <= 0. or flux_vega_through_filter <= 0.:
        return np.nan

    # Computes the synthetic Vega magnitude as:
    # m_Vega = -2.5 * log10(F_SN / F_Vega)
    # This is the standard Vega magnitude definition — the magnitude of the SN relative to Vega. 
    # Since Vega is defined to have magnitude 0 in all filters by convention, this ratio directly 
    # gives the magnitude of the SN. 
    return -2.5 * np.log10(flux_sn_through_filter / flux_vega_through_filter)


def synthetic_mag_AB(wave_sed, flux_sed, wave_filt, trans_filt):
    """
    Computes a synthetic AB magnitude by integrating an SED through a filter.

    m_AB = -2.5 * log10( int(F_nu * T * dnu) / int(T * dnu) ) - 48.6
         = -2.5 * log10( int(F_lambda * T * dlambda) /
                          int(T * c/lambda^2 * dlambda) ) - 48.6

    Parameters
    ----------
    wave_sed : array
        SED wavelengths in Angstroms.
    flux_sed : array
        SED f_lambda flux.
    wave_filt : array
        Filter wavelengths in Angstroms.
    trans_filt : array
        Filter transmission (0 to 1).

    Returns
    -------
    mag : float
        Synthetic AB magnitude (arbitrary zero point since we only need
        color differences), or NaN if integration fails.
    """

    # Interpolate the filter transmission onto the SED wavelength grid and clips negative values
    f_interp = interp1d(wave_filt, trans_filt, bounds_error=False, fill_value=0.0)
    trans_on_sed = f_interp(wave_sed)
    trans_on_sed = np.clip(trans_on_sed, 0., None)

    # Computes the wavelength step size at each point using central differences, 
    # handling non-uniform wavelength sampling correctly.
    dlambda_sed = np.gradient(wave_sed)

    # Define the speed of light in Angstroms per second. 
    # This is needed to convert between f_lambda (flux per unit wavelength) and f_nu (flux per unit frequency) via:
    # f_nu = f_lambda * lambda^2 / c
    # The AB system is defined in terms of f_nu, so this conversion is necessary.
    c = 2.998e18  # speed of light in Angstroms/s

    # numerator: int(F_lambda * T * dlambda)
    # The AB magnitude numerator is defined in frequency space as: integral(F_nu * T * dnu)
    # Substituting F_nu = F_lambda * lambda^2/c and dnu = (c/lambda^2) * dlambda:
    # integral(F_lambda * lambda^2/c * T * c/lambda^2 * dlambda)
    # The lambda^2/c from F_nu and the c/lambda^2 from dnu cancel exactly:
    # = integral(F_lambda * T * dlambda)
    numerator = np.sum(flux_sed * trans_on_sed * dlambda_sed)

    # denominator: int(T * c/lambda^2 * dlambda)  [AB zero point]
    # Computes the denominator of the AB magnitude formula:
    # denominator = integral(T * dnu) = integral(T * c/lambda^2 * dlambda)
    # This is the AB zero point normalization — it represents what a flat f_nu source 
    # (the AB reference spectrum) would contribute to the denominator. The c/lambda^2 factor 
    # converts from wavelength to frequency space. This normalization ensures the AB magnitude 
    # is properly calibrated to the AB zero point of -48.6 (in CGS units of erg/s/cm²/Hz).
    denominator = np.sum(trans_on_sed * c / wave_sed**2 * dlambda_sed)

    # Prevent taking the log of zero or negative values
    if numerator <= 0. or denominator <= 0.:
        return np.nan

    # Computes the synthetic AB magnitude:
    # The -48.6 is the AB magnitude zero point in CGS units — it comes from the definition 
    # of the AB system where a source with constant f_nu = 3631 Jy has magnitude 0 in all filters. 
    # In CGS units, 3631 Jy = 3.631e-20 erg/s/cm²/Hz, and -2.5*log10(3.631e-20) ≈ 48.6.
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
    model_names : list of str
        Names of models contributing peak spectra.
    """
    with open(pkl_path, 'rb') as f:
        models_dict = pickle.load(f, encoding='latin1')

    peak_spectra = []
    model_names = []

    for model_name, spec in models_dict.items():
        # Find all rows within peak_age_tolerance days of age=0. . Skips this model if 
        # no spectra exist near peak, or if all flux values are zero (pre-explosion baseline).
        idx = np.where(np.abs(spec[:, 0]) <= peak_age_tolerance)[0]
        if len(idx) == 0:
            continue
        if np.sum(spec[idx, 2]) == 0.:
            continue

        # Extract the wavelength and flux columns for the rows near peak.
        wave = spec[idx, 1]   # wavelengths in Angstroms
        flux = spec[idx, 2]   # f_lambda flux

        # Handle the case where multiple age steps fall within the tolerance window. 
        # Rather than averaging spectra across multiple ages, it selects the single age 
        # closest to zero — the true peak. This is important because even a few days difference 
        # in age can meaningfully change the SED shape near peak.
        if len(np.unique(spec[idx, 0])) > 1:
            unique_ages = np.unique(spec[idx, 0])
            # use the age closest to 0
            best_age = unique_ages[np.argmin(np.abs(unique_ages))]
            idx = np.where(spec[:, 0] == best_age)[0]
            wave = spec[idx, 1]
            flux = spec[idx, 2]

        # Sort the wavelength and flux arrays by wavelength in ascending order. 
        # This is necessary for the interpolation functions in synthetic_mag_vega() 
        # and synthetic_mag_AB() which require monotonically increasing wavelengths. 
        # Different SED files may not store wavelengths in order.
        sort_idx = np.argsort(wave)
        wave = wave[sort_idx]
        flux = flux[sort_idx]

        # Remove unphysical zero or near-zero wavelengths
        valid_wave = wave > 100.  # minimum physical wavelength in Angstroms
        wave = wave[valid_wave]
        flux = flux[valid_wave]

        # Only append the spectrum if it has more than 2 wavelength points (the minimum 
        # needed for interpolation) and has positive total flux. Then appends the 
        # wavelength/flux tuple to peak_spectra and the model name to model_names.
        if len(wave) > 2 and np.sum(flux) > 0.:
            peak_spectra.append((wave, flux))
            model_names.append(model_name)

    # Return the list of peak spectra as (wave, flux) tuples and the corresponding model names
    return peak_spectra, model_names


def plot_color_diagnostic(sntype, B_mags_vega, g_mags_AB, B_minus_g, model_names, diag_dir):
    """
    Plots the distribution of synthetic (B-g) colors across templates
    for a given SN subtype.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- left panel: B(Vega) vs g(AB) for each template ---
    ax1.scatter(B_mags_vega, g_mags_AB, color='steelblue', s=30, zorder=5)
    # plot 1:1 line for reference
    all_mags = np.concatenate([B_mags_vega, g_mags_AB])
    valid = np.isfinite(all_mags)
    if np.sum(valid) > 0:
        mag_min = np.nanmin(all_mags) - 0.5
        mag_max = np.nanmax(all_mags) + 0.5
        ax1.plot([mag_min, mag_max], [mag_min, mag_max], 'k--', lw=1,
                 alpha=0.5, label='1:1 line')
    ax1.set_xlabel('Synthetic B (Vega, arbitrary zero point)', fontsize=11)
    ax1.set_ylabel('Synthetic g (AB, arbitrary zero point)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title('%s synthetic B vs g across templates' % sntype.upper(), fontsize=11)
    ax1.grid(alpha=0.3)

    # --- right panel: (B-g) color distribution ---
    valid_colors = B_minus_g[np.isfinite(B_minus_g)]
    if len(valid_colors) > 0:
        if len(valid_colors) == 1:
            # only one template -- just show a vertical line instead of histogram
            ax2.axvline(valid_colors[0], color='steelblue', lw=3,
                        label='Single template value = %.3f mag' % valid_colors[0])
        else:
            ax2.hist(valid_colors, bins=max(3, len(valid_colors)//3),
                     color='steelblue', alpha=0.7, edgecolor='k')
            ax2.axvline(np.median(valid_colors), color='red', ls='--', lw=2,
                        label='Median = %.3f mag' % np.median(valid_colors))
            ax2.axvline(np.mean(valid_colors), color='orange', ls='-.', lw=2,
                        label='Mean = %.3f mag' % np.mean(valid_colors))
    ax2.set_xlabel('Synthetic (B_Vega - g_AB) color (arbitrary zero point)', fontsize=11)
    ax2.set_ylabel('N templates', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title('%s (B-g) color distribution\nN=%d templates' % (
                  sntype.upper(), len(valid_colors)), fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('%s/Bessell_B_to_SDSS_g/%s_B_to_g_color_diagnostic.png' % (diag_dir, sntype))
    plt.clf()
    print('  Diagnostic plot saved to %s/Bessell_B_to_SDSS_g/%s_B_to_g_color_diagnostic.png' % (diag_dir, sntype))

def write_absmags(absmags_sdss_g_AB, output_path):
    """
    Writes the converted SDSS g-band (AB) peak absolute magnitudes to a
    text file for use in BoomRate config files.

    File format:
        # comment lines starting with #
        sntype  mean_mag  sigma  scatter
        e.g.:
        iip  -16.71  0.97  0.37

    Parameters
    ----------
    absmags_sdss_g_AB : dict
        Dictionary mapping sntype to [mean_mag, sigma, scatter].
    output_path : str
        Path to the output text file.
    """
    with open(output_path, 'w') as f:
        f.write('# Peak absolute magnitudes in SDSS g-band (AB)\n')
        f.write('# Converted from Richardson+2014 Bessell B-band (Vega)\n')
        f.write('# using synthetic photometry from SED pkl templates\n')
        f.write('#\n')
        f.write('# Convention: mean_mag is the mean peak absolute magnitude\n')
        f.write('#             sigma is the 1-sigma luminosity function width\n')
        f.write('#             scatter is the additional intrinsic scatter\n')
        f.write('#\n')
        f.write('# sntype  mean_mag  sigma  scatter\n')
        f.write('#\n')
        for sntype, vals in absmags_sdss_g_AB.items():
            f.write('%s  %.3f  %.3f  %.3f\n' % (
                    sntype, vals[0], vals[1], vals[2]))
    print('\nPeak absolute magnitudes written to %s' % output_path)

# -----------------------------------------------------------------------
# LOAD FILTERS AND VEGA SPECTRUM
# -----------------------------------------------------------------------

print('Loading filters and Vega spectrum...')

# Bessell B filter
bessell_B_path = '%s/filters/Bessell90/Bessell90_K09/Bessell90_B.dat' % sndata_root
wave_B, trans_B = load_filter(bessell_B_path)

# SDSS g filter
sdss_g_path = '%s/filters/SDSS/SDSS_web2001/g.dat' % sndata_root
wave_g, trans_g = load_filter(sdss_g_path)

# Vega spectrum (use the standard version)
vega_path = '%s/templates/vega_model.dat' % base_root
vega_data = np.loadtxt(vega_path)
wave_vega = vega_data[:, 0]
flux_vega = vega_data[:, 1]

print('  Bessell B central wavelength: %.1f AA' % (
      np.sum(wave_B * trans_B * wave_B) / np.sum(trans_B * wave_B)))
print('  SDSS g central wavelength:    %.1f AA' % (
      np.sum(wave_g * trans_g * wave_g) / np.sum(trans_g * wave_g)))

# -----------------------------------------------------------------------
# MAIN COMPUTATION
# -----------------------------------------------------------------------

absmags_sdss_g_AB = {}
print('\n' + '='*60)
print('Computing B(Vega) -> g(AB) conversions...')
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

    # compute synthetic magnitudes for each template
    B_mags_vega = []
    g_mags_AB   = []

    for i, (wave, flux) in enumerate(peak_spectra):
        # synthetic B (Vega) -- relative, zero point cancels
        B_vega = synthetic_mag_vega(wave, flux, wave_B, trans_B, wave_vega, flux_vega)
        # synthetic g (AB) -- relative, zero point cancels
        g_AB   = synthetic_mag_AB(wave, flux, wave_g, trans_g)

        B_mags_vega.append(B_vega)
        g_mags_AB.append(g_AB)

    B_mags_vega = np.array(B_mags_vega)
    g_mags_AB   = np.array(g_mags_AB)

    # compute (B_Vega - g_AB) color for each template
    # note: zero points are arbitrary but cancel in the difference
    # the color term captures the filter shape difference only
    B_minus_g = B_mags_vega - g_mags_AB

    # use median color across templates (robust to outliers)
    valid = np.isfinite(B_minus_g)
    if np.sum(valid) == 0:
        print('  WARNING: no valid color measurements, skipping')
        continue

    if np.sum(valid) == 1:
        print('  WARNING: only 1 template available for %s -- color term based on single SED' % sntype)
        median_color = B_minus_g[valid][0]
        std_color = 0.0
    else:
        median_color = np.median(B_minus_g[valid])
        std_color = np.std(B_minus_g[valid])
    print('  Synthetic (B_Vega - g_AB) color: %.3f +/- %.3f mag (N=%d templates)' % (
          median_color, std_color, np.sum(valid)))

    # convert Richardson+2014 M_B(Vega) to M_g(AB)
    # M_g(AB) = M_B(Vega) + B_vega_to_AB + (g-B) color correction
    # where (g-B) = -median_color since we computed (B-g)
    M_B_vega = absmags_richardson_2014_B_Vega[sntype][0]
    sigma    = absmags_richardson_2014_B_Vega[sntype][1]
    scatter  = absmags_richardson_2014_B_Vega[sntype][2]

    M_g_AB = M_B_vega + B_vega_to_AB - median_color
    print('  M_B(Vega) = %.2f -> M_g(AB) = %.2f' % (M_B_vega, M_g_AB))
    print('  (color correction: B_Vega_to_AB=+%.2f, B_to_g=%.3f)' % (
          B_vega_to_AB, -median_color))

    # store result -- keep same sigma and scatter from Richardson+2014
    absmags_sdss_g_AB[sntype] = [M_g_AB, sigma, scatter]

    # diagnostic plot
    plot_color_diagnostic(sntype, B_mags_vega, g_mags_AB,
                          B_minus_g, model_names, diag_dir)

# -----------------------------------------------------------------------
# PRINT FINAL RESULTS
# -----------------------------------------------------------------------

print('\n' + '='*60)
print('RESULTS: absmags_richardson_2014 converted to SDSS g-band (AB)')
print('='*60)
print('\nOriginal (Bessell B, Vega):')
for sntype, vals in absmags_richardson_2014_B_Vega.items():
    print('  %s: [%.2f, %.2f, %.2f]' % (sntype, vals[0], vals[1], vals[2]))

print('\nConverted (SDSS g, AB):')
print('absmags_sdss_g_AB = {')
for sntype, vals in absmags_sdss_g_AB.items():
    print("    '%s': [%.2f, %.2f, %.2f]," % (sntype, vals[0], vals[1], vals[2]))
print('}')

print('\nConversion offsets (M_g_AB - M_B_Vega):')
for sntype in absmags_sdss_g_AB.keys():
    offset = absmags_sdss_g_AB[sntype][0] - absmags_richardson_2014_B_Vega[sntype][0]
    print('  %s: %+.3f mag' % (sntype, offset))

output_absmags_path = os.path.join(base_root, 'JADES/absmags_sdss_g_AB.txt')
write_absmags(absmags_sdss_g_AB, output_absmags_path)