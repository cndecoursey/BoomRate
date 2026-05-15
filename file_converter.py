#!/usr/bin/env python
"""
sed_to_dat.py

Converts SNANA .SED spectral template files into .DAT broadband light curve
files compatible with control_time.py's read_lc_model() function.

For each .SED file, computes synthetic AB magnitudes in each SDSS filter
by integrating the f_lambda SED through the filter transmission curves,
then writes the results in .DAT format.

The .DAT format is:
    NEPOCH: N
    SNTYPE: X
    FILTER: a $SNDATA_ROOT/filters/SDSS/SDSS_web2001/u.dat
    FILTER: b $SNDATA_ROOT/filters/SDSS/SDSS_web2001/g.dat
    ...
    EPOCH: phase  mag_u  mag_g  mag_r  mag_i  mag_z
    ...
    END:

Usage:
    python sed_to_dat.py

Output:
    One .DAT file per .SED file in the output directory.
"""

import os
import glob
import numpy as np
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
sndata_root    = '/Users/christadecoursey/Documents/SNANA/SNANA_2025'
boomrate_root  = '/Users/christadecoursey/Documents/JADES/SN_Classification_and_Rates_Paper/classification/BoomRate'
spec_temp_ref  = 'NON1ASED.J17_CC'
sed_dir        = sndata_root + '/models/NON1ASED/' + spec_temp_ref  # directory with .SED files 
dat_output_dir = boomrate_root + '/broadband_lightcurves/' + spec_temp_ref  # where to write .DAT files
# -----------------------------------------------------------------------

# SDSS filter labels and paths as they appear in .DAT files
sdss_filters = [
    ('a', 'u', '%s/filters/SDSS/SDSS_web2001/u.dat' % sndata_root),
    ('b', 'g', '%s/filters/SDSS/SDSS_web2001/g.dat' % sndata_root),
    ('c', 'r', '%s/filters/SDSS/SDSS_web2001/r.dat' % sndata_root),
    ('d', 'i', '%s/filters/SDSS/SDSS_web2001/i.dat' % sndata_root),
    ('e', 'z', '%s/filters/SDSS/SDSS_web2001/z.dat' % sndata_root),
]

# magnitude value to use for pre-explosion baseline (unphysically faint)
BASELINE_MAG = 25.0

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def load_filter(filter_path):
    """
    Load a SDSS filter transmission file.

    Returns
    -------
    wave : numpy.ndarray
        Wavelengths in Angstroms.
    trans : numpy.ndarray
        Filter transmission (0 to 1).
    """
    data  = np.loadtxt(filter_path)
    wave  = data[:, 0]
    trans = data[:, 1]
    return wave, trans


def synthetic_mag_AB(wave_sed, flux_sed, wave_filt, trans_filt):
    """
    Computes a synthetic AB magnitude by integrating an f_lambda SED
    through a filter transmission curve.

    Uses the AB magnitude formula in wavelength space:
        m_AB = -2.5*log10( int(F_lambda * T * dlambda) /
                            int(T * c/lambda^2 * dlambda) ) - 48.6

    Parameters
    ----------
    wave_sed : numpy.ndarray
        SED wavelengths in Angstroms. Must be > 100 AA and sorted ascending.
    flux_sed : numpy.ndarray
        SED f_lambda flux in erg/s/cm^2/AA.
    wave_filt : numpy.ndarray
        Filter wavelengths in Angstroms.
    trans_filt : numpy.ndarray
        Filter transmission (0 to 1).

    Returns
    -------
    mag : float
        Synthetic AB magnitude, or NaN if integration fails.
    """
    # interpolate filter onto SED wavelength grid
    f_interp    = interp1d(wave_filt, trans_filt,
                           bounds_error=False, fill_value=0.0)

    # np.clip removes any tiny negative values from interpolation near zero.
    trans       = np.clip(f_interp(wave_sed), 0., None)

    # Compute the wavelength step size at each point using central differences, handling 
    # non-uniform wavelength sampling correctly. More accurate than assuming a constant step size.
    dlambda     = np.gradient(wave_sed)
    c           = 2.998e18  # speed of light in AA/s

    # Computes the numerator of the AB magnitude formula by integrating the SED through the filter:
    # numerator = integral(F_lambda * T(lambda) * dlambda)
    # This comes from converting the AB frequency-space formula to wavelength space — the 
    # F_lambda * lambda^2/c from converting f_nu to f_lambda and the c/lambda^2 from converting dnu 
    # to dlambda cancel exactly, leaving just F_lambda * T * dlambda
    numerator   = np.sum(flux_sed * trans * dlambda)

    # Computes the AB zero point denominator:
    # denominator = integral(T(lambda) * c/lambda^2 * dlambda)
    # This represents what a hypothetical flat f_nu source (the AB reference spectrum, 
    # defined as constant f_nu = 3631 Jy) contributes through this filter in frequency space, 
    # converted to wavelength space. The c/lambda^2 factor is the Jacobian of the frequency-to-wavelength 
    # conversion for the reference source.
    denominator = np.sum(trans * c / wave_sed**2 * dlambda)

    if numerator <= 0. or denominator <= 0.:
        return np.nan

    # Compute the final AB magnitude:
    # m_AB = -2.5 * log10(numerator/denominator) - 48.6
    # The -48.6 is the AB magnitude zero point in CGS units — it comes from the definition 
    # of the AB system where a source with constant f_nu = 3631 Jy has magnitude 0 in all filters. 
    # In CGS units, 3631 Jy = 3.631e-20 erg/s/cm²/Hz, and -2.5*log10(3.631e-20) ≈ 48.6. This constant 
    # only gives the correct absolute AB magnitude if F_lambda is in erg/s/cm²/Å — if the units are 
    # different, the result is offset by a constant that gets absorbed into magoff.
    return -2.5 * np.log10(numerator / denominator) - 48.6


def read_sed_file(sed_path):
    """
    Reads a SNANA .SED file and returns the SNTYPE and SED data.

    Parameters
    ----------
    sed_path : str
        Path to the .SED file.

    Returns
    -------
    sntype : str
        SN subtype from the header (e.g. 'Ic', 'IIP').
    sed_data : numpy.ndarray
        2D array with columns: phase (days), wavelength (AA), f_lambda flux.
    """
    sntype   = None
    sed_data = []

    with open(sed_path, 'r') as f:
        for line in f.readlines():
            # extract SNTYPE from header
            if 'SNTYPE:' in line and line.startswith('#'):
                sntype = line.split('SNTYPE:')[1].strip()
            # skip comment and blank lines
            if line.startswith('#') or line.strip() == '':
                continue
            try:
                vals = list(map(float, line.split()))
                if len(vals) == 3:
                    sed_data.append(vals)
            except:
                continue

    return sntype, np.array(sed_data)


def write_dat_file(dat_path, sntype, phases, epoch_mags, sdss_filters):
    """
    Writes a .DAT format light curve file compatible with read_lc_model().

    Parameters
    ----------
    dat_path : str
        Output path for the .DAT file.
    sntype : str
        SN subtype label (e.g. 'Ic').
    phases : list of float
        Rest-frame phases in days.
    epoch_mags : list of list
        Per-epoch AB magnitudes in each SDSS filter.
        Shape: (N_epochs, N_filters).
    sdss_filters : list of tuples
        List of (label, name, path) for each SDSS filter.
    """
    with open(dat_path, 'w') as f:
        f.write('NEPOCH: %d \n' % len(phases))
        f.write('SNTYPE: %s \n' % sntype)
        for label, fname, fpath in sdss_filters:
            f.write('FILTER:  %s  $SNDATA_ROOT/filters/SDSS/SDSS_web2001/%s.dat \n' % (
                    label, fname))
        for phase, mags in zip(phases, epoch_mags):
            mag_str = '   '.join(['%8.3f' % m for m in mags])
            f.write('EPOCH:  %10.4f     %s \n' % (phase, mag_str))
        f.write('END: \n')


# -----------------------------------------------------------------------
# LOAD SDSS FILTERS ONCE
# -----------------------------------------------------------------------

print('Loading SDSS filters...')
sdss_filter_data = []
for label, fname, fpath in sdss_filters:
    wave_f, trans_f = load_filter(fpath)
    sdss_filter_data.append((label, fname, wave_f, trans_f))
    cen = np.sum(wave_f * trans_f * wave_f) / np.sum(trans_f * wave_f)
    print('  SDSS %s: central wavelength = %.0f AA' % (fname, cen))

# -----------------------------------------------------------------------
# PROCESS EACH .SED FILE
# -----------------------------------------------------------------------

os.makedirs(dat_output_dir, exist_ok=True)

sed_files = sorted(glob.glob(os.path.join(sed_dir, '*.SED')))
print('\nFound %d .SED files to convert' % len(sed_files))

n_success = 0
n_failed  = 0

for sed_path in sed_files:
    model_name = os.path.basename(sed_path).replace('.SED', '')
    dat_path   = os.path.join(dat_output_dir, model_name + '.DAT')

    print('\nConverting %s...' % model_name)

    # read the SED file
    sntype, sed_data = read_sed_file(sed_path)

    if sntype is None:
        print('  WARNING: SNTYPE not found in header, skipping')
        n_failed += 1
        continue

    if len(sed_data) == 0:
        print('  WARNING: no SED data found, skipping')
        n_failed += 1
        continue

    print('  SNTYPE: %s' % sntype)
    print('  SED data shape: %s' % str(sed_data.shape))

    # get unique phases in chronological order
    phases = sorted(list(set(sed_data[:, 0])))
    print('  N epochs: %d  (%.1f to %.1f days)' % (
          len(phases), min(phases), max(phases)))

    # compute synthetic AB magnitude in each SDSS filter at each phase
    epoch_mags = []
    n_baseline = 0
    n_computed = 0

    for phase in phases:
        # extract SED at this phase
        idx  = np.where(sed_data[:, 0] == phase)[0]
        wave = sed_data[idx, 1]
        flux = sed_data[idx, 2]

        # sort by wavelength and remove unphysical values
        sort_idx     = np.argsort(wave)
        wave, flux   = wave[sort_idx], flux[sort_idx]
        valid        = wave > 100.
        wave, flux   = wave[valid], flux[valid]

        # check if SN has flux at this phase
        if len(wave) < 2 or np.sum(flux) == 0.:
            # pre-explosion baseline -- use placeholder magnitude
            mags = [BASELINE_MAG] * len(sdss_filters)
            n_baseline += 1
        else:
            # compute synthetic AB magnitude in each filter
            mags = []
            for label, fname, wave_f, trans_f in sdss_filter_data:
                mag = synthetic_mag_AB(wave, flux, wave_f, trans_f)
                if not np.isfinite(mag):
                    # filter doesn't overlap SED -- use baseline
                    mag = BASELINE_MAG
                mags.append(mag)
            n_computed += 1

        epoch_mags.append(mags)

    print('  Computed: %d epochs, Baseline: %d epochs' % (n_computed, n_baseline))

    # print peak magnitudes for sanity check
    peak_idx  = np.argmin(abs(np.array(phases)))
    peak_mags = epoch_mags[peak_idx]
    print('  Peak mags (u,g,r,i,z) at phase %.1f days: %s' % (
          phases[peak_idx],
          '  '.join(['%.2f' % m for m in peak_mags])))

    # write the .DAT file
    write_dat_file(dat_path, sntype, phases, epoch_mags, sdss_filters)
    print('  Written to %s' % dat_path)
    n_success += 1

# -----------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------

print('\n' + '='*60)
print('Conversion complete.')
print('  Successful: %d' % n_success)
print('  Failed:     %d' % n_failed)
print('  Output directory: %s' % dat_output_dir)
print('='*60)