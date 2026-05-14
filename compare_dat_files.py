#!/usr/bin/env python
"""
compare_dat_files.py

Compares newly generated .DAT files (from SED synthetic photometry) against
the original .DAT files for each SN template. For each template, plots the
light curves in all 5 SDSS filters side by side, and computes residuals.

Usage:
    python compare_dat_files.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# CONFIGURATION -- update these paths for your machine
# -----------------------------------------------------------------------
boomrate_root     = '/Users/christadecoursey/Documents/JADES/SN_Classification_and_Rates_Paper/classification/BoomRate'
original_dat_dir  = boomrate_root + '/templates/non1a'        # original .DAT files
generated_dat_dir = boomrate_root + '/broadband_lightcurves'  # new .DAT files
diag_dir          = boomrate_root + '/diagnostic_plots/sed2dat'
# -----------------------------------------------------------------------

os.makedirs(diag_dir, exist_ok=True)

# SDSS filter labels and colors
filter_labels  = ['u', 'g', 'r', 'i', 'z']
filter_colors  = {
    'u': '#7B2FBE',
    'g': '#2E8B57',
    'r': '#CC2936',
    'i': '#E07B39',
    'z': '#8B4513',
}

# magnitude threshold for physically meaningful values
MAG_THRESHOLD = 24.0

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def read_dat_file(dat_path):
    """
    Reads a .DAT format light curve file.

    Returns
    -------
    sntype : str
        SN subtype from the SNTYPE line.
    phases : numpy.ndarray
        Rest-frame phases in days.
    mags : numpy.ndarray
        2D array of shape (N_epochs, N_filters) with magnitudes.
    """
    sntype = None
    phases = []
    mags   = []

    with open(dat_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('SNTYPE:'):
                sntype = line.split()[1]
            if line.startswith('EPOCH:'):
                vals = list(map(float, line.split()[1:]))
                phases.append(vals[0])
                mags.append(vals[1:])

    return sntype, np.array(phases), np.array(mags)


def mask_baseline(mags, threshold=MAG_THRESHOLD):
    """Returns a boolean mask where magnitudes are physically meaningful."""
    return mags < threshold


# -----------------------------------------------------------------------
# FIND MATCHING .DAT FILES
# -----------------------------------------------------------------------

original_files  = sorted(glob.glob(os.path.join(original_dat_dir, '*.DAT')))
generated_files = sorted(glob.glob(os.path.join(generated_dat_dir, '*.DAT')))

# build lookup by model name
original_dict  = {os.path.basename(f).replace('.DAT', ''): f
                  for f in original_files}
generated_dict = {os.path.basename(f).replace('.DAT', ''): f
                  for f in generated_files}

# find models present in both
common_models = sorted(set(original_dict.keys()) & set(generated_dict.keys()))
print('Found %d models in original directory' % len(original_dict))
print('Found %d models in generated directory' % len(generated_dict))
print('Found %d models in common' % len(common_models))

if len(common_models) == 0:
    print('ERROR: no matching models found. Check your directory paths.')
    exit(1)

# -----------------------------------------------------------------------
# SUMMARY STATISTICS
# -----------------------------------------------------------------------

all_residuals = {f: [] for f in filter_labels}  # residuals across all models

# -----------------------------------------------------------------------
# PER-MODEL COMPARISON PLOTS
# -----------------------------------------------------------------------

for model_name in common_models:
    orig_path = original_dict[model_name]
    gen_path  = generated_dict[model_name]

    # read both files
    sntype_orig, phases_orig, mags_orig = read_dat_file(orig_path)
    sntype_gen,  phases_gen,  mags_gen  = read_dat_file(gen_path)

    print('\n%s (%s):' % (model_name, sntype_orig))

    # find common phases
    phases_orig_set = set(np.round(phases_orig, 3))
    phases_gen_set  = set(np.round(phases_gen, 3))
    common_phases   = sorted(phases_orig_set & phases_gen_set)

    if len(common_phases) == 0:
        print('  WARNING: no common phases found, skipping')
        continue

    # extract magnitudes at common phases
    def get_mags_at_phases(phases_arr, mags_arr, target_phases):
        result = []
        for tp in target_phases:
            idx = np.argmin(np.abs(phases_arr - tp))
            result.append(mags_arr[idx])
        return np.array(result)

    common_phases_arr = np.array(common_phases)
    mags_orig_common  = get_mags_at_phases(phases_orig, mags_orig, common_phases)
    mags_gen_common   = get_mags_at_phases(phases_gen,  mags_gen,  common_phases)

    # -----------------------------------------------------------------------
    # PLOT: 2 rows x 5 cols
    # Top row: light curves overlaid
    # Bottom row: residuals (generated - original)
    # -----------------------------------------------------------------------

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 5, height_ratios=[2.5, 1], hspace=0.08, wspace=0.35)

    axes_lc  = [fig.add_subplot(gs[0, j]) for j in range(5)]
    axes_res = [fig.add_subplot(gs[1, j], sharex=axes_lc[j]) for j in range(5)]

    # compute offset to put both on the same scale
    # use the median of the original magnitudes at physically meaningful phases
    # as a reference -- subtract it so both curves are on the same relative scale
    # (this is necessary because the AB zero point may differ from the original system)
    offsets = []
    for j, fname in enumerate(filter_labels):
        if mags_orig.shape[1] <= j:
            offsets.append(0.)
            continue
        valid_orig = mask_baseline(mags_orig_common[:, j])
        valid_gen  = mask_baseline(mags_gen_common[:, j])
        valid_both = valid_orig & valid_gen
        if np.sum(valid_both) > 0:
            # offset = median difference between generated and original
            # at physically meaningful phases
            diff = mags_gen_common[valid_both, j] - mags_orig_common[valid_both, j]
            offsets.append(np.nanmedian(diff))
        else:
            offsets.append(0.)

    median_offset = np.nanmedian(offsets)
    print('  Median magnitude offset (generated - original): %.3f mag' % median_offset)

    for j, fname in enumerate(filter_labels):
        ax_lc  = axes_lc[j]
        ax_res = axes_res[j]
        color  = filter_colors[fname]

        if mags_orig.shape[1] <= j or mags_gen.shape[1] <= j:
            ax_lc.text(0.5, 0.5, 'No data', transform=ax_lc.transAxes,
                       ha='center', va='center')
            continue

        # masks for physically meaningful magnitudes
        valid_orig = mask_baseline(mags_orig_common[:, j])
        valid_gen  = mask_baseline(mags_gen_common[:, j])

        # --- light curve panel ---
        # plot original
        ax_lc.plot(common_phases_arr[valid_orig],
                   mags_orig_common[valid_orig, j],
                   color=color, lw=2, ls='-',
                   label='Original .DAT')

        # plot generated, shifted by median offset for visual comparison
        ax_lc.plot(common_phases_arr[valid_gen],
                   mags_gen_common[valid_gen, j] - median_offset,
                   color=color, lw=1.5, ls='--', alpha=0.8,
                   label='Generated (offset %.2f)' % median_offset)

        ax_lc.invert_yaxis()
        ax_lc.set_title('SDSS %s' % fname, fontsize=10,
                         color=color, fontweight='bold')
        ax_lc.set_xlim(min(common_phases) - 5, max(common_phases) + 5)
        ax_lc.set_ylim(-14, min(mags_orig_common[valid_gen, j])-0.5)
        ax_lc.grid(alpha=0.2)
        ax_lc.tick_params(labelbottom=False)
        if j == 0:
            ax_lc.set_ylabel('Absolute magnitude', fontsize=9)
            ax_lc.legend(fontsize=7, loc='lower right')

        # --- residual panel ---
        valid_both = valid_orig & valid_gen
        if np.sum(valid_both) > 0:
            residuals = (mags_gen_common[valid_both, j] - median_offset) - \
                         mags_orig_common[valid_both, j]
            ax_res.plot(common_phases_arr[valid_both], residuals,
                        color=color, lw=1.5, marker='o',
                        markersize=2, alpha=0.8)
            ax_res.axhline(0., color='k', lw=0.8, ls='--')
            ax_res.axhline(+0.1, color='gray', lw=0.5, ls=':', alpha=0.5)
            ax_res.axhline(-0.1, color='gray', lw=0.5, ls=':', alpha=0.5)
            ax_res.set_ylim(-0.5, 0.5)
            ax_res.grid(alpha=0.2)

            # accumulate residuals for summary statistics
            all_residuals[fname].extend(residuals.tolist())

            # print per-filter stats
            print('  SDSS %s: residual mean=%.3f, std=%.3f, max=%.3f mag' % (
                  fname,
                  np.nanmean(residuals),
                  np.nanstd(residuals),
                  np.nanmax(np.abs(residuals))))

        if j == 0:
            ax_res.set_ylabel('Residual\n(gen - orig)', fontsize=8)
        ax_res.set_xlabel('Phase (days)', fontsize=9)

    fig.suptitle('%s  (%s)    median offset = %.3f mag' % (
                 model_name, sntype_orig, median_offset),
                 fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(diag_dir, '%s_dat_comparison.png' % model_name),
                bbox_inches='tight', dpi=120)
    plt.close(fig)
    print('  Plot saved.')

# -----------------------------------------------------------------------
# SUMMARY PLOT: residual distributions across all models
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)

for j, fname in enumerate(filter_labels):
    ax    = axes[j]
    color = filter_colors[fname]
    resid = np.array(all_residuals[fname])
    resid = resid[np.isfinite(resid)]

    if len(resid) == 0:
        continue

    ax.hist(resid, bins=40, color=color, alpha=0.7, edgecolor='k')
    ax.axvline(0.,              color='k',    lw=1.5, ls='--')
    ax.axvline(np.median(resid), color='red',  lw=1.5, ls='-',
               label='Median = %.3f' % np.median(resid))
    ax.axvline(np.mean(resid),   color='orange', lw=1.5, ls='-.',
               label='Mean = %.3f' % np.mean(resid))
    ax.set_xlabel('Residual (gen - orig) (mag)', fontsize=10)
    ax.set_title('SDSS %s\n(N=%d phases)' % (fname, len(resid)),
                 fontsize=10, color=color, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    if j == 0:
        ax.set_ylabel('N', fontsize=10)

    print('\nSummary SDSS %s: median=%.3f, mean=%.3f, std=%.3f, max=%.3f mag' % (
          fname,
          np.median(resid), np.mean(resid),
          np.std(resid), np.max(np.abs(resid))))

fig.suptitle('Residual distributions: generated vs original .DAT files\n'
             'across all %d models (after median offset correction)' % len(common_models),
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(diag_dir, 'summary_residual_distributions.png'),
            bbox_inches='tight', dpi=150)
plt.close(fig)

print('\nAll plots saved to %s' % diag_dir)
print('Done.')