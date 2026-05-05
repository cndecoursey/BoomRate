"""
Run control_time per subtype for the JADES F200W cc_base z=0.95-1.42 bin
and produce the tc_per_type stacked-bar diagnostic for ALL three
subtype-combination methods:

  - no_volfrac      : T_raw                 (biascor='flat'      -> rel_num=1)
  - divide_average  : T_raw / f_X           (biascor='fractional', subtype_combination='divide_average')
  - forward         : T_raw * f_X           (biascor='fractional', subtype_combination='forward')

We compute T_raw once per (subtype, redshift) with biascor='flat' and apply
the rel_num scaling ourselves for the other two methods. This matches the
math in control_time.run() exactly and saves running it 3x.
"""

import os, json, warnings
from numpy import array, linspace, isnan, diff
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.pyplot import subplots, tight_layout, savefig, clf, close, cm

import control_time
warnings.filterwarnings("ignore")

CONFIG = 'JADES/config_jades_260330_F200W_cc_base.json'
Z_LOW, Z_HIGH = 0.95, 1.42

def fline(x, *p):
    m, b = p
    return m * x + b

def get_unique_visits(survey):
    temp = ['_'.join(item.astype('str')) for item in survey]
    seen = sorted(set(temp), key=lambda x: float(x.split('_')[1]), reverse=True)
    return array([list(map(float, s.split('_'))) + [temp.count(s)] for s in seen])

def plot_tc_per_type(tc_per_type, types, method, sens, out_path):
    contributions = array([tc_per_type[t] for t in types]) * 365.25  # years -> days
    total = contributions.sum()
    print('  tc_tot (%s) = %.2f rest-frame days' % (method, total))

    palette = cm.tab10(linspace(0, 1, max(len(types), 3)))
    fig, ax = subplots(figsize=(5, 6))
    bottom = 0.0
    for i, t in enumerate(types):
        h = contributions[i]
        ax.bar(0, h, 0.6, bottom=bottom,
               color=palette[i % len(palette)], edgecolor='white', linewidth=0.5,
               label='%s  %.2f d' % (t.upper(), h))
        bottom += h

    ax.set_xticks([0])
    ax.set_xticklabels(['z=%.2f-%.2f' % (Z_LOW, Z_HIGH)])
    ax.set_ylabel('Control time (rest-frame days)')
    title = 'tc_tot = %.2f d  (%s)' % (total, method)
    if sens is not None:
        title += '  | maglim %.1f' % sens
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='Subtype contribution', frameon=False)
    ax.margins(x=0.4)
    tight_layout()
    savefig(out_path, bbox_inches='tight')
    print('  saved', out_path)
    clf(); close('all')

def main():
    with open(CONFIG) as f:
        cfg = json.load(f)

    base_root = cfg['base_root']
    sndata_root = cfg['sndata_root']
    model_path = cfg['model_path']
    types = cfg['sntypes']
    passband = os.path.normpath(os.path.join(os.path.dirname(CONFIG), cfg['passband']))
    passskiprow = cfg['passskiprow']
    passwavemult = cfg['passwavemult']
    extinction = json.loads(cfg['extinction'])
    obs_extin = cfg['obs_extin']
    dstep = cfg['day_step']
    dmstep = cfg['abs_mag_step']
    dastep = cfg['extinction_step']
    lc_smoothing_window = cfg.get('lc_smoothing_window', 3)
    vol_frac_set = cfg.get('vol_frac_set')
    cosmology = cfg.get('cosmology')

    cadence_path = os.path.normpath(
        os.path.join(os.path.dirname(CONFIG), cfg['cadence_file']))
    survey_raw = []
    with open(cadence_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            survey_raw.append(list(map(float, line.split())))
    survey = array(survey_raw)
    if len(survey.shape) == 1:
        from numpy import append as np_append
        survey = array([list(np_append(survey, 1.))])
    else:
        survey = get_unique_visits(survey)

    # rel_num per type, exactly as control_time.run computes it for biascor='fractional'
    vol_frac_local = control_time.load_vol_frac(vol_frac_set)
    rel_num = {t: float(vol_frac_local[t]) for t in types}
    print('vol_frac_set=%s  rel_num=%s' % (vol_frac_set, rel_num))

    diag_dir = 'diagnostic_plots/jades_260330_F200W_cc_base_methods'
    os.makedirs(diag_dir, exist_ok=True)

    # tc_raw_per_type accumulates T_raw (years), summed over cadence rows with multipliers
    tc_raw_per_type = {t: 0.0 for t in types}
    sens_used = None

    for item in survey:
        baseline = item[0]
        sens = item[2]
        prev = item[3]
        multiplier = item[-1] if len(item) > 4 else 1.0
        sens_used = sens

        for t in types:
            tc1 = control_time.run(Z_LOW, baseline, sens, type=[t], prev=prev,
                                   passband=passband, passwavemult=passwavemult,
                                   passskiprow=passskiprow,
                                   dstep=dstep, dmstep=dmstep, dastep=dastep,
                                   lc_smoothing_window=lc_smoothing_window,
                                   extinction=extinction, obs_extin=obs_extin,
                                   biascor='flat',
                                   subtype_combination='divide_average',
                                   vol_frac_set=vol_frac_set, cosmology=cosmology,
                                   parallel=True, Nproc=4,
                                   base_root=base_root, sndata_root=sndata_root,
                                   model_path=model_path, diag_dir=diag_dir,
                                   plot=False, verbose=False, review=False)
            tc2 = control_time.run(Z_HIGH, baseline, sens, type=[t], prev=prev,
                                   passband=passband, passwavemult=passwavemult,
                                   passskiprow=passskiprow,
                                   dstep=dstep, dmstep=dmstep, dastep=dastep,
                                   lc_smoothing_window=lc_smoothing_window,
                                   extinction=extinction, obs_extin=obs_extin,
                                   biascor='flat',
                                   subtype_combination='divide_average',
                                   vol_frac_set=vol_frac_set, cosmology=cosmology,
                                   parallel=True, Nproc=4,
                                   base_root=base_root, sndata_root=sndata_root,
                                   model_path=model_path, diag_dir=diag_dir,
                                   plot=False, verbose=False, review=False)

            xx = array([Z_LOW, Z_HIGH])
            yy = array([tc1, tc2])
            yy[isnan(yy)] = 0.0
            pout, _ = curve_fit(fline, xx, yy, p0=[1.0, 0.0])
            tc = quad(fline, xx[0], xx[1], args=tuple(pout))[0] / diff(xx)
            tc = float(tc[0]) if hasattr(tc, '__len__') else float(tc)

            tc_raw_per_type[t] += tc * multiplier
            print('  %s : tc1=%.4f tc2=%.4f mean=%.4f yr  (= %.2f rest-frame days)'
                  % (t, tc1, tc2, tc, tc * 365.25))

    # Build per-method tc_per_type from T_raw
    tc_methods = {
        'no_volfrac':     {t: tc_raw_per_type[t]               for t in types},
        'divide_average': {t: tc_raw_per_type[t] / rel_num[t]  for t in types},
        'forward':        {t: tc_raw_per_type[t] * rel_num[t]  for t in types},
    }

    print('\nGenerating plots:')
    for method, tc_per_type in tc_methods.items():
        out = '%s/tc_per_type_z%.2f-%.2f_%s.png' % (diag_dir, Z_LOW, Z_HIGH, method)
        plot_tc_per_type(tc_per_type, types, method, sens_used, out)

if __name__ == '__main__':
    main()
