#!/usr/bin/env python

'''
This script calculates rates from an observed number distribution
with redshift
'''


import os,sys,pdb,scipy,glob, time, pickle
from pylab import *
from scipy.optimize import curve_fit
from scipy.integrate import quad
import util as u
import diagnostic_plot_util as plot_util
import rates_z_new as rz
import imf
import volume, control_time, cosmocalc
import warnings#,exceptions
warnings.simplefilter("error",RuntimeWarning)
warnings.filterwarnings("ignore")
import multiprocessing
import json, logging
import pandas as pd
from scipy import stats
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.io.ascii as ascii


def snrates(z,*p):
    A,B,C,D=p
    k = 0.007 ## for low-mass core-collapse supernovaae
    return(1e4*k*A*((1+z)**C)/(1+((1+z)/B)**D))

def cc_snrates(z,vol_frac):
    #k = 0.0091*(.70)**2. ## for low-mass core-collapse supernovaae
    k = 0.007 ## for low-mass core-collapse supernovaae
    #return(1e4*k*vol_frac[type]*rz.sfr_2020(z))
    return(1e4*k*vol_frac*rz.sfr_2020(z)) 

def snrates_Ia(z):
    ##data = loadtxt('SNRmodelTable.dat')
    data = loadtxt('LGSfitTable.dat')
    try:
        junk, yy = u.recast(z,0.,data[:,0], data[:,1])
    except:
        pdb.set_trace()
    yy = [x if x > 0.0 else 0.0 for x in yy]
    return(yy)

def make_cadence_table(types,redshift,tess_sens=19):
    rise_times = {
        'iip':20,
        }
    rise = rise_times['iip']*(1.+redshift)
    N = int(365./rise)
    
    cadence_table = 'cadences.txt'
    if os.path.isfile(cadence_table): os.remove(cadence_table)
    f = open(cadence_table,'w')
    f.write("#Cadence(days)  Area(sqarcmin)  Sens(iA) Prev_baseline\n")
    data = []
    for i in range(N):
        f.write("%d %4.2f %3.1f %d\n" %(rise, tess_area, tess_sens, rise))
        data.append([int(rise),float(tess_area),float(tess_sens), int(rise)])
    data = array(data)
    f.close()
    return(data)
    

def get_unique_visits(survey):
    temp=[]
    for item in survey:
        temp.append('_'.join(item.astype('str')))
    out=[]
    temp2 = sorted(set(temp), key = lambda x: float(x.split("_")[1]), reverse=True)
    for i,item in enumerate(temp2):
        out.append(list(map(float,item.split('_')))+ [temp.count(item)])
    return(array(out))

def fline(x,*p):
    m,b = p
    return m*x+b

def det_eff(delta_mag,mc=30, T=1.0, S=0.3):
    result=T/(1+exp((delta_mag-mc)/S))
    return(result)

def load_color_corrections(color_cor_file):
    """
    Reads per-subtype peak color corrections from a text file.

    Parameters
    ----------
    color_cor_file : str
        Path to the color corrections text file.

    Returns
    -------
    color_corrections : dict
        Nested dictionary mapping sntype -> central_wave_nm -> color_cor_mag.
        e.g. color_corrections['iip'][619] = -0.31
    """
    color_corrections = {}
    with open(color_cor_file, 'r') as f:
        for line in f.readlines():
            # skip comment and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            sntype   = parts[0]
            cen_wave = int(parts[1])
            try:
                color = float(parts[2])
            except ValueError:
                color = np.nan
            if sntype not in color_corrections:
                color_corrections[sntype] = {}
            color_corrections[sntype][cen_wave] = color
    return color_corrections

def load_absmags(absmags_file):
    """
    Reads peak absolute magnitudes in SDSS g-band (AB) from a text file.

    Parameters
    ----------
    absmags_file : str
        Path to the absolute magnitudes text file.

    Returns
    -------
    absmags : dict
        Dictionary mapping sntype to [mean_mag, sigma, scatter].
        e.g. absmags['iip'] = [-16.71, 0.97, 0.37]
    """
    absmags = {}
    with open(absmags_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            sntype  = parts[0]
            mean    = float(parts[1])
            sigma   = float(parts[2])
            scatter = float(parts[3])
            absmags[sntype] = [mean, sigma, scatter]
    return absmags

def load_vol_frac(name=None, vol_frac_file=None):
    with open(vol_frac_file) as f:
        data = json.load(f)
    if name is None:
        name = data.get('_default', 'li_2011')
    if name not in data or name.startswith('_'):
        raise KeyError("Unknown vol_frac_set %r; options: %s"
                       % (name, [k for k in data if not k.startswith('_')]))
    vf = {k: v for k, v in data[name].items() if not k.startswith('_')}
    cc_keys = [k for k in vf if k not in ('ia', 'slsn')]
    cc_sum  = sum([vf[k] for k in cc_keys])
    if cc_sum > 0:
        vf = {k: (v/cc_sum if k in cc_keys else v) for k, v in vf.items()}
    return vf


def run(redshift2, redshift1, rate_guess, Nobs, diag_dir, base_root, sndata_root, lightcurve_path,sed_path,
        types,passband,survey,m50=30,T=1.0,S=0.30,Nproc=1,extinction=True,obs_extin='nominal',
        verbose=True, parallel=True, box_tc=True, passskiprow=1, passwavemult=0.1,
        dstep=0.5, dmstep=0.1, dastep=0.1, lc_smoothing_window=3,
        biascor=None, vol_frac=None,
        cosmology=None, review = False, ratefile=None, eventtable=None, color_corrections=None,
        absmags=None):

    '''
    Computes the supernova rate and expected number of detections for a single
    redshift bin by integrating the control time over the survey cadence.

    Parameters
    ----------
    redshift2: float
        Upper redshift bin edge
    redshift1: float
        Lower redshift bin edge
    rate_guess: float
        Initial rate guess for input redshift bin, based on cosmic SFRD; used to compute N_exp
    Nobs: float
        Observed number of SNe in your redshift bin from your survey
    diag_dir: str
        Name attached to your output diagnostic plots directory (e.g., 'diagnostic_plots/<run_name>/')
    base_root: str
        Absolute path to your BoomRate directory
    sndata_root: str
        Absolute path to your SNANA directory
    lightcurve_path: str
        Absolute path to your lightcurve files
    types: list
        List of SN types to compute rates for
    passband: str
        Path to the observed filter transmission file
    m50 : float, optional
        Magnitude at which your survey recovers 50% of sources. Default is 28.5
    S: float, optional
        Fitted parameter describing how rapidly your detection efficiency curve falls off. Default is 0.30
    survey: array
        Cadence array loaded from the cadence file. 
    Nproc: int, optional
        Number of parallel processors used to run the calculation. Default is 1
    extinction: bool, optional
        Whether to include host galaxy extinction in the control time calculation. Default is True
    obs_extinc: str,optional
        Observational extinction treatment. Default is 'nominal'
    verbose: bool, optional
        Whether to print progress updates during the calculation.
    parallel: bool, optional
        Whether to run the kcor calculation in parallel. Default is True.
    box_tc: bool, optional
        If True, computes control times at both redshift bin edges and 
        integrates linearly between them for a more accurate average.
        If False, computes at the bin midpoint only. Default is True.
    passskiprow: int, optional
        Number of header rows to skip when reading the passband file. Default is 1.
    passwavemult: float, optional
        Wavelength multiplier for the passband file to convert to Angstroms. Default is 0.1.
    dstep: float, optional
        Step size in days for the SN age grid. Default is 0.5.
    dmstep: float, optional
        Step size in magnitudes for the luminosity function grid. Default is 0.1.
    dastep: float, optional
        Step size in magnitudes for the extinction grid. Default is 0.1.
    biascor: str or None, optional
        Bias correction method to apply ('flat', 'malmquist', or 'fractional'). Default is None.
    subtype_combination: str, optional
        Method for combining control times across subtypes. 'forward' sums
        fractionally weighted control times; 'divide_average' averages them.
        Default is 'divide_average'.
    vol_frac: dict, optional
        The dictionary of volumetric fractions to use, keyed by SN subtype. Default is None.
    cosmology: list
        List containing cosmological parameters; should be [H0, Omega_m, Omega_Lambda]
        Default is None
    review: bool, optional
        If True, generates diagnostic plots of the kcor and light curves. Default is False.
    ratefile: str, optional
        Path to the output rate text file where results are appended. Default is None.
    eventtable: str, optional
        Path to the SN classification table. Default is None.

    Returns
    -------
    Nexp: float
        Expected number of SNe detectable by the survey in this redshift bin, given the rate guess.
    Nexp_hi: float
        Upper 1-sigma Poisson error on Nexp (Gehrels 1986).
    Nexp_lo: float
        Lower 1-sigma Poisson error on Nexp (Gehrels 1986).
    tc_tot: float
        Total weighted control time in rest-frame years, summed across all survey visits and SN subtypes.
    '''
    
    Nexp_subtype={}
    redshift = (redshift2+redshift1)/2.
    print('z = %2.2f | Rate Guess = %2.2f | N_obs = %2.1f' %(redshift, rate_guess, Nobs))
    rate_guess = rate_guess*1.0e-4

    if len(shape(survey))==1:
        survey = append(survey, 1.)
        survey = array([list(survey)])
    else:
        # Deduplicate the cadence entries and attaches a count of how many times each 
        # unique visit configuration appears. This is what produces the multiplier that 
        # gets used later in the loop.
        survey = get_unique_visits(survey)
    _cosmo = cosmocalc.resolve_cosmology(cosmology)
    Dvol = volume.run(redshift2, **_cosmo)-volume.run(redshift1, **_cosmo)    

    # Diagnostic: per-subtype contribution to tc_tot under the active method.
    tc_per_type = {t: 0.0 for t in types}

    tc_tot=0
    for i,item in enumerate(survey):
        print("Computing SN Rates for Epoch %s" % i)
        baseline=item[0]
        area = item[1]

        # Convert the survey area from arcminutes to fraction of the total sky
        area_frac = area * (1./60.)**2*(pi/180)**2*(4.0*pi)**(-1)

        prev = item[3]
        for type in types:

            if not box_tc:
                tc = control_time.run(redshift, baseline, m50=m50, T=T, S=S, Nproc=Nproc, parallel=parallel,
                                      extinction=extinction, obs_extin=obs_extin,
                                      type=[type], prev=prev, passband=passband,
                                      passwavemult=passwavemult, passskiprow=passskiprow,
                                      dstep=dstep, dmstep=dmstep, dastep=dastep,
                                      lc_smoothing_window=lc_smoothing_window, color_corrections=color_corrections,
                                      biascor=biascor, cosmology=cosmology, review=review,
                                      verbose=verbose, absmags=absmags,
                                      base_root=base_root, sndata_root=sndata_root, lightcurve_path=lightcurve_path, 
                                      sed_path=sed_path,diag_dir=diag_dir)
            else:
                print("Calculating Control Time 1 at z=%s for %s" % (redshift1, type))
                print("-"*60)
                tc1 = control_time.run(redshift1, baseline, m50=m50, T=T, S=S, Nproc=Nproc, parallel=parallel,
                                       extinction=extinction, obs_extin=obs_extin,
                                       type=[type], prev=prev,passband=passband,
                                       passwavemult=passwavemult, passskiprow=passskiprow,
                                       dstep=dstep, dmstep=dmstep, dastep=dastep,
                                       biascor=biascor, cosmology=cosmology, review=review,
                                       verbose=verbose, color_corrections=color_corrections, absmags=absmags,
                                       base_root=base_root, sndata_root=sndata_root, lightcurve_path=lightcurve_path, 
                                       sed_path=sed_path,diag_dir=diag_dir)
                print("")
                print("Calculating Control Time 2 at z=%s for %s" % (redshift2, type))
                tc2 = control_time.run(redshift2, baseline, m50=m50, T=T, S=S, Nproc=Nproc, parallel=parallel,
                                       extinction=extinction, obs_extin=obs_extin,
                                       type=[type], prev=prev,passband=passband,
                                       passwavemult=passwavemult, passskiprow=passskiprow,
                                       dstep=dstep, dmstep=dmstep, dastep=dastep,
                                       biascor=biascor, cosmology=cosmology, review=review,
                                       verbose=verbose, color_corrections=color_corrections, absmags=absmags,
                                       base_root=base_root, sndata_root=sndata_root, lightcurve_path=lightcurve_path, 
                                       sed_path=sed_path,diag_dir=diag_dir)
                
                xx =array([redshift1, redshift2])
                yy = array([tc1,tc2])
                yy[isnan(yy)]=0.0 ## remove any nans
                p0=[1.0,0.0]
                pout = curve_fit(fline,xx,yy,p0=p0)[0]
                tc = quad(fline,xx[0],xx[1],args=tuple(pout))[0]/diff(xx)
                tc = tc[0]

            rel_num = 1.*(vol_frac[type])
            exp_num = rate_guess*rel_num*(tc*Dvol*area_frac)
            print("="*60)
            print("Mean Control Time of type %s = %4.2f rest frame days" %(type.upper(), tc*365.25))
            print("Volume probed = %.2f x10^4 Mpc^3 @ z= %.2f" %(area_frac*Dvol/1e4, redshift))
            print("="*60)
            print("")
            print("")

            try:
                multiplier = item[-1]
            except:
                multiplier = 1.0
            try:
                Nexp_subtype[type]+=(exp_num*multiplier)
            except:
                Nexp_subtype[type]=exp_num*multiplier
            tc_tot += tc*multiplier
            tc_per_type[type] += tc * multiplier
        print("")
        print("")
        print("\n")

    print('-------\n')

    # Get the expected number of events by summing the expected number of each subtype
    Nexp = sum(list(Nexp_subtype.values()))

    print('Redshift= %2.2f (%2.2f - %2.2f)' %(redshift,redshift1, redshift2))
    if Nexp > 1.0:
        Nexp_hi,Nexp_lo = poisson_error(Nexp)
    elif Nexp == 0:
        Nexp_hi = 0
        Nexp_lo = 0
    else:
        temp = 1.0/Nexp
        Nexp_hi,Nexp_lo = poisson_error(1.0)
        Nexp_hi=Nexp_hi/temp
        Nexp_lo=Nexp_lo/temp

    Nobs_hi, Nobs_lo = poisson_error(Nobs)

    Robs, Rerr_hi, Rerr_lo = 0.0, 0.0, 0.0
    for type in types:
        # Get your relative subtype fraction
        rel_num = 1.*(vol_frac[type])
        # Get your subtype's control time (years)
        tc_subtype = tc_per_type[type]

        # compute per-subtype central rate and errors
        denom = tc_subtype * Dvol * area_frac
        R_central = (Nobs    * rel_num / denom) * 1e4
        R_hi      = (Nobs_hi * rel_num / denom) * 1e4
        R_lo      = (Nobs_lo * rel_num / denom) * 1e4

        Robs     += R_central
        Rerr_hi  += R_hi - R_central  # upper error on this subtype's contribution
        Rerr_lo  += R_central - R_lo  # lower error on this subtype's contribution


    print('Total Control Time = %.2f days' %(tc_tot*365.25))
    print('Rate from %2.1f expected events to %2.1f mag: R=%2.2f+%2.2f-%2.2f' %(Nobs, m50, Robs, Rerr_hi, Rerr_lo))
    print('Number expected to %2.1f mag from expected rate of R=%2.2f: N=%2.1f+%2.1f-%2.1f' %(m50, rate_guess*1e4, Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo))
    print('-------\n')
    print('%.2f   %.2f   %.2f   %.2f   %.2f   %.2f   %.1f   %.1f   %.1f   %.1f'
          %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo, Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo, Nobs))
    print('[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f],'
          %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo))
    f=open(ratefile,'a')
    f.write('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.3f,%.2f,%.2f\n'
            %(redshift, redshift1, redshift2, Robs, Rerr_hi, Rerr_lo,
              Nexp, Nexp_hi-Nexp, Nexp-Nexp_lo, Nobs, tc_tot,
              area_frac*Dvol*1e-4, rate_guess*1e4)
            )
    f.close()
    if review:
        plot_util.plot_tc_per_type(tc_per_type, redshift1, redshift2, #subtype_combination,
                                   diag_dir, mag=m50)
    return([Nexp, Nexp_hi, Nexp_lo, tc_tot])
    

    
def poisson_error(n):
    #from table in Gehrels (1986), where CL are determined from Newton's method solution.
    if type(n) is not ndarray:
        n = array([n])
    ul =array([
        [0,1.841],
        [1,3.300],
        [2,4.638],
        [5,8.382],
        [10,14.27],
        [20,25.55],
        [40,47.38],
        [100,111.0],
        ])
    ll =array([
        [0,0.0],
        [1,0.173],
        [2,0.708],
        [5,2.840],
        [10,6.891],
        [20,15.57],
        [40,33.70],
        [100,90.02],
        ])
    (junk,nul)=u.recast(n,0.0,ul[:,0],ul[:,1])
    (junk,nll)=u.recast(n,0.0,ll[:,0],ll[:,1])
    return(nul[0],nll[0])


def main(configfile=None):

    '''
    Top-level function for the BoomRate supernova rate calculator. Reads
    configuration from a JSON file, computes probabilistic redshift
    distributions, defines redshift bins, computes an initial rate guess,
    and iterates over redshift bins and survey visits to compute the
    observed supernova rate in each bin.
    '''

    if not configfile:
        print("No configfile specified\n")
        return() 
    with open(configfile) as data_file:
        config = json.loads(data_file.read())

    run_name = config['run_name']
    base_root = config['base_root']
    sndata_root = config['sndata_root']
    spectral_template_ref = config['spectral_template_ref']
    sed_path = sndata_root + '/models/NON1ASED/' + spectral_template_ref
    lightcurve_path = base_root + '/broadband_lightcurves/' + spectral_template_ref

    clobber = json.loads(config['clobber'])
    verbose = json.loads(config['verbose'])

    #review
    review = json.loads(config['review'])
    
    #survey
    sntypes = config['sntypes']
    imf_evol = config['imf_evol']
    extinction = json.loads(config['extinction'])
    obs_extin = config['obs_extin']
    biascor = config['biascor']
    cosmology = config.get('cosmology', None)        # None -> use default in cosmologies.json

    cadence_file=config['cadence_file']
    itermag = json.loads(config['itermag'])
    band = config['band']
    passband = config['passband']
    passskiprow = config['passskiprow']
    passwavemult = config['passwavemult']

    # Mg_to_MX = M_X(AB) - M_g(AB)
    # Applied as: apparent_mag = lc + dm + Mg_to_Mx + Mx_to_MQ + AQ + mu -2.5log10(1+z)
    # Positive = X fainter than g, Negative = X brighter than g
    color_cor_file = config.get('color_correction_file', None)
    if color_cor_file is None:
        raise ValueError(
            "No color_cor_file specified in config -- "
            "a peak color correction file must be provided via 'color_correction_file'."
        )
    color_corrections = load_color_corrections(color_cor_file)


    absmags_file = config.get('peak_absmags_file', None)
    if absmags_file is None:
        raise ValueError(
            "No absmags_file specified in config -- "
            "an absolute magnitudes file must be provided via 'absmags_file'."
        )
    absmags = load_absmags(absmags_file)

    vol_frac_set = config.get('vol_frac_set', None)
    if vol_frac_set is not None:
        vol_frac_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vol_fractions.json')
        vol_frac = load_vol_frac(vol_frac_set, vol_frac_file)

    else:
        raise ValueError(
            "No vol_frac_set specified in config -- "
            "a volumetric fraction file must be provided via 'vol_frac_set'."
        )
    
    outfile_rates = config['outfile_rates']
    outfile_numbers = config['outfile_numbers']
    multiproc = json.loads(config['multiproc'])
    
    #sample table
    eventtable = config['eventtable']
    determinate = json.loads(config['determinate'])

    # detection efficiency table
    det_eff_tab = config['det_eff_table']

    #fake binned SNe
    falseevents = json.loads(config['falseevents'])
    falsetable = config['falsetable']
    if not falsetable:
        falsetable = config['outfile_numbers']

    Nbins = config['nbins']
    redshift_binning = config['redshift_binning']
    if not Nbins: Nbins = 3

    if multiproc:
        Nproc=int(multiprocessing.cpu_count()-2)
    else:
        Nproc=1
    if falseevents and determinate: determinate=False    


    ## some for setting the fineness of SN parameter eval
    dstep=config['day_step']
    lc_smoothing_window=config['lc_smoothing_window']
    dmstep=config['abs_mag_step']
    dastep=config['extinction_step']
    box_tc=json.loads(config['box_tc'])

    # If review is True, generate a directory for this run within diagnostic plots
    if review:
        diag_dir = 'diagnostic_plots/%s' % run_name
        if not os.path.isdir(diag_dir):
            os.makedirs(diag_dir)
    else:
        diag_dir=None

    # Runs if your output file either does not exist or you are fine with replacing it
    if not os.path.isfile(outfile_rates) or clobber:
        if falseevents and os.path.isfile(falsetable):
            tmp = pickle.load(open(falsetable,'rb'))
            redshifts = tmp[:,-2]
            redshifts=append(redshifts,tmp[-1,-1])
            #ng = tmp[:,1]
            Nobs= tmp[:,1]
        elif falseevents:
            print('No event table')
            print('Generating...')
            ## redshifts = arange(5, 10.5, 0.5)
            a1 = arange(0,0.5,0.1) #to get a bit more resolution at low-z
            a2 = arange(0.5,10,0.5) #resolution not necessary at high-z
            redshifts=concatenate((a1,a2),)
            if redshifts[0]==0: redshifts[0]=0.001 #redhift==0 problem
            #ng = ones(len(redshifts)-1)
            Nobs = ones(len(redshifts)-1)
            ## the following might be commented out for testing
            if redshift_binning is None:
                redshift_bins = stats.mstats.mquantiles(redshifts, splits)
            else:
                redshift_bins = array(redshift_binning)
            #ng, redshifts = histogram(redshifts, redshift_bins)
            Nobs, redshifts = histogram(redshifts, redshift_bins)
        else:
            sn_table = pd.read_csv(eventtable, sep='\t')
            Nbins = Nbins
            splits = arange(0, 1+1./Nbins, 1./Nbins).tolist()
            if  max(sn_table['pIa']+sn_table['pII']+sn_table['pIbc']) > 99:
                sn_table['pIa']=sn_table['pIa'].apply(lambda x: x*0.01)
                sn_table['pII']=sn_table['pII'].apply(lambda x: x*0.01)
                sn_table['pIbc']=sn_table['pIbc'].apply(lambda x: x*0.01)

            if determinate:
                if 'ia' in sntypes:
                    sn_condition = sn_table['pIa']>0.5
                elif (('iil' in sntypes) or ('iip' in sntypes)
                      or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
                    sn_condition1 = sn_table['pII']>0.5
                    sn_condition2 = sn_table['pIbc']>0.5
                    sn_condition = sn_condition1 | sn_condition2
                else:
                    print('Only set for SNe Ia and CCSNe\n')
                    pdb.set_trace()
                sne_select = sn_table.where(sn_condition)
                sne_select['Redshift']=sne_select['z_host']
                if redshift_binning is None:
                    redshift_bins = stats.mstats.mquantiles(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], splits)
                else:
                    redshift_bins = array(redshift_binning)
                ### need a more elegant counter than this soon.
                #ng, redshifts = histogram(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], redshift_bins)
                Nobs, redshifts = histogram(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])], redshift_bins)

                ax = subplot(111)
                style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
                #ax.hist(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])],
                #        redshift_bins, label='%.1f total'%(sum(ng)),**style)
                ax.hist(sne_select['Redshift'][~np.isnan(sne_select['Redshift'])],
                        redshift_bins, label='%.1f total'%(sum(Nobs)),**style)
                ax.set_xlabel('Redshift')
                ax.set_xlim(0,6)
                if 'ia' in sntypes:
                    ax.set_ylabel('Number of SNeIa')
                else:
                    ax.set_ylabel('Number of CCSNe')
                ax.legend(loc=1)
                savefig('diagnostic_plots/%s_redshift_dist.png' % run_name)
                clf()
                
            else: #not determinate

                ####################################################################
                # Computing the Summed Redshift Probability Distribution for all SNe
                ####################################################################

                # Set the redshift step size
                dzza=0.001

                redshift_tmp=arange(0,10+dzza,dzza)
                rv = zeros(len(redshift_tmp))

                for index, row in sn_table.iterrows():
                    if row['pIa']==row['pII']==row['pIbc']==-0.99: continue
                    if row['pIa']==row['pII']==row['pIbc']==-99: continue

                    zz = row['z_host']
                    dzz = row['z_host_err']

                    # For spec-zs (where 'z_host_err' is -99 or NaN or 0), set redshift uncertainty to 0.001 (to avoid infinitely narrow Gaussians)
                    if ((dzz == -99.0) | (isnan(dzz)) | (dzz==0.0)): dzz = dzza

                    if 'ia' in sntypes:
                        if isnan(row['pIa']): continue
                        # Build a Gaussian probability distribution centered at host_z (zz), with host_z_err as width (dzz), and weighted by Ia probability
                        rv_tmp = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIa']
                    elif (('iil' in sntypes) or ('iip' in sntypes)
                          or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):
                        rv_tmp1 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pII']
                        rv_tmp2 = stats.norm.pdf(redshift_tmp, loc=zz, scale=dzz)*row['pIbc']
                        stack = dstack((rv_tmp1, rv_tmp2))
                        rv_tmp = nansum(stack, axis=2)[0]
                        ## rv_tmp=rv_tmp1+rv_tmp2 ## this method doesn't handle nan's all that well
                    else:
                        print('Only set for SNe Ia and CCSNe\n')

                    # Stacking all of the individual source redshift probablity distributions on top of each other
                    if not isnan(sum(rv_tmp)):
                        rv+=rv_tmp

                # Computing the redshift cumulative probability distribution
                pv = cumsum(rv)*dzza#/sum(rv)

                if redshift_binning is not None:
                    bins = array(redshift_binning)

                ############################################
                # Setting the Non-Determinate Redshift Bins 
                ############################################

                else:
                    bins=zeros(Nbins+1) # bin edges array
                    for i,split in enumerate(splits): # e.g., if Nbins=4, splits is [0, 0.25, 0.5, 0.75, 1.0]
                        if i==0: continue
                        ii =where(pv/pv[-1]<=split) # pv/pv[-1] normalizes cumulative distribution to [0,1]
                        try:
                            # Set the bin edge to the redshift at which exactly 'split' fraction of the sample has been accumulated
                            bins[i]=redshift_tmp[ii][-1]
                        except:
                            pdb.set_trace()

                    # Set the lower edge of the first bin to the redshift where the cumulative distribution first becomes non-negligible
                    ii = where(pv/pv[-1] < 0.001)
                    bins[0]=redshift_tmp[ii][-1]

                    # Sets the upper edge of the last bin to the redshift where the cumulative distribution reaches negligibly close to 1
                    ii = where((pv[-1]-pv)/pv[-1] < 0.001) #set to 1/1000 of a difference. Could be smaller.
                    bins[-1]=redshift_tmp[ii][0]

                # Determine the observed number of SNe are in each of your determinate redshift bins
                #ng = zeros(len(bins)-1,)
                Nobs = zeros(len(bins)-1,)
                for i,bin in enumerate(bins):
                    if i == 0: continue
                    ii = where((redshift_tmp>bins[i-1])&(redshift_tmp <= bins[i]))
                    #ng[i-1] = sum(rv[ii])*0.001
                    Nobs[i-1] = sum(rv[ii])*0.001
                redshifts = bins
        med_z = (redshifts[1:]+redshifts[:-1])/2.
        ## pdb.set_trace()

        # Analyzing any difference between cumulative redshift distribution total and sum of ng
        #print("pv[-1]:", pv[-1])
        #print("sum(rv)*dzza:", sum(rv)*dzza)
        ##print("sum(ng):", sum(ng))
        #print("sum(Nobs):", sum(Nobs))
        #print("pv at bins[-1]:", pv[argmin(abs(redshift_tmp - bins[-1]))])
        #print("pv at bins[0]:", pv[argmin(abs(redshift_tmp - bins[0]))])
        #print("pv at bins[-1] - pv at bins[0]:", pv[argmin(abs(redshift_tmp - bins[-1]))] - pv[argmin(abs(redshift_tmp - bins[0]))])

        # Redshift Distribution Diagnostic Plots
        if review:
            #plot_util.plot_redshift_dist(redshift_tmp, rv, dzza, redshifts, ng, sntypes, diag_dir)
            plot_util.plot_redshift_dist(redshift_tmp, rv, dzza, redshifts, Nobs, sntypes, diag_dir)


        ########################################################
        # Computing the Initial Rate Guess for each Redshift Bin 
        ########################################################
        
        if 'ia' in sntypes:
            rg = array(snrates_Ia(med_z))## for SNe Ia, from Strolger et al. 2020
        elif (('iil' in sntypes) or ('iip' in sntypes) or ('ic' in sntypes) or ('ib' in sntypes) or ('iin' in sntypes)):

            # Compute CCSN rate guess without evolving IMF
            if imf_evol is None:
                rg = zeros(len(med_z))
                for sntype in sntypes:
                    rg += cc_snrates(array(med_z), vol_frac[sntype])
                #rg = cc_snrates(array(med_z),sntypes[0]) # why is sn type required for this?

            # Compute CCSN rate guess with Dave evolving IMF
            elif imf_evol=='dave':
                print('Evol. k(z) testing...')
                sfrg = rz.sfr_2020(med_z)
                k_z=[]
                for zz in med_z:
                    mb = 0.5*(1.+zz)**2.
                    c3 = (1.+zz)**2.
                    p0 = [mb,c3]
                    num = quad(imf.kroupa,8,50,args=tuple(p0))[0] ## covering mass range of CCSNe
                    den = quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                    k_z.append(num/den)
                k_z = array(k_z)
                k_z = 1.e4*k_z
                rg = k_z * sfrg#*vol_frac[type]
            else:
                pdb.set_trace()
                  
        elif 'slsn' in sntypes:
            if imf_evol is None:
                #rg = cc_snrates(array(med_z),sntypes[0])
                #rg = rg * 2.2e-4 
                sfrg = rz.sfr_2020(med_z)
                ## k=quad(imf.salpeter,50,350)[0]/quad(imf.salpeter1,50,350)[0]
                p0=[0.5,1.]
                k = quad(imf.kroupa,50,350,args=tuple(p0))[0]/quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                rg =  1.e4*k*sfrg
                rg = rg *0.022
                
            elif imf_evol=='dave':
                print('Evol. k(z) testing...')
                sfrg = rz.sfr_2020(med_z)
                k_z=[]
                for zz in med_z:
                    mb = 0.5*(1.+zz)**2.
                    c3 = (1.+zz)**2.
                    p0 = [mb,c3]
                    num = quad(imf.kroupa,50,350,args=tuple(p0))[0] ## covering mass range of popIII SNe
                    den = quad(imf.kroupa1,0.1,350,args=tuple(p0))[0]
                    k_z.append(num/den)
                k_z = array(k_z)
                k_z = 1.e4*k_z ## the factor of 10 accounts for fewer really high mass stars in IMF
                rg = k_z * sfrg
                #rg = rg *10. ## a guess, really, at efficiency
                rg = rg *0.022 ## a guess, really, at efficiency
            ##if imf_evol=='chary': ## not yet ready on this
            else:
                pdb.set_trace()
                

        survey = loadtxt(cadence_file)

        # Get the survey sensitivity (in magnitudes) from the cadence file
        if itermag:
            mags = arange(24, 31, 2)
        else:
            try:
                mags = array(list(set(survey[:,2])))
            except:
                mags = array([survey[2]])
        ctr = 0

        # Use the input detection efficiency table to obtain best-fit parameters describing detection efficiency curve
        detection_efficiency_tab = ascii.read(det_eff_tab)
        band_mask = detection_efficiency_tab['band'] == band
        inj_mags = np.array(detection_efficiency_tab[band_mask]['inj_mag'])
        avg_recovery = np.array(detection_efficiency_tab[band_mask]['mean_det_eff'])
        m50_guess = inj_mags[np.argmin(abs(avg_recovery - 0.5))]
        popt, _ = curve_fit(det_eff, inj_mags, avg_recovery, p0=[m50_guess, max(avg_recovery), 0.3])
        m50_fit, T_fit, S_fit = popt 

        # Remove the output file if it already exists
        if os.path.isfile(outfile_rates): os.remove(outfile_rates)
        f=open(outfile_rates,'a')
        f.write('#z,z_low,z_hi,R,R_+err,R_-err,N_exp,N_+err,N_-err,Nobs,tc_rest,Dvol,R_g\n')
        f.close()
        for j,mag in enumerate(mags):
            for i,redshift in enumerate(redshifts):
                if i==0: continue
                ctr+=1
                print('\n\n Iteration %d of %d\n\n' %(ctr,len(mags)*(len(redshifts)-1)))
                print('Rate guess = %2.2f' %rg[i-1])
                num, nhi, nlo, tc = run(redshift2=redshifts[i], redshift1=redshifts[i-1], rate_guess=rg[i-1], 
                                        Nobs=Nobs[i-1],#ng[i-1],
                                        types=sntypes, passband=passband, m50=m50_fit, T=T_fit, S=S_fit,
                                        Nproc=Nproc,parallel=multiproc,
                                        verbose=verbose, extinction=extinction, obs_extin=obs_extin,
                                        survey = survey,
                                        box_tc=box_tc,
                                        dstep=dstep, dmstep=dmstep, dastep=dastep,
                                        lc_smoothing_window=lc_smoothing_window,
                                        passwavemult=passwavemult,
                                        passskiprow=passskiprow,
                                        eventtable=eventtable,
                                        #ratefile=out_rate_file,
                                        ratefile=outfile_rates,
                                        biascor=biascor,
                                        #subtype_combination=subtype_combination,
                                        vol_frac=vol_frac,
                                        color_corrections=color_corrections,
                                        absmags=absmags,
                                        cosmology=cosmology,
                                        review = review,
                                        diag_dir = diag_dir,
                                        base_root = base_root,
                                        sndata_root = sndata_root,
                                        lightcurve_path = lightcurve_path,
                                        sed_path=sed_path
                                        )
                #numbers.append([mag, num, nhi, nlo, (redshifts[i]+redshifts[i-1])/2., redshifts[i-1], redshifts[i]])
        #numbers=array(numbers)
        #if os.path.isfile(outfile) and clobber: os.remove(outfile)
        #pickle.dump(numbers,open(outfile,'wb'))
        #savetxt(outfile_numbers, numbers, header='mag,N_exp,N_exp_hi,N_exp_lo,z_mid,z_low,z_hi',delimiter=',')
    else:
        outdata = pickle.load(open(outfile,'rb'))


if __name__=='__main__':
    
    if sys.argv[1] and sys.argv[1].endswith('.json'):
        configfile='./'+sys.argv[1]
    else:
        configfile = './config.json'
    print("Proceeding with rate_estimator using %s" %(configfile))
    logging.info("Proceeding with rate_estimator using %s" %(configfile))
    main(configfile=configfile)
            
    



    
