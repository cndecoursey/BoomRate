#!/usr/bin/env python
#### Note: Update with Holwerda extinction


import os,sys,pdb,scipy,pickle,json
from pylab import *
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime, time
from matplotlib import dates as mdates
#from strolger_util import util as u
#from strolger_util import cosmocalc
import util as u
import cosmocalc
import volume
import glob
from pprint import pprint

import multiprocessing
from functools import partial

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

import warnings#,exceptions
warnings.simplefilter("error",RuntimeWarning)

rcParams['figure.figsize']=12,9
rcParams['font.size']=16.0


# Volumetric subtype fractions live in vol_fractions.json next to this module.
_VOL_FRAC_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vol_fractions.json')
# Per-subtype absolute magnitudes live in absmags.json next to this module.
_ABSMAGS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'absmags.json')

def load_vol_frac(name=None, path=None):
    with open(path or _VOL_FRAC_FILE) as f:
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

def load_absmags(name=None, path=None):
    with open(path or _ABSMAGS_FILE) as f:
        data = json.load(f)
    if name is None:
        name = data.get('_default', 'richardson_2014')
    if name not in data or name.startswith('_'):
        raise KeyError("Unknown absmag_set %r; options: %s"
                       % (name, [k for k in data if not k.startswith('_')]))
    return {k: v for k, v in data[name].items() if not k.startswith('_')}

vol_frac = load_vol_frac()  # module-level default (li_2011)
absmags = load_absmags()    # module-level default (richardson_2014)

template_peak = { ##the assumed normalization for the SNANA templates
    'iip': -16.05,
    'iin': -17.05,
    'iil': -16.33,
    'ib' : -15.05,
    'ic' : -15.05,
    'ibc': -15.05,
    'ia' : -19.46,
    'slsn': -21.7,
    }


# For color_cor_Ia, the reference filter is 551 nm (Bessell V-band)
color_cor_Ia={
    360:1.8,
    442:0.15,
    551:0.0,
    663:-0.61,
    806:-0.56
    }

# For color_cor_gen (CCSNe), the reference filter is unclear?
color_cor_gen={
    356:2.7,
    472:0.15,
    619:-0.3,
    750:-0.61,
    896:0.8
    }

color_cor_slsn={
    356: 2.0,
    472: 0.15,
    }


def run(redshift, baseline, sens, base_root, sndata_root, model_path, diag_dir,
        type, dstep=3, dmstep=0.5, dastep=0.5, lc_smoothing_window=3,
        parallel=False, extinction=True, obs_extin=True, Nproc=23, prev=45.,
        passband = None, passskiprow=1, passwavemult=1000.,
        plot=False, verbose=False, review=False, biascor='flat',
        subtype_combination='divide_average', vol_frac_set=None, cosmology=None,
        absmag_set=None):

    # Resolve the subtype-fraction dict: per-call override, else module default.
    vol_frac_local = load_vol_frac(vol_frac_set) if vol_frac_set else vol_frac
    # Resolve the absolute-magnitude dict: per-call override, else module default.
    absmags_local = load_absmags(absmag_set) if absmag_set else absmags
    # Resolve cosmology: [H0, Om, Ol] from config, or {} to use module defaults.
    cosmo = cosmocalc.resolve_cosmology(cosmology)

    # Define the filters; important for later
    if verbose: print('defining restframe sloan filters...')

    # Build a dict that maps filter central wavelengths to filter file paths, which will
    # be used later to find the best matching rest-frame filter for the K-correction
    filter_dict={}
    if 'ia' in type:
        for bessel_filter in glob.glob(sndata_root +'/filters/Bessell90/Bessell90_K09/Bessell90_?.dat'):
            elam = get_central_wavelength(bessel_filter, wavemult=0.1) # wavemult converts Ang to nm
            filter_dict[elam]=bessel_filter
    else:
        for sdss_filter in glob.glob(sndata_root+'/filters/SDSS/SDSS_web2001/?.dat'):
            elam = get_central_wavelength(sdss_filter, wavemult=0.1) # wavemult converts Ang to nm
            filter_dict[elam]=sdss_filter

    # Observed filter
    if verbose: print('observed filter...')
    #if passband is not None:
    #    observed_filter = passband
    #else:
    #    #observed_filter=sndata_root+'/filters/JWST/NIRCAM/F444W_NRC_and_OTE_ModAB_mean.txt'
    #    #observed_filter=sndata_root+'/filters/HST/HST_GOODS/F850LP_ACS.dat'
    #    observed_filter=sndata_root+'/filters/HST/HST_Candles/ACS_WFC_F435W.dat'
    #    passwavemult=0.1
    observed_filter = passband # you are forced to specify a passband when calling run(); cannot use None
    ofilter_cen = get_central_wavelength(observed_filter,skip=passskiprow,wavemult=passwavemult)
    if verbose: print('observed filter effective wavelength= %4.1f nm'%ofilter_cen)

    # Rest-frame Lightcurve
    if 'slsn'in type:
        if verbose: print('getting best rest-frame lightcurve...')
        rest_age,rflc = rest_frame_slsn_lightcurve(dstep=dstep,verbose=verbose)
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)
        observed_frame_lightcurve = zeros((len(array(rflc[best_rest_filter])),5))
        observed_frame_lightcurve[:,0] = array(rflc[best_rest_filter]) - template_peak[type[0]]+absmags_local[type[0]][0]
    elif 'ia' not in type:
        if verbose: print('getting best rest-frame lightcurve...')

        # Load and interpolate non1a SED templates onto a uniform rest-frame age grid, keyed by filter central wavelength
        rest_age,rflc,models_used = rest_frame_lightcurve(type,model_path,sndata_root,dstep=dstep,verbose=verbose)

        # Plot the full restframe lightcurves for each model in used_models, for each available restframe filter
        if review:
            plot_restframe_lightcurves(rflc=rflc, rest_age=rest_age, models_used=models_used, type=type[0],
                                       diag_dir=diag_dir)

        # Find the SDSS filter that most closely matches the rest-frame wavelength of your observed filter
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)

        # Note: despite the name, this is still in the rest frame at this point.
        # The conversion to observed frame (K-correction + distance modulus) happens
        # inside the extinction/luminosity function loop below.
        observed_frame_lightcurve=mean_pop(mag_array=array(rflc[best_rest_filter]), review=review, 
                                           diag_dir=diag_dir, rest_age=rest_age, type=type[0])#-template_peak[type[0]]+absmags[type[0]][0]
        observed_frame_lightcurve_unsmoothed = observed_frame_lightcurve.copy()

        # Smooth the representative light curve along the age axis using a Gaussian kernel. 
        # Gaussian1DKernel(dstep) creates a Gaussian with a standard deviation of dstep age steps 
        # So with dstep=5, it smooths over a scale of 5 age steps × 5 days/step = 25 days.
        observed_frame_lightcurve[:,0]= convolve(observed_frame_lightcurve[:,0], Gaussian1DKernel(lc_smoothing_window), boundary='extend') 

        # Plot the representative light curve before/after smoothing to ensure smoothing is reasonable
        if review:
            plot_smoothing_diagnostic(observed_frame_lightcurve_unsmoothed=observed_frame_lightcurve_unsmoothed, 
                                      observed_frame_lightcurve_smoothed=observed_frame_lightcurve, 
                                      rest_age=rest_age, type=type[0], diag_dir=diag_dir, dstep=dstep)

        observed_frame_lightcurve_unanchored = observed_frame_lightcurve.copy()
        #lc_normalized_to_0 = observed_frame_lightcurve - template_peak[type[0]]
        lc_normalized_to_0 = observed_frame_lightcurve - nanmin(observed_frame_lightcurve[:,0])
        #observed_frame_lightcurve = observed_frame_lightcurve -template_peak[type[0]]+absmags[type[0]][0]

        # Anchor composite light curve to your choice of mean peak absolute magnitude by shifting its 
        # brightest point to absmags_local[type[0]][0]
        observed_frame_lightcurve = observed_frame_lightcurve - nanmin(observed_frame_lightcurve[:,0]) + absmags_local[type[0]][0]

        if review:
            plot_anchoring_diagnostic(observed_frame_lightcurve_unanchored, lc_normalized_to_0, observed_frame_lightcurve,
                                      rest_age, type[0], diag_dir, absmags_local)

    else:
        if verbose: print('getting best rest-frame lightcurve SNIA ...')
        rest_age, rflc = rest_frame_Ia_lightcurve(dstep=dstep,verbose=verbose)
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)
        observed_frame_lightcurve = zeros((len(array(rflc[best_rest_filter])),5))
        observed_frame_lightcurve[:,0] = array(rflc[best_rest_filter]) - template_peak[type[0]]+absmags_local[type[0]][0]


    ### kcorrecting rest lightcurve
    if verbose: print('kcorrecting rest-frame lightcurve...')

    model_pkl = 'SEDs_'+'_'.join(type)+'.pkl'
    if not os.path.isfile(model_pkl):
        pkl_file = open(model_pkl,'wb')
        if verbose: print('... loading model SEDs')
        models_used_dict={}
        total_age_set=[]
        if 'ia' in type:
            models_used = ['Hsiao07']#'Foley07_lowz_uhsiao']
            model_path = sndata_root +'/snsed'
        if 'slsn' in type:
            models_used = ['slsn_blackbody']

        for model in models_used:
            print('...... %s' %model)
            if 'ia' not in type:
                try:
                    data = loadtxt(os.path.join(model_path,model+'.SED'))
                except:
                    print('testing', os.path.join(model_path,model+'.SED'))
                    pdb.set_trace()
            else:
                data = loadtxt(os.path.join(model_path,model+'.dat'))

            #Extract the unique ages present in this model's SED data (column 0 is age) 
            # and stores the full SED array in models_used_dict keyed by model name.
            ages = list(set(data[:,0]))
            models_used_dict[model]=data

            # Accumulate all unique ages across all models into total_age_set. This is used later
            # to ensure the K-correction is computed at every age that any model has data for.
            for age in ages:
                if age not in total_age_set:
                    total_age_set.append(age)

        pickle.dump(models_used_dict,pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(model_pkl,'rb')
        if verbose: print('reading %s saved file' %model_pkl)
        models_used_dict = pickle.load(pkl_file, encoding='latin1')
        pkl_file.close()
        total_age_set=[]
        for model in models_used_dict.keys():
            ages = list(set(models_used_dict[model][:,0]))
            for age in ages:
                if age not in total_age_set:
                    total_age_set.append(age)
    
    # Read in your observed filter transmission curve
    f1 = loadtxt(observed_filter,skiprows=passskiprow)
    # Convert filter transmission curve wavelength to Angstrom (since k-correction works in Angstrom)
    f1[:,0]=f1[:,0]*passwavemult*10.
    
    # Read in your closest-matching rest-frame wavelength filter transmission curve
    f2 = loadtxt(filter_dict[best_rest_filter])

    # Select the appropriate color correction dictionary based on SN type
    if 'ia' in type:
        color_cor = color_cor_Ia
    elif 'slsn' in type:
        color_cor = color_cor_slsn
    else:
        color_cor = color_cor_gen

    # Finds the color correction dictionary key closest to best_rest_filter 
    ccnl =  min(color_cor.keys(), key=lambda x:abs(x-best_rest_filter)) # redundant? or intentional?
    ccn = color_cor[ccnl] ## color correcting filers...


    if redshift > 1.5:
        vega_spec = loadtxt(base_root+'/templates/vega_model.dat')
    else:
        vega_spec = loadtxt(base_root+'/templates/vega_model_mod.dat')

    start_time = time.time()
    if parallel:
        if verbose: print('... running parallel kcor by model SN age on %d processors' %Nproc)
        run_kcor_x= partial(kcor, f1=f1, f2=f2, models_used_dict=models_used_dict, redshift=redshift,
                            vega_spec=vega_spec, AB=False)
        pool = multiprocessing.Pool(processes=Nproc)
        result_list = pool.map(run_kcor_x, rest_age)
        # Convert result_list into a 2D numpy array of shape (N_ages, 2) 
        # where column 0 is the mean K-correction and column 1 is the std at each age.
        obs_kcor=array(result_list)
        pool.close()

    else:
        obs_kcor=[]
        if verbose: print('... running serial kcor iterating over model SN age')
        for age in rest_age:
            mkcor,skcor=kcor(age, f1,f2,models_used_dict,redshift,vega_spec, AB=False)
            if verbose > 1: print(age,mkcor)
            obs_kcor.append([mkcor,skcor])
        obs_kcor=array(obs_kcor)
    if verbose: print('kcor processing time = %2.1f seconds'%(time.time()-start_time))

    
    ### try a low-order smooth over kcor for valid points, then extrapolate
    obs_kcor_raw = obs_kcor.copy()
    if 'ia' in type:
        obs_kcor[:,0][-1]=nanmean(obs_kcor[:,0])## add anchoring at end of kcor curve (mean kcor for Ia's)
    else: 
        obs_kcor[:,0][-1]=0.0## add am anchoring at end of kcor curve (late-phase kcor anchor=0 for CCSNe)
       
    obs_kcor[:,0] = convolve(obs_kcor[:,0], Gaussian1DKernel(lc_smoothing_window), boundary='extend')#'fill', fill_value=nanmean(obs_kcor[:,0]))
        
    ### replace NaNs in kcor with linearly interpolated data, and constant interpolated error
    idx = where(obs_kcor[:,0]==obs_kcor[:,0])
    if len(idx[0]) > 0:
        junk,obs_kcor_temp= u.recast(range(len(obs_kcor)),0.,idx[0],obs_kcor[idx][:,0])
        obs_kcor[:,0]=obs_kcor_temp
        idx2 = where(obs_kcor[:,1]!=obs_kcor[:,1])
        obs_kcor_err_temp=np.interp(idx2[0],idx[0],obs_kcor[idx][:,1])
        obs_kcor[idx2][:,1]=obs_kcor_err_temp
        obs_kcor[:,1][where(obs_kcor[:,1]!=obs_kcor[:,1])]=0. ## remove nan's in errors.
    apl_kcor = obs_kcor[:,0]

    # Plot mean kcor vs phase (smoothed and unsmoothed); plot kcor std/mean vs phase
    if review:
        plot_kcor_diagnostic(rest_age=rest_age, obs_kcor=obs_kcor_raw, obs_kcor_smoothed=obs_kcor[:,0], type=type[0], 
                             best_rest_filter=best_rest_filter,redshift=redshift, diag_dir=diag_dir)

    ### distance modulus and time dilation
    # peak the distance modulus shifted by 19.5 magnitudes; quick reference magnitude for the Ia light curve normalization
    # d is luminosity distance, mu is distance modulus, td is (1+z)
    d, mu, peak = cosmocalc.run(redshift, **cosmo)
    td = (1.+redshift)

    ## control times
    # Initialize four empty lists that will store the light curve magnitude and K-correction 
    # at two previous epochs for each current age step. These represent what the SN looked like 
    # one baseline ago (template) and two baselines ago (prev), which is what a difference imaging 
    # pipeline compares against to detect a new transient.
    template_light_curve=[]
    prev_light_curve=[]
    template_kcor = []
    prev_kcor = []
    # Convert the observer-frame survey baseline to the rest frame by dividing by the time dilation factor td = (1+z)
    rest_base=baseline/td

    # Loop over every age step in the rest-frame age grid
    for i,age in enumerate(rest_age):
        # If subtracting one rest-frame baseline from the current age falls before the start 
        # of the age grid (before -50 days), the SN hadn't appeared yet at the previous epoch. 
        # Sets the previous-epoch magnitude to 999.0 (undetectably faint) and K-correction to 0. 
        # In the difference imaging calculation downstream, this means the entire current flux 
        # counts as a new detection since there was nothing to subtract
        if age - rest_base < min(rest_age):
            template_light_curve.append(999.0)
            template_kcor.append(0)
        
        else:
            # For ages where one baseline back falls within the age grid, find the index 
            # in rest_age closest to age - rest_base
            # Require the age to be within dstep days of the target, then among those candidates 
            # pick the one with the minimum distance
            idx = where((abs(age - rest_base - rest_age)<=dstep) & (abs(age - rest_base - rest_age)==min(abs(age - rest_base - rest_age))))
            # Append the light curve magnitude and K-correction at that previous-epoch age
            template_light_curve.append(observed_frame_lightcurve[idx][:,0][0])
            template_kcor.append(apl_kcor[idx][0])

        # baseline — the time between the current epoch and the previous epoch (observer-frame days)
        # prev — the time between the previous epoch and the epoch before that (observer-frame days)
        #if age - rest_base-prev/td < min(rest_age):
        if prev == 0 or age - rest_base-prev/td < min(rest_age):
            prev_light_curve.append(999.0)
            prev_kcor.append(0)
        else:
            #idx2 = where(abs(age-rest_base-prev/td - rest_age+dstep)<=dstep)
            idx2 = where((abs(age-rest_base-prev/td - rest_age)<=dstep) & 
             (abs(age-rest_base-prev/td - rest_age)==min(abs(age-rest_base-prev/td - rest_age))))

            prev_light_curve.append(observed_frame_lightcurve[idx2][:,0][0])
            prev_kcor.append(apl_kcor[idx2][0])

    template_light_curve=array(template_light_curve)
    prev_light_curve=array(prev_light_curve)

    template_kcor = array(template_kcor)
    prev_kcor = array(prev_kcor)

    
    tot_ctrl=0.0

    if verbose: print('dstep=%.1f, dmstep=%.1f, dastep=%.1f'%(dstep,dmstep,dastep))
    if plot:
        clf()
        ax1=subplot(121)
        ax2=subplot(122)
        ax3=ax1.inset_axes([0.9,0.0,0.08,1.0])
        yminl=[]
        ymaxl=[]

    # Loop on extinction function
    # ext_normalization accumulates the integral of the extinction probability distribution for normalization later
    ext_normalization=0.0
    if extinction:
        dastep = dastep
        # Integrate over host galaxy extinction values from 0 to 10 magnitudes in steps of dastep
        darange = arange(0.,10.0+dastep,dastep)
    else:
        dastep = 1.0
        darange = [0.]

    # Outer loop over extinction values. da is the host galaxy extinction in magnitudes (A_V or similar) 
    # applied uniformly to all filters via the Calzetti law later.
    for da in darange:

        # Loop on luminosity function
        # Inner loop over absolute magnitude offsets from -5 to +5 mag in steps of dmstep. 
        # dm samples the luminosity function — negative values represent intrinsically brighter 
        # SNe than the mean, positive values represent fainter ones. lum_normalization 
        # accumulates the integral of the luminosity function for normalization.
        dmstep=dmstep
        dmrange=arange(-5,5+dmstep,dmstep)
        lum_normalization=0.0
        for dm in dmrange:
            # Convert the current epoch light curve from magnitudes to fluxes
            # The magnitude at each age is:
            # m = apl_kcor + observed_frame_lightcurve[:,0] + mu + dm + da + ccn
            # Where:
            # apl_kcor — K-correction converting rest-frame to observed-frame magnitude
            # observed_frame_lightcurve[:,0] — rest-frame absolute magnitude from templates
            # mu — distance modulus converting absolute to apparent magnitude
            # dm — luminosity function offset
            # da — host galaxy extinction
            # ccn — color correction
            # This gives the apparent flux of the SN as it would appear through the observed JWST filter at each age
            f1 = 10**(-2./5.*(apl_kcor+observed_frame_lightcurve[:,0]+mu+dm+da+ccn))
            # Same conversion but for the previous epoch light curve (one baseline ago)
            # This is what the reference image contains — the SN flux at the previous observation
            f2 = 10**(-2./5.*(template_kcor+template_light_curve+mu+dm+da+ccn))
            # Compute the flux difference between the current epoch and the previous epoch. 
            # This is what a difference imaging pipeline actually detects
            # A positive diff_f means the SN got brighter since the last epoch (rising or near peak), 
            # while negative means it got fainter (declining).
            diff_f = (f1 - f2)
            # Convert the flux difference back to magnitudes, only counting positive fluxes (detections)
            delta_mag = zeros(diff_f.shape)
            tdx = where(diff_f>0)
            delta_mag[tdx]=-2.5*log10(diff_f[tdx])
            delta_mag[where(diff_f<=0)]=99.99
            ## efficiency=det_eff(delta_mag,mc=sens,T=0.96, S=0.38) ## for GOODS
            efficiency=det_eff(delta_mag,mc=sens,T=1.0, S=0.30)

            f3 = 10**(-2./5.*(prev_kcor+prev_light_curve+mu+dm+da+ccn))
            diff_f2 = (f2 - f3)
            delta_mag2 = zeros(diff_f2.shape)
            tdx = where(diff_f2 > 0)
            delta_mag2[tdx]=-2.5*log10(diff_f2[tdx])
            delta_mag2[where(diff_f2<=0)]=99.99
            ## efficiency2=det_eff(delta_mag2,mc=sens, T=0.96, S=0.38) ## for GOODS
            efficiency2=det_eff(delta_mag2,mc=sens, T=1.0, S=0.30)

                
            sig_m = absmags_local[type[0]][1]
            ## Holz & Linder GL LumFunc smoothing
            sig_gl = 0.093*(redshift)
            sig_m = 1*sqrt(sig_m**2+sig_gl**2)

            P_lum= scipy.stats.norm(absmags_local[type[0]][0],sig_m).pdf(absmags_local[type[0]][0]+dm)
            if extinction:
                if 'ia' in type:
                    P_ext = ext_dist_Ia(da, observed_filter, redshift, passskiprow, passwavemult, sndata_root)
                else:
                    P_ext = ext_dist(da,observed_filter,redshift,passskiprow,
                                     passwavemult,sndata_root,obs_extin=obs_extin)#, Rv=8.0)
            else:
                P_ext=1.0

            if plot:
                yminl.append(min(apl_kcor+observed_frame_lightcurve[:,0]+mu+da+dm+ccn)-2.0)
                ymaxl.append(min(apl_kcor+observed_frame_lightcurve[:,0]+mu+da+dm+ccn)+4.5)
                ax1.plot(rest_age,apl_kcor+observed_frame_lightcurve[:,0]+mu+dm+da+ccn,'r-')
                ax1.plot(rest_age,template_kcor+template_light_curve+mu+dm+da+ccn,'k--')
                ax1.plot(rest_age,prev_kcor+prev_light_curve+mu+dm+da+ccn, ls=':', color='0.4')
                ax2.plot(rest_age,efficiency,'k.-')#,label='Type %s, z=%.1f'%(type[0], redshift))
                ax2.plot(rest_age,efficiency2,'r.:')
                ax1.set_ylim(max(ymaxl),min(yminl))
                ax2.set_ylim(0,1.2)
                ax1.grid()

                ax1.set_xlabel('rest age (days)')
                ax2.set_xlabel('rest age (days)')
                ax1.set_title('%s at z=%.1f' %(type[0].upper(), redshift))
                ax2.set_title('%s at z=%.1f' %(type[0].upper(), redshift))
                
            if prev > 0:
                idx = where(efficiency2 < 0.5) #if eff2 > 0.5 assume would have been detected in previous epoch
                tot_ctrl += nansum(efficiency[idx])*P_lum*P_ext*dstep*dmstep*dastep
            else:
                tot_ctrl += nansum(efficiency)*P_lum*P_ext*dstep*dmstep*dastep
            lum_normalization += P_lum*dmstep
        ext_normalization += P_ext*dastep
    if plot:
        ax3.plot(scipy.stats.norm(absmags_local[type[0]][0],sig_m).pdf(absmags_local[type[0]][0]+dmrange),
                 absmags_local[type[0]][0]+dmrange+mu+ccn+apl_kcor[where(rest_age == min(abs(rest_age)))],
                 'k-')
        ax3.set_xticks([])
        ax3.set_ylim(max(ymaxl),min(yminl))
        #≈ßax3.invert_yaxis()
        ax1.axhline(sens, color='blue', ls=':')
        ax1.hlines(y=sens-1.5,  xmin=0, xmax=baseline/td, color='purple', lw=3)
        tight_layout(); savefig('efficiencies.png')

    tot_ctrl=tot_ctrl/(lum_normalization*ext_normalization)
    print('Correcting control time %.4f days by %s relative number' %(tot_ctrl, biascor))
    if biascor == 'fractional':
        ## fractional bias correction-- The relative number of each subtype one would expect in a volume
        ## Using z=0.0 observations from Li et al. 2011, already corrected for malmquist bias
        if not 'ia' in type:
            if 'ia' in vol_frac_local.keys():
                rel_num = 1.*(vol_frac_local[type[0]])#/(sum(list(vol_frac_local.values()))-vol_frac_local['ia'])
            else:
                rel_num = 1.*(vol_frac_local[type[0]])#/sum(list(vol_frac_local.values()))
        else:
            rel_num = 1.0
        print('... Relative number %.1f' %rel_num)
    elif biascor == 'malmquist':
        ## malmquist bias correction-- use this if going with some other measure of relative number
        if not 'ia' in type:
            rel_lum = 10.**((absmags_local[type[0]][0]-(sens-mu+mean(apl_kcor)))/(-2.5))
            rel_num = rel_lum**(-1.5)
        else:
            rel_num = 1.0
    else: ## assume flat
        rel_num =  1.0

    if subtype_combination == 'forward':
        tot_ctrl = tot_ctrl * rel_num      # forward model: T_raw * f_X (weighted contribution)
    else:
        tot_ctrl = tot_ctrl / rel_num      # divide_average (original)

    if verbose:
        print('for %s at redshift=%.1f'%(type[0].upper(), redshift))
        print('and SN Fraction of %.2f'%(rel_num))
        print("Weighted Control Time= %.4f rest frame days" %tot_ctrl)

        
    if plot:
        clf()
        ax = subplot(211)
        ymin=min(apl_kcor+observed_frame_lightcurve[:,0]+mu+ccn)-2.0
        ymax=min(apl_kcor+observed_frame_lightcurve[:,0]+mu+ccn)+4.5
        xmin=(-50*td)
        xmax=(730.5*td)
        ax.plot(rest_age*td,apl_kcor+observed_frame_lightcurve[:,0]+mu+ccn,'r--')
        ax.axhline(sens, color='b', ls=':')
        sig = sqrt(absmags_local[type[0]][1]**2.+obs_kcor[:,1]**2.)
        ax.fill_between(rest_age*td, apl_kcor+observed_frame_lightcurve[:,0]+mu+ccn+sig,
                        apl_kcor+observed_frame_lightcurve[:,0]+mu+ccn-sig,
                        facecolor='red',alpha=0.3,interpolate=True)
                        
        ax.set_ylim(ymax,ymin)
        ax.set_xlim(xmin,xmax)
        ax.set_xlabel('Observed Frame Age (Days)')
        ax.set_ylabel('Observed Magnitude (%.1f nm)' %ofilter_cen)
    
        ax2 = subplot (212)
        ymin=min(observed_frame_lightcurve[:,0])-1.0
        ymax=min(observed_frame_lightcurve[:,0])+3.5
        xmin=(-50)
        xmax=(730.5)
        ax2.plot(rest_age,observed_frame_lightcurve[:,0],'k-')
        ax2.fill_between(rest_age,
                         observed_frame_lightcurve[:,0]+absmags_local[type[0]][1],
                         observed_frame_lightcurve[:,0]-absmags_local[type[0]][1],
                         facecolor='black', alpha=0.3,interpolate=True)
        ax2.set_ylim(ymax,ymin)
        ax2.set_xlim(xmin,xmax)
        ax2.set_xlabel('Rest Frame Age (Days)')
        ax2.set_ylabel('Closest Template Abs Mag (%.1f nm)' %best_rest_filter)
        tight_layout()
        savefig('lightcurves.png')
    return(tot_ctrl/365.25)
    

def plot_kcor_diagnostic(rest_age, obs_kcor, obs_kcor_smoothed, type, best_rest_filter, redshift, diag_dir):
    """
    Plots the mean K-correction and its standard deviation across template models
    as a function of rest-frame age, along with the smoothed K-correction curve

    Parameters
    ----------
    rest_age: numpy.ndarray
        1D array of rest-frame ages in days.
    obs_kcor: numpy.ndarray
        2D array of shape (N_ages, 2) where column 0 is the mean
        K-correction and column 1 is the standard deviation across
        templates at each age. This is the raw unsmoothed K-correction
    obs_kcor_smoothed: numpy.ndarray
        1D array of shape (N_ages,) containing the smoothed K-correction
        after Gaussian convolution and anchoring. This is what actually
        gets used in the control time calculation.
    type: str
        SN subtype 
    best_rest_filter: int
        Central wavelength of the best matching rest-frame filter in nm.
    redshift: float
        Redshift of the current bin, used for the plot title.
    diag_dir: str
        Path to the diagnostic plots directory.
    """
    fig, (ax1, ax2) = subplots(1, 2, figsize=(14, 6))

    # Left panel: raw mean kcor with sigma shading and smoothed curve
    mean_kc = obs_kcor[:,0]
    std_kc  = obs_kcor[:,1]

    # mask NaN values for plotting
    valid = isfinite(mean_kc) & isfinite(std_kc)

    ax1.fill_between(rest_age[valid],
                     mean_kc[valid] - 2*std_kc[valid],
                     mean_kc[valid] + 2*std_kc[valid],
                     alpha=0.15, color='blue', label='2-sigma template spread')
    ax1.fill_between(rest_age[valid],
                     mean_kc[valid] - std_kc[valid],
                     mean_kc[valid] + std_kc[valid],
                     alpha=0.3, color='blue', label='1-sigma template spread')
    ax1.plot(rest_age[valid], mean_kc[valid], 'k-', lw=2, label='Mean K-correction (raw)')
    ax1.plot(rest_age, obs_kcor_smoothed, 'r--', lw=2, label='K-correction (smoothed)')
    ax1.axhline(0., color='gray', ls='--', lw=1, alpha=0.5)
    ax1.axvline(0., color='gray', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('Rest-frame age (days)')
    ax1.set_ylabel('K-correction (mag)')
    #ax1.set_xlim(-50, 200)
    ax1.set_xlim(-50, 730)
    ax1.legend(fontsize=9)
    ax1.set_title('%s K-correction vs age\nz=%.2f, rest filter=%d nm' % (
                  type, redshift, best_rest_filter))

    # Right panel: std as a fraction of mean kcor
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frac_std = abs(std_kc[valid] / mean_kc[valid])

    ax2.plot(rest_age[valid], std_kc[valid], 'r-', lw=2, label='Absolute std (mag)')
    ax2.plot(rest_age[valid], frac_std, 'b--', lw=2, label='Fractional std (|std/mean|)')
    ax2.axhline(0.1, color='gray', ls=':', lw=1, label='0.1 mag reference')
    ax2.axhline(0.5, color='gray', ls='--', lw=1, label='0.5 mag reference')
    ax2.axvline(0., color='gray', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel('Rest-frame age (days)')
    ax2.set_ylabel('K-correction spread')
    ax2.set_xlim(-50, 200)
    ax2.set_ylim(0, min(nanmax(std_kc[valid])*1.5, 5.))
    ax2.legend(fontsize=9)
    ax2.set_title('K-correction template spread\nz=%.2f, rest filter=%d nm' % (
                  redshift, best_rest_filter))

    tight_layout()
    savefig('%s/%s_%dnm_z%.2f_kcor_diagnostic.png' % (
            diag_dir, type, best_rest_filter, redshift))
    clf()


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
        Dictionary of mean peak absolute magnitudes from Richardson 2014,
        keyed by SN subtype.
    """
    fig, ax = subplots(1, 1, figsize=(10, 14))

    # plot all three light curves, column 0 only
    ax.plot(rest_age, lc_smoothed[:,0], 'b-', lw=1.5, alpha=0.7,
            label='Before anchoring')
    ax.plot(rest_age, lc_normalized[:,0], 'g-', lw=1.5, alpha=0.7,
            label='Normalized to peak = 0.0 mag')
    ax.plot(rest_age, lc_anchored[:,0], 'r-', lw=2,
            label='Anchored to Richardson 2014 peak = %.2f mag' % absmags[type][0])

    # reference lines
    ax.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.5, label='0.0 mag reference')
    ax.axhline(absmags[type][0], color='red', ls=':', lw=1, alpha=0.5,
               label='Richardson 2014 peak = %.2f mag' % absmags[type][0])

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
    

def det_eff(delta_mag,mc=25.8, T=1.0, S=0.4):
    result=T/(1+exp((delta_mag-mc)/S))
    return(result)

def det_eff_box(delta_mag,mc=25.8):
    result = zeros(delta_mag.shape)
    result[where(delta_mag <=25.8)]=1.0
    return(result)

def get_central_wavelength(filter_file, skip=0, wavemult=1.):
    filter_data = loadtxt(filter_file,skiprows=skip)
    filter_data[:,0]=filter_data[:,0]*wavemult
    # Create a smooth, evenly sampled grid to interpolate the filter profile onto. Padded on each side
    fit_x = arange(min(filter_data[:,0])-25.,max(filter_data[:,0])+25.,5.)
    # Interpolate the filter transmission profile onto the new uniform wavelength grid fit_x
    (junk,fit_y) = u.recast(fit_x,0.,filter_data[:,0],filter_data[:,1])
    # Compute the effective central wavelength as a flux-weighted mean
    elam = int(sum(fit_y*fit_x)/sum(fit_y)+0.5)
    return(elam)
    
    
def read_lc_model(model,sndata_root):
    """
    Reads a SNANA-format .DAT light curve template file and returns the
    filter central wavelengths, light curve data, and SN subtype.

    Parameters
    ----------
    model: str
        Absolute path to the .DAT template file to read.
    sndata_root: str
        Absolute path to the SNANA data root directory, used to resolve
        $SNDATA_ROOT placeholders in filter paths within the .DAT file.

    Returns
    -------
    filters: numpy.ndarray
        1D array of effective central wavelengths in nm for each filter
        the template has light curve data for. Length equals the number
        of FILTER entries in the .DAT file.
    lcdata: numpy.ndarray
        2D array of light curve data where column 0 is the rest-frame
        age in days and each subsequent column contains magnitudes in
        the corresponding filter from the filters array. Shape is
        (N_epochs, N_filters + 1).
    type: str
        SN subtype label as read from the SNTYPE keyword in the .DAT
        file (e.g. 'IIP', 'IIL', 'Ib', 'Ic').
    """
    f = open (model,'r')
    lines = f.readlines()
    f.close()
    filters=[]
    lcdata = []
    for line in lines:
        if line.startswith('FILTER'):
            filter_path = line.split()[2]
            filter_path = filter_path.replace('$SNDATA_ROOT',sndata_root)
            filter_path = filter_path.replace('SDSS','SDSS/SDSS_web2001')
            elam=get_central_wavelength(filter_path, wavemult=0.1)
            filters.append(elam)
        if line.startswith('EPOCH'):
            # Putting the SN rest-frame age and multi-band magnitudes at that age into a list
            c = list(map(float,line.split()[1:]))
            lcdata.append(c)
        if line.startswith('SNTYPE'):
            type = line.split()[1]
    return(array(filters), array(lcdata), type)
            
        
def match_peak(model,model_path):
    '''
    Look up the peak magnitude offset for a given SN template in the SNANA calibration file. 
    This offset is needed to anchor each template to a standard magnitude system so that 
    light curves from different templates can be meaningfully compared and combined. 
    Magoff is essentially used to de-redshift the observed SN templates (i.e., standardizing them)
    '''
    modelname = os.path.basename(model).replace('.DAT','').lower()
    f = open(model_path+'/SIMGEN_INCLUDE_NON1A.INPUT')
    lines = f.readlines()
    f.close()
    magoff=0.0
    # Iterate through SNANA config file to find input sn model (if it exists in file)
    # If model is in SNANA config file, extract and return its magnitude offset (magoff)
    # If model not in config file, magnitude offset (magoff) returned as 0 
    for line in lines:
        if line.startswith('NON1A:'):
            if modelname == line.split()[-1].replace('(','').replace(')','').lower():
                magoff = float(line.split()[3])
                break
    return(magoff)

def mean_pop(mag_array, mag_threshold=50, review=False, rest_age=None, type=None, diag_dir=None):
    """
    Computes representative population statistics across a set of SN template
    light curves at each rest-frame age step. Uses the median as the
    representative magnitude at each age, which is robust to the highly skewed
    distributions that arise from pre-explosion baseline values in the templates
    at early ages. Values above mag_threshold are excluded before computing
    statistics to avoid contamination from unphysical placeholder values.

    Parameters
    ----------
    mag_array: numpy.ndarray
        2D array of shape (N_models, N_ages) containing template light curve magnitudes for
        the best matching rest-frame filter, resampled onto the uniform rest_age grid. 
    mag_threshold: float, optional
        Maximum magnitude value to include in statistics. Values above this
        threshold are excluded as unphysical placeholders. Default is 50.

    Returns
    -------
    data: numpy.ndarray
        2D array of shape (N_ages, 5) containing population statistics at
        each rest-frame age step. The columns are:
        - Column 0: median magnitude across templates (representative value
          used in the control time calculation)
        - Column 1: 1-sigma standard deviation across templates
        - Column 2: 2-sigma standard deviation across templates
        - Column 3: maximum magnitude (faintest template) at this age
        - Column 4: minimum magnitude (brightest template) at this age
        Age steps where all templates exceed mag_threshold are assigned
        values of [999., 0., 0., 999., 999.] to flag them as undetectable.
    """
    data =[]
    modes = []
    for i in range(len(mag_array[0])):
        mags = mag_array[:, i]

        # Mask out placeholder/unphysical values before computing statistics
        mags_clean = mags[mags < mag_threshold]
        
        if len(mags_clean) == 0:
            # All templates are unphysical at this age -- SN not yet visible
            data.append([999., 0., 0., 999., 999.])
            modes.append(nan)
            continue

        med = median(mags_clean)
        try:
            mode = u.binmode(mags_clean)[0]
        except:
            mode = average(mags_clean)
        modes.append(mode)
        sig = std(mags_clean)
        data.append([med,1.0*sig,2.0*sig,max(mags_clean),min(mags_clean)])

    # Diagnostic plot for mode light curve vs median light curve
    if review:
        fig, ax = subplots(1, 1, figsize=(10, 10))

        data_array = array(data)
        ax.plot(rest_age, array(modes), 'r--', lw=2, label='Mode')
        ax.plot(rest_age, data_array[:,0], 'b:', lw=2, label='Median')
        ax.fill_between(x=rest_age, y1=data_array[:,4], y2=data_array[:,3],
                         alpha=0.2, color='gray', label='Min/Max range')
        ax.fill_between(rest_age, data_array[:,0]-data_array[:,1],
                         data_array[:,0]+data_array[:,1],
                         alpha=0.3, color='blue', label='1-sigma')
        ax.set_xlabel('Rest-frame age (days)')
        ax.set_ylabel('Absolute magnitude')
        ymin = nanmin(data_array[:,0][data_array[:,0] < mag_threshold]) - 1
        ymax = nanmax(data_array[:,0][data_array[:,0] < mag_threshold]) + 1
        ax.set_ylim(ymax, ymin-2)  # reversed order inverts the axis
        ax.set_xlim(-50, 200)
        ax.legend(fontsize=9)
        ax.set_title('%s representative light curve comparison' % type)

        tight_layout()
        savefig('%s/%s_mean_pop_lightcurve_diagnostic.png' % (diag_dir, type))
        clf()

    return(array(data))

def rest_frame_lightcurve(types,model_path,sndata_root,dstep=3,verbose=True):
    """
    Loads and interpolates the non1a SED template light curves onto a uniform
    rest-frame age grid for a given set of SN subtypes. Templates are
    normalized using magnitude offsets from SIMGEN_INCLUDE_NON1A.INPUT so
    that only light curve shape information is retained, decoupled from the
    intrinsic luminosity of each individual template.

    Parameters
    ----------
    types: list of str
    model_path: str
    sndata_root: str
    dstep: float, optional
        Step size in days for the rest-frame age grid. Default is 3.
    verbose: bool, optional

    Returns
    -------
    rest_age: numpy.ndarray
        1D array of rest-frame ages in days from -50 to 730.5 in steps of dstep.
        This is the uniform time axis all light curves are resampled onto.
    mag_dict: dict
        Dictionary mapping filter central wavelengths (int, in nm) to
        lists of light curve magnitude arrays. Each list contains one
        array per template model that passed the type and magoff checks,
        resampled onto rest_age and normalized by the magnitude offset.
        mag_dict[filter] has shape (N_models, N_ages).
    models_used : list of str
        List of template model names (filenames without .DAT extension)
        that were successfully loaded and included in mag_dict.
    """
    models = glob.glob(model_path+'/*.DAT')
    rest_age = arange(-50,730.5,dstep)
    mag_dict={}
    models_used=[]
    for model in models:
        filters,mdata,type=read_lc_model(model,sndata_root)
        magoff = match_peak(model,model_path)
        
        ## models need anchoring...
        # Append a row at the end of mdata with your largest rest_age (e.g., 730.5) and mag=5
        # for each filter. This prevents wild extrapolation beyond last datapoint
        append(mdata, zeros(len(mdata[0]),))
        mdata[-1][0]=rest_age[-1]; mdata[-1,1:]=5.00

        for cnt,filter in enumerate(filters):

            # Only process SNe whose types are in your config file type list
            # Models with no magoff are not likely to be reliable
            if type.lower() in types  and magoff!=0.0: 

                # Interpolate this model's light curve, shifted by magoff, onto the uniform rest_age grid
                # new_y is the light curve resampled at every age step, ready to be compared across models.
                (junk,new_y)=u.recast(rest_age,0.,mdata[:,0],mdata[:,cnt+1]+magoff)
                ## if average(new_y) > 30:
                ##     ## if verbose>1:
                ##     print('Omitting ',model, filter, average(new_y))
                ##     continue
                ## else:
                ##     print('Keeping ', model, filter, average(new_y))
                if os.path.basename(model)[:-4] not in models_used:
                    models_used.append(os.path.basename(model)[:-4])
                try:
                    mag_dict[filter].append(new_y)
                except:
                    mag_dict[filter]=[new_y]
    return(rest_age,mag_dict,models_used)


def rest_frame_Ia_lightcurve(dstep=3, verbose=True):
    models_dir = sndata_root+'/models/mlcs2k2/mlcs2k2.v007/'
    rest_age = arange(-20,730.5,dstep)
    ## rest_age= arange(-20, -7, dstep) ## to limit to pre-peak discoveries
    mag_dict={}
    for model in glob.glob(models_dir+'vectors_?.dat'):
        data = loadtxt(model)
        junk, yy = u.recast(rest_age, 0., data[:,0],data[:,1])
        filter = os.path.basename(model).split('_')[1][0]
        elam = get_central_wavelength(sndata_root+'/filters/Bessell90/Bessell90_K09/Bessell90_'+filter+'.dat', wavemult=0.1)
        mag_dict[elam]=yy
    return(rest_age,mag_dict)
    
    
def rest_frame_slsn_lightcurve(dstep=3, verbose=True, tm=29.94, b14=3.82, pms=1.0):
    ### from a perscription from Inserra et al. 2013
    phase= arange(-30,730.5,dstep) #rest_age
       
    Ek = 1.0e51
    k = 0.1
 
    xt =phase+15 ## this model works from the approx. explosion date, changing it to phase
    idx = where(xt==0)
    if len(idx[0]) > 0.:
        xt[idx]=0.1
    Mej=((tm*3600*24)*(13.7*3e10)**(1/2.)*(1.05)**(-1.)*k**(-1/2.)*Ek**(1/4.))**(4/3.)
    tp = 4.7*b14**(-2.)*pms**2
    delt = 1-exp(-(9*k*Mej**2*xt**(-2.))/(40*pi*Ek))
    lums = []
    for t in xt:
        Lum = 4.9e46*exp(-(t/tm)**2)*delt*quad(lambda x: 2*x*tm**(-2)*exp((x/tm)**2)*b14**2*pms**(-4)*(1+x/tp)**(-2), 0, t)[0]
        lums.append(Lum[0])
    lums=array(lums)
    mags=-2.5*log10(lums)+111.5+absmags['slsn'][0]
    mag_dict={}
    mag_dict[356]=mags
    mag_dict[472]=mags+0.24 #from average colors, Inserra et al. 2018
    return(phase, mag_dict)

def slsn_lc(xt, tm=29.94, b14=3.82, pms=1.0):
    Ek = 1.0e51
    k = 0.1
    xt = float(xt)
    xt-=15
    if xt == 0.: xt=0.1
    Mej=((tm*3600*24)*(13.7*3e10)**(1/2)*(1.05)**(-1)*k**(-1/2)*Ek**(1/4))**(4/3)
    tp = 4.7*b14**(-2)*pms**2
    delt = 1-exp(-(9*k*Mej**2*xt**(-2))/(40*pi*Ek))
    Lum = 4.9e46*exp(-(xt/tm)**2)*delt*quad(lambda x: 2*x*tm**(-2)*exp((x/tm)**2)*b14**2*pms**(-4)*(1+x/tp)**(-2), 0, xt)[0]
    return(Lum)
  
    

def kcor(best_age,f1,f2,models_used_dict,redshift,vega_spec, extrapolated=True, AB=False):
    """
    Computes the K-correction at a single rest-frame age by integrating
    SN SED templates through the observed and rest-frame filters. The
    K-correction converts the rest-frame template magnitude to the
    observed-frame magnitude, accounting for both the SN's spectral
    shape and the zero-point difference between the two filter systems.

    Parameters
    ----------
    best_age: float
        Rest-frame age in days at which to compute the K-correction.
    f1: numpy.ndarray
        2D array of shape (N_wavelengths, 2) containing the observed
        filter transmission curve. Column 0 is wavelength in Angstroms,
        column 1 is transmission. 
    f2: numpy.ndarray
        2D array of shape (N_wavelengths, 2) containing the best matching
        rest-frame filter transmission curve. Column 0 is wavelength in
        Angstroms, column 1 is transmission.
    models_used_dict: dict
        Dictionary mapping model names to their full SED data arrays.
        Each SED array has columns: age (days), wavelength (Angstroms),
        f_lambda flux. 
    redshift: float
        Redshift of the SN
    vega_spec: numpy.ndarray
        2D array containing the Vega reference spectrum. Column 0 is
        wavelength in Angstroms, column 1 is flux. 
    extrapolated: bool, optional
        If True, extends the SED spectrum to cover the full wavelength
        range of both filters by anchoring to zero flux at 1000 and
        60000 Angstroms and interpolating. This handles cases where the
        observed filter (blueshifted) falls outside the native SED
        wavelength range. Default is True.
    AB: bool, optional
        If True, uses the AB magnitude K-correction formula which
        includes a cosmological bandwidth correction term
        -2.5*log10(1+redshift) and uses the AB reference integrals
        (synth_AB, nearest_AB) instead of the Vega reference integrals.
        If False, uses the Vega magnitude K-correction formula.
        Default is False.

    Returns
    -------
    result: tuple of (float, float)
        A tuple of (mean_kcor, std_kcor) where:
        - mean_kcor is the mean K-correction in magnitudes averaged
          across all template models that had valid spectra near best_age.
          This is the value added to the template light curve magnitude
          in the control time calculation.
        - std_kcor is the standard deviation of the K-correction across
          template models, representing the uncertainty from template
          diversity. A large std_kcor indicates the K-correction is
          sensitive to the choice of template.
        Returns (NaN, NaN) if no valid K-corrections could be computed
        at this age, e.g. if no templates have spectra near best_age
        or if all flux integrals are zero or negative.
    """

    import warnings#,exceptions
    
    # Treat all RuntimeWarnings as exceptions that will crash the program (to catch numerical issues)
    warnings.simplefilter("error",RuntimeWarning)

    # RuntimeWarnings are treated as errors globally, so np.nanmean() on an empty array would crash
    # the program. my_nanmean() locally catches that specific RuntimeWarning and returns NaN instead,
    # allowing the K-correction for that model/age to be flagged as invalid rather than crashing
    # the entire parallel calculation.
    def my_nanmean(a):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                x=np.nanmean(a)
            except RuntimeWarning:
                x=np.NaN
        return(x)
        
    # Initialize an empty list to accumulate K-correction values across all SED template models.
    kcor = []

    # Find the indices of the Vega spectrum where the wavelength falls within the wavelength range of the 
    # observed filter f1. This clips the Vega spectrum to only the wavelengths covered by the observed filter
    idx = where((vega_spec[:,0]>=min(f1[:,0]))&(vega_spec[:,0]<=max(f1[:,0])))

    # Interpolate the observed filter transmission curve f1 onto the Vega spectrum wavelength 
    # grid within the filter range. restf1 is the filter transmission evaluated at every Vega 
    # spectrum wavelength point 
    (junk,restf1)=u.recast(vega_spec[idx][:,0],0.,f1[:,0],f1[:,1])

    # Compute the synthetic Vega flux through the observed filter f1
    # This is a discrete approximation of the integral: 
    # synth_vega = integral(lambda * T(lambda) * F_vega(lambda) * dlambda)
    # Where:
    # vega_spec[idx][:,0] — wavelength (lambda)
    # array(restf1) — filter transmission T(lambda) interpolated onto the Vega wavelength grid
    # vega_spec[idx][:,1] — Vega flux F_vega(lambda)
    # my_nanmean(diff(vega_spec[idx][:,0])) — mean wavelength step size dlambda
    # The lambda weighting converts from energy units (f_lambda) to photon-counting units, 
    # which is what a detector actually measures.
    synth_vega = sum(vega_spec[idx][:,0]*array(restf1)*vega_spec[idx][:,1])*my_nanmean(diff(vega_spec[idx][:,0]))
    
    # Computes the AB reference flux through the observed filter f1. The AB magnitude system is defined 
    # such that a source with constant flux density f_nu has the same magnitude in all filters. 
    # The AB zero point integral is: synth_AB = integral(T(lambda)/lambda * dlambda)
    # Where:
    # f1[:,0]**-1 — 1/lambda
    # f1[:,1] — filter transmission T(lambda)
    # my_nanmean(diff(f1[:,0])) — mean wavelength step size dlambda
    # This is used in the AB magnitude K-correction formula when AB=True.
    synth_AB = sum(f1[:,0]**-1*f1[:,1])*my_nanmean(diff(f1[:,0]))

    # Clip the Vega spectrum to the rest-frame filter wavelength range
    idx = where((vega_spec[:,0]>=min(f2[:,0]))&(vega_spec[:,0]<=max(f2[:,0])))

    # Interpolate the rest-frame filter transmission onto the Vega wavelength grid
    (junk,restf2)=u.recast(vega_spec[idx][:,0],0.,f2[:,0],f2[:,1])

    # Synthetic Vega flux through the rest-frame filter
    nearest_vega = sum(vega_spec[idx][:,0]*array(restf2)*vega_spec[idx][:,1])*my_nanmean(diff(vega_spec[idx][:,0]))
    
    # AB reference flux through the rest-frame filter
    nearest_AB = sum(f2[:,0]**-1*f2[:,1])*my_nanmean(diff(f2[:,0]))

    
    ### now sn spectrum
    for model in models_used_dict.keys():
        spec = models_used_dict[model]
        # Find all rows in the SED where the age is within 3 days of best_age
        idx = where(abs(spec[:,0]-best_age)<3.)
        # Skip this model at this age if either:
        # No spectra were found within 3 days of best_age (the model has no data near this age)
        # The total flux is zero (the SN hasn't exploded yet — pre-explosion baseline)
        if (len(idx[0])==0.0) or (sum(spec[idx][:,2]) == 0.0): continue

        if extrapolated:
            ### extrapolated spectrum method
            # Extrapolate the SED templates down up to 60000 Angstrom and down to 1000 Angstrom
            wave_plus = arange(spec[idx][:,1][-1],60000.,10.)
            #wave_minus = arange(1000.,spec[idx][:,1][-1],10.)
            wave_minus = arange(1000.,spec[idx][:,1][0],10.)
            # Create an anchored version of the SED by adding two boundary points: 
            # zero flux at 1000 Å and zero flux at 60000 Å.
            anchored_x = array([1000.]+list(spec[idx][:,1])+[60000.])
            anchored_y = array([0.]+list(spec[idx][:,2])+[0.])
            # Interpolate the anchored SED onto the extended wavelength grids (smooth taper to 0 at extremes)
            j1, counts_plus = u.recast(wave_plus, 0., anchored_x, anchored_y)
            j1, counts_minus = u.recast(wave_minus, 0., anchored_x, anchored_y)
            # Concatenate the below-SED extension, SED, and above-SED extension into single wavelength+flux arrays
            xx = array(list(wave_minus)+list(spec[idx][:,1])+list(wave_plus))
            yy = array(list(counts_minus)+list(spec[idx][:,2])+list(counts_plus))
            # Sort by wavelength to ensure xx (wavelength) is monotonically increasing
            xx, yy = zip(*sorted(zip(xx,yy)))
            xx, yy = array(xx), array(yy)
            
            # Finds the indices of the extended restframe SED array xx that fall within 
            # the wavelength range of the blueshifted observed filter
            idx2 = where((xx >=min(f1[:,0]/(1+redshift)))&(xx<=max(f1[:,0]/(1+redshift))))
            # Interpolate the blueshifted observed filter transmission onto the restframe SED 
            # wavelength grid within the filter range. restf1 is the filter transmission evaluated 
            # at every restframe SED wavelength point in xx[idx2], putting the filter and SED on the same 
            # wavelength grid for integration
            (junk,restf1)=u.recast(xx[idx2],0.,f1[:,0]/(1+redshift),f1[:,1])
            
            # Computes the synthetic SN flux through the blueshifted observed filter:
            # synth_obs = integral(lambda * T(lambda/(1+z)) * F_SN(lambda) * dlambda)
            # Where:
            # xx[idx2] — rest-frame wavelengths (lambda)
            # array(restf1) — blueshifted filter transmission T(lambda/(1+z))
            # yy[idx2] — rest-frame SN SED flux F_SN(lambda)
            # my_nanmean(diff(xx[idx2])) — mean wavelength step dlambda
            # This represents the flux you would observe from this SN through your JWST filter.
            if AB:
                # MAY NEED TO CHANGE BACK LATER
                #synth_obs = sum(xx[idx2]*array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2]))
                synth_obs = sum(array(restf1)/xx[idx2]*yy[idx2])*my_nanmean(diff(xx[idx2]))
            else:
                synth_obs = sum(xx[idx2]*array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2]))

            # Find the indices of the extended SED that fall within the wavelength range 
            # of the rest-frame filter f2 (this time without any blueshifting, since f2 
            # is already in the rest frame)
            idx3 = where((xx >=min(f2[:,0]))&(xx<=max(f2[:,0])))
            # Interpolate the rest-frame filter transmission onto the SED wavelength grid 
            # within the rest-frame filter range
            (junk,restf2)=u.recast(xx[idx3],0.,f2[:,0],f2[:,1])
            # Compute the synthetic SN flux through the rest-frame filter
            # This is what you would observe if you could observe the SN directly through 
            # the rest-frame filter without any redshift — essentially the "idealized" flux that 
            # the templates were calibrated in
            if AB:
                # MAY NEED TO CHANGE THIS BACK LATER
                #nearest_obs = sum(xx[idx3]*array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3]))
                nearest_obs = sum(array(restf2)/xx[idx3]*yy[idx3])*my_nanmean(diff(xx[idx3]))
            else:
                nearest_obs = sum(xx[idx3]*array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3]))

        else:
            ### reduce the computation by only working with wavelengths that are defined in filter throughputs
            ## this would work fine, except at redshifts where the observed filter does not overlap the template spectra
            idx2 = where((spec[idx][:,1]>=min(f1[:,0]/(1+redshift)))&(spec[idx][:,1]<=max(f1[:,0]/(1+redshift))))
            (junk,restf1) = u.recast(spec[idx][idx2][:,1],0.,f1[:,0]/(1+redshift),f1[:,1])
            if AB:
                synth_obs = sum(spec[idx][idx2][:,1]*array(restf1)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))
            else:
                synth_obs = sum(spec[idx][idx2][:,1]*array(restf1)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))
        
            idx2 = where((spec[idx][:,1]>=min(f2[:,0]))&(spec[idx][:,1]<=max(f2[:,0])))
            (junk,restf2) = u.recast(spec[idx][idx2][:,1],0.,f2[:,0],f2[:,1])
            if AB:
                nearest_obs = sum(spec[idx][idx2][:,1]*array(restf2)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))
            else:
                nearest_obs = sum(spec[idx][idx2][:,1]*array(restf2)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))

            ## synth_obs = sum(spec[idx][idx2][:,1]*array(restf1)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))
            ## idx2 = where((spec[idx][:,1]>=min(f2[:,0]))&(spec[idx][:,1]<=max(f2[:,0])))
            ## (junk,restf2) = u.recast(spec[idx][idx2][:,1],0.,f2[:,0],f2[:,1])
            ## nearest_obs = sum(spec[idx][idx2][:,1]*array(restf2)*spec[idx][idx2][:,2])*my_nanmean(diff(spec[idx][idx2][:,1]))

        if AB:
            try:
                kc = -1*(-2.5*log10(1.+redshift)+2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_AB/nearest_AB))
            except:
                # float('Nan')
                kc = float('Nan')
        else:
            try:
                kc = -1*(2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_vega/nearest_vega))
            except:
                kc = float('Nan')

        # Append this model's K-correction to the list. After the loop over all models, 
        # kcor contains one K-correction value per template model.
        kcor.append(kc)

    # If kcor is empty — meaning no models had valid spectra near this age — returns a NaN tuple.
    if not kcor:
        result=(float('Nan'),float('Nan'))
    # If there's only one K-correction value and it's NaN, returns a NaN tuple.
    elif len(kcor)==1 and kcor[0]!=kcor[0]:
        result=(float('Nan'),float('Nan'))
    # For all other cases, computes the mean and standard deviation of the K-corrections across all template models
    # The result is a tuple of (mean_kcor, std_kcor) — the mean is the K-correction value used in the 
    # control time calculation, while the std captures the template-to-template uncertainty in the K-correction. 
    else:
        try:
            result=(my_nanmean(kcor),nanstd(kcor))
        except:
            result=(float('Nan'),float('Nan'))
    return(result)



def ext_dist_Ia(ext,observed_filter,redshift,passskiprow,passwavemult,sndata_root):
    from scipy.optimize import curve_fit
    f1 = sndata_root +'/filters/Bessell90/Bessell90_K09/Bessell90_V.dat'
    w1 = get_central_wavelength(f1, wavemult=0.1)/1e3
    w2 = get_central_wavelength(observed_filter, skip=passskiprow, wavemult=passwavemult)/1e3/(1.0+redshift)
    A_1 = calzetti(array([w1]))
    A_2 = calzetti(array([w2]))


    Jha = loadtxt(base_root+'/templates/Jha_ext.txt')
    Jha[:,0] = Jha[:,0]/A_1*A_2
    p0 = [1.,1.]
    p1,pcov = curve_fit(u.exp_fit,Jha[:,0], Jha[:,1], p0=p0)
    norm = quad(u.exp_fit,0., inf, args=tuple(p1))[0]
    return(u.exp_fit(ext,*p1)/norm)
    

def ext_dist(ext,observed_filter,redshift,passskiprow,passwavemult,sndata_root,Rv=4.05,obs_extin='nominal'):
    from scipy.optimize import curve_fit

    if obs_extin=='nominal': #shallowest
        lambda_v = 0.187
    elif obs_extin=='steep':
        lambda_v= 5.36 #from HP02
        #lambda_v=9.72 #from HBD98
    elif obs_extin=='kelly12':
        lambda_v = 1 #from Kelly12
    elif obs_extin=='arp299':
        lambda_v = 0.025 ## nuclear region of Arp299, see ref. in Bondi et al. 2012
    else:## assuming 'shallow'
        lambda_v =2.27 ## for dahlen 2012.
    f1 = sndata_root + '/filters/Bessell90/Bessell90_K09/Bessell90_V.dat'
    w1 = get_central_wavelength(f1, wavemult=0.1)/1e3
    w2 = get_central_wavelength(observed_filter, skip=passskiprow, wavemult=passwavemult)/1e3/(1.0+redshift)
    A_1 = calzetti(array([w1]),Rv=Rv)
    A_2 = calzetti(array([w2]),Rv=Rv)

    AL = ext*A_2/A_1
    PAL = abs(1/lambda_v)*scipy.stats.expon.pdf(AL,scale=1/lambda_v)
    return(PAL[0])

def ext_dist_ccsn_old(ext,observed_filter,redshift,passskiprow, passwavemult,obs_extin='nominal',observed=False):
    
    if obs_extin=='nominal': observed=True
    from scipy.optimize import curve_fit
    
    if observed:
        f1 = sndata_root + '/filters/Bessell90/Bessell90_K09/Bessell90_V.dat'
    else:
        f1 = sndata_root+ '/filters/Bessell90/Bessell90_K09/Bessell90_B.dat'
    w1 = get_central_wavelength(f1, wavemult=0.1)/1e3
    w2 = get_central_wavelength(observed_filter, skip=passskiprow, wavemult=passwavemult)/1e3/(1.0+redshift)
    A_1 = calzetti(array([w1]))
    A_2 = calzetti(array([w2]))

    if observed:
        if not os.path.isfile(base_root+'/templates/ext_model.pkl'):
            f = open(base_root+'/templates/ext_model.txt','r')
            lines = f.readlines()
            f.close()
            Av=[]
            for line in lines:
                if line.startswith('#'):continue
                Av.append(float(line.split()[3]))
            Av=array(Av)
            pickle.dump(Av,open(base_root+'/templates/ext_model.pkl','wb'))
        else:
            Av = pickle.load(open(base_root+'/templates/ext_model.pkl','rb'), encoding='latin1')
        Av=Av/A_1*A_2
        n,bins=histogram(Av,bins=5)
        p0=[1.,1.]
        pout, pcov = curve_fit(u.exp_fit,bins[:-1]+0.5*average(diff(bins)),n,p0=p0)
        P_ext = abs(pout[1])*scipy.stats.expon(pout[1]).pdf(ext)
        return(P_ext)
    else:
        HBD = loadtxt(base_root+'/templates/HBD_ext.txt')
        HBD[:,0]=HBD[:,0]/A_1*A_2
        xx = arange(0.0,5.0,0.05)
        (junk,yy)=u.recast(xx,0.,HBD[:,0],HBD[:,1])
        yy = array(yy)
        yy[where(yy<0)]=0.0
        yy = yy/(sum(yy)*0.05)
        (junk,P_ext)=u.recast([ext],0.,xx,yy)
        return(P_ext[0])


def calzetti(x,Rv=4.05): # in microns
    y=2.659*(-2.156 + 1.509*(x)**(-1.)-0.198*(x)**(-2.)+0.011*(x)**(-3))+Rv
    ii = where (x>0.63)
    y[ii]=2.659*(-1.857 + 1.040*(x[ii])**(-1.))+Rv
    y[where(y < 0)]=1e-4 ## arbitrary, non-negative
    return(y)

def fline(x,*p):
    m,b = p
    return m*x+b

def fline2(x,*p):
    m,b = p
    return (m*x+b)*(1.0+x)


if __name__=='__main__':

    # types = ['ia']
    types = ['iip']#,'iil','iin','ib','ic']
    #types = ['slsn']
    redshift = 1.0
    baseline = 365
    sens = 29.8
    dstep=5.0 ## in days, probably shouldn't adjust
    dmstep=0.5 ## in magnitude
    dastep=0.5 ## in magnitude
    parallel = True
    Nproc=int(multiprocessing.cpu_count()-2)
    previous = 0.0
    plot = True
    verbose = True
    extinction= True
    if len(types)>1:
        biascor = 'fractional'
    else:
        biascor = 'flat'
    
    if 'ia' in types:
        rate = 5e-5
    elif 'slsn' in types:
        rate = 1e-9
    else:
        rate = 5e-4

    multiplier = 1.0
    all_events = 0
    area = 300.*(1./60.)**2*(pi/180.)**2*(4.0*pi)**(-1)
    dvol = volume.run(redshift+0.2, **cosmo)-volume.run(redshift-0.2, **cosmo)
    
    box_tc = False
    tc_tot=0
    for type in types:
        type=[type]
        if box_tc :
            tc1=run(redshift-0.2,baseline,sens,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction)
            tc2=run(redshift+0.2,baseline,sens,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction, biascor=biascor)
            xx =array([redshift - 0.2, redshift+0.2])
            yy = array([tc1,tc2])
            p0=[1.0,0.0]
            pout = curve_fit(fline,xx,yy,p0=p0)[0]
            tc = quad(fline2,xx[0],xx[1],args=tuple(pout))[0]/diff(xx)
        else:
            tc=run(redshift,baseline,sens,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction, biascor=biascor)#, obs_extin='extra')
            tc_tot+=tc
    print("Total Control Time = %2.4f years" %(tc_tot))
    nevents = tc*dvol*area*rate*multiplier
    print("%2.4f total events" %all_events)
