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
import util as u
import diagnostic_plot_util as plot_util
import cosmocalc
import volume
import glob
from pprint import pprint
from scipy.interpolate import interp1d as scipy_interp1d

import multiprocessing
from functools import partial

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

import warnings#,exceptions
warnings.simplefilter("error",RuntimeWarning)

rcParams['figure.figsize']=12,9
rcParams['font.size']=16.0

def run(redshift, baseline, base_root, sndata_root, lightcurve_path, sed_path, diag_dir,
        type, m50=30, T=1.0, S=0.3, dstep=3, dmstep=0.5, dastep=0.5, lc_smoothing_window=3,
        parallel=False, extinction=True, obs_extin='nominal', Nproc=23, prev=45.,
        passband = None, passskiprow=1, passwavemult=1000.,
        plot=False, verbose=False, review=False, biascor='flat',
        cosmology=None,color_corrections=None, absmags=None):

    print('--- absmags diagnostic ---')
    print('type: %s' % type[0])
    print('absmags mean: %.2f' % absmags[type[0]][0])
    print('absmags sigma: %.2f' % absmags[type[0]][1])
    print('--------------------------')

    # Resolve cosmology: [H0, Om, Ol] from config, or {} to use module defaults.
    cosmo = cosmocalc.resolve_cosmology(cosmology)

    # Ensure you have the peak absolute magnitude for your subtype
    if absmags is None:
        raise ValueError(
            "absmags is None -- peak absolute magnitudes must be passed "
            "into control_time.run() via the absmags parameter."
        )
    if type[0] not in absmags:
        raise ValueError(
            "SN type '%s' not found in absmags. "
            "Available types: %s" % (type[0], list(absmags.keys()))
        )

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
    observed_filter = passband # you are forced to specify a passband when calling run(); cannot use None
    ofilter_cen = get_central_wavelength(observed_filter,skip=passskiprow,wavemult=passwavemult)
    if verbose: print('observed filter effective wavelength= %4.1f nm'%ofilter_cen)

    # Rest-frame Lightcurve
    if 'ia' not in type:
        #if verbose: print('getting best rest-frame lightcurve...')

        # Load and interpolate non1a SED templates onto a uniform rest-frame age grid, keyed by filter central wavelength
        rest_age,rflc,models_used = rest_frame_lightcurve(type,lightcurve_path,sed_path,sndata_root,dstep=dstep,verbose=verbose)

        # Plot the full restframe lightcurves for each model in used_models, for each available restframe filter
        if review:
            plot_util.plot_restframe_lightcurves(rflc=rflc, rest_age=rest_age, models_used=models_used, 
                                                 type=type[0],diag_dir=diag_dir)

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
            plot_util.plot_smoothing_diagnostic(observed_frame_lightcurve_unsmoothed=observed_frame_lightcurve_unsmoothed, 
                                                observed_frame_lightcurve_smoothed=observed_frame_lightcurve, 
                                                rest_age=rest_age, type=type[0], diag_dir=diag_dir, dstep=dstep)

        observed_frame_lightcurve_unanchored = observed_frame_lightcurve.copy()
        lc_normalized_to_0 = observed_frame_lightcurve - nanmin(observed_frame_lightcurve[:,0])

        # Anchor composite light curve to your choice of mean peak absolute magnitude by shifting its 
        # brightest point to absmags[type[0]][0]
        observed_frame_lightcurve = observed_frame_lightcurve - nanmin(observed_frame_lightcurve[:,0]) + absmags[type[0]][0]

        if review:
            plot_util.plot_anchoring_diagnostic(observed_frame_lightcurve_unanchored, lc_normalized_to_0, 
                                                observed_frame_lightcurve,rest_age, type[0], diag_dir, absmags)

    # Get the Type Ia rest-frame light curve
    else:
        #if verbose: print('getting best rest-frame lightcurve SNIA ...')
        rest_age, rflc = rest_frame_Ia_lightcurve(dstep=dstep,verbose=verbose)
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)
        observed_frame_lightcurve = zeros((len(array(rflc[best_rest_filter])),5))
        observed_frame_lightcurve[:,0] = array(rflc[best_rest_filter]) - template_peak[type[0]]+absmags[type[0]][0]

    # Get the spectral templates
    model_pkl = 'SEDs_'+'_'.join(type)+'.pkl'
    if not os.path.isfile(model_pkl):
        pkl_file = open(model_pkl,'wb')
        #if verbose: print('... loading model SEDs')
        models_used_dict={}
        total_age_set=[]
        if 'ia' in type:
            models_used = ['Hsiao07']#'Foley07_lowz_uhsiao']
            sed_path = sndata_root +'/snsed'
        if 'slsn' in type:
            models_used = ['slsn_blackbody']

        for model in models_used:
            #print('...... %s' %model)
            if 'ia' not in type:
                try:
                    data = loadtxt(os.path.join(sed_path,model+'.SED'))
                except:
                    print('testing', os.path.join(sed_path,model+'.SED'))
                    pdb.set_trace()
            else:
                data = loadtxt(os.path.join(sed_path,model+'.dat'))

            #Extract the unique ages present in this model's SED data (column 0 is age) 
            # and stores the full SED array in models_used_dict keyed by model name.
            ages = list(set(data[:,0]))
            models_used_dict[model]=data

            # Accumulate all unique ages across all models into total_age_set
            for age in ages:
                if age not in total_age_set:
                    total_age_set.append(age)

        pickle.dump(models_used_dict,pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(model_pkl,'rb')
        #if verbose: print('reading %s saved file' %model_pkl)
        models_used_dict = pickle.load(pkl_file, encoding='latin1')
        pkl_file.close()
        total_age_set=[]
        for model in models_used_dict.keys():
            ages = list(set(models_used_dict[model][:,0]))
            for age in ages:
                if age not in total_age_set:
                    total_age_set.append(age)
    

    # Color correcting from peak M_g to peak M_x, where x is your best-matching SDSS restframe filter
    if color_corrections is None:
        raise ValueError(
            "color_corrections is None -- a color correction file must be "
            "specified in the config via 'color_correction_file'."
        )
    if type[0] not in color_corrections:
        raise ValueError(
            "SN type '%s' not found in color corrections file. "
            "Available types: %s" % (type[0], list(color_corrections.keys()))
        )

    # Get the color correction at peak relative to SDSS g-band 
    # This is the x-g color correction, where x-band is the rest-frame SDSS filter 
    # that most closely matches your rest-frame JWST filter (i.e., the Q-band)
    # This color_correction dictionary is loaded in by rate_calculator from the 
    # file specified in the config file
    color_cor_subtype_dict = color_corrections[type[0]]
    xband_key = min(color_cor_subtype_dict.keys(), key=lambda x: abs(x-best_rest_filter))
    Mg_to_Mx_color = color_cor_subtype_dict[xband_key]

    # Color correcting from M_x to M_Q at every lightcurve phase, 
    # where x is your best-matching SDSS restframe filter and 
    # Q is your observed JWST filter in the restframe (e.g., F150W/(1+z))
    Mx_to_MQ_color = Mx_to_MQ(rest_age=rest_age, models_used_dict=models_used_dict, best_rest_filter=best_rest_filter, 
                              filter_dict=filter_dict, observed_filter=observed_filter, redshift=redshift,
                              passskiprow=passskiprow, passwavemult=passwavemult)
    Mx_to_MQ_color_raw = Mx_to_MQ_color.copy()

    # Interpolate over NaN gaps before smoothing
    # Anchor latest phase color correction to 0 so that late phase color correction linearly tapers to 0
    # No early phase anchor is set, so a constant extrapolation is applied to earlier phases by default
    Mx_to_MQ_color = np.array(Mx_to_MQ_color, dtype=float)
    Mx_to_MQ_color[-1] = 0
    valid_idx = np.where(np.isfinite(Mx_to_MQ_color))[0]
    if len(valid_idx) > 1:
        Mx_to_MQ_color_interp = np.interp(np.arange(len(Mx_to_MQ_color)),valid_idx,Mx_to_MQ_color[valid_idx])
    else:
        raise ValueError("Invalid Color Correction between M_X and M_Q")

    # Smooth the color correction
    Mx_to_MQ_color = convolve(Mx_to_MQ_color_interp, Gaussian1DKernel(lc_smoothing_window),boundary='extend')

    if review:
        plot_util.plot_phase_color_correction(rest_age=rest_age, color_cor_array=Mx_to_MQ_color, 
                                              color_cor_raw=Mx_to_MQ_color_raw, best_rest_filter=best_rest_filter, 
                                              redshift=redshift, type=type[0],observed_filter=observed_filter, 
                                              diag_dir=diag_dir)

    # d is luminosity distance, mu is distance modulus
    d, mu, _ = cosmocalc.run(redshift, **cosmo)

    # Control times
    # Initialize four empty lists that will store the light curve magnitude and Q-x color corrections 
    # at two previous epochs for each current age step. These represent what the SN looked like 
    # one baseline ago (template) and two baselines ago (prev), which is what a difference imaging 
    # pipeline compares against to detect a new transient.
    template_light_curve=[]
    prev_light_curve=[]
    template_Mx_to_MQ = []
    prev_Mx_to_MQ = []
    # Convert the observer-frame survey baseline to the rest frame by dividing by the 
    # time dilation factor (1+z)
    rest_base=baseline/(1.+redshift)

    # Loop over every age step in the rest-frame age grid
    for i,age in enumerate(rest_age):
        # If subtracting one rest-frame baseline from the current age falls before the start 
        # of the age grid (before -50 days), the SN hadn't appeared yet at the previous epoch. 
        # Sets the previous-epoch magnitude to 999.0 (undetectably faint) and Q-x color is 0. 
        # In the difference imaging calculation downstream, this means the entire current flux 
        # counts as a new detection since there was nothing to subtract
        if age - rest_base < min(rest_age):
            template_light_curve.append(999.0)
            template_Mx_to_MQ.append(0)
        
        else:
            # For ages where one baseline back falls within the age grid, find the index 
            # in rest_age closest to age - rest_base
            # Require the age to be within dstep days of the target, then among those candidates 
            # pick the one with the minimum distance
            idx = where((abs(age - rest_base - rest_age)<=dstep) & (abs(age - rest_base - rest_age)==min(abs(age - rest_base - rest_age))))
            # Append the light curve magnitude and Q-x color at that previous-epoch age
            template_light_curve.append(observed_frame_lightcurve[idx][:,0][0])
            template_Mx_to_MQ.append(Mx_to_MQ_color[idx][0])

        # baseline — the time between the current epoch and the previous epoch (observer-frame days)
        # prev — the time between the previous epoch and the epoch before that (observer-frame days)
        #if age - rest_base-prev/(1.+redshift) < min(rest_age):
        if prev == 0 or age - rest_base-prev/(1.+redshift) < min(rest_age):
            prev_light_curve.append(999.0)
            prev_Mx_to_MQ.append(0)
        else:
            #idx2 = where(abs(age-rest_base-prev/(1.+redshift) - rest_age+dstep)<=dstep)
            idx2 = where((abs(age-rest_base-prev/(1.+redshift) - rest_age)<=dstep) & 
             (abs(age-rest_base-prev/(1.+redshift) - rest_age)==min(abs(age-rest_base-prev/(1.+redshift) - rest_age))))

            prev_light_curve.append(observed_frame_lightcurve[idx2][:,0][0])
            prev_Mx_to_MQ.append(Mx_to_MQ_color[idx2][0])

    template_light_curve=array(template_light_curve)
    prev_light_curve=array(prev_light_curve)
    template_Mx_to_MQ = array(template_Mx_to_MQ)
    prev_Mx_to_MQ = array(prev_Mx_to_MQ)

    tot_ctrl=0.0
    if verbose: print('dstep=%.1f, dmstep=%.1f, dastep=%.1f'%(dstep,dmstep,dastep))

    # Retrieve the intrinsic scatter in peak absolute magnitude for this SN subtype 
    sig_m = absmags[type[0]][1]
    ## Holz & Linder GL LumFunc smoothing
    # Compute the gravitational lensing magnification scatter from Holz & Linder (2005)
    # At high redshift, random gravitational lensing by intervening matter along the line 
    # of sight introduces additional scatter in the apparent brightness of SNe — some lines 
    # of sight pass through more mass and get magnified, others get demagnified
    sig_gl = 0.093*(redshift)
    # Combine the intrinsic luminosity scatter and gravitational lensing scatter in 
    # quadrature to get the total magnitude scatter
    sig_m = 1*sqrt(sig_m**2+sig_gl**2)

    # Get the Calzetti Law ratio between AL and Av, where AL is the Calzetti extinction in the 
    # restframe filter closest to your blueshifted observed frame filter at your redshift
    Av_over_AQ_calzetti_ratio = get_Av_over_AQ_calzetti_ratio(observed_filter=observed_filter,redshift=redshift,
                                                              passskiprow=passskiprow,passwavemult=passwavemult,
                                                              sndata_root=sndata_root,Rv=4.05)

    # Loop on extinction function
    # ext_normalization accumulates the integral of the extinction probability distribution for normalization later
    ext_normalization=0.0
    if extinction:
        dastep = dastep
        # Integrate over host galaxy extinction values from 0 to 10 magnitudes in steps of dastep
        # da values are AQ values 
        darange = arange(0.,10.0+dastep,dastep)
    else:
        dastep = 1.0
        darange = [0.]

    # Outer loop over extinction values. da is the host galaxy extinction in magnitudes (AQ) 
    # applied uniformly to all filters via the Calzetti law later.
    for da in darange:

        # Convert AQ (da) to Av, where AQ is your extinction in the rest-frame Q-band filter
        Av = da * Av_over_AQ_calzetti_ratio 

        # Compute a physically motivated extinction probability weight P_ext for the current da value
        # P_ext is the probability that a SN of this type has host galaxy extinction equal to da magnitudes
        # obs_extin controls the shape of the host galaxy extinction probability distribution
        # It weights the control time contribution from SNe with this particular extinction — highly 
        # extincted SNe (da large) get low weight if such high extinctions are rare for this SN type, 
        # while moderately extincted SNe get higher weight.
        if extinction:
            if 'ia' in type:
                P_ext = ext_dist_Ia(da, observed_filter, redshift, passskiprow, passwavemult, sndata_root)
            else:
                # Get the host extinction probability
                P_ext = prob_Av(Av=Av, obs_extin=obs_extin)
        else:
            P_ext=1.0
        ext_normalization += P_ext*dastep # this should technically be the Av step (not AQ step), but cancels later

        # Loop on luminosity function
        # Inner loop over absolute magnitude offsets in steps of dmstep. 
        # dm samples the luminosity function — negative values represent intrinsically brighter 
        # SNe than the mean, positive values represent fainter ones. lum_normalization 
        # accumulates the integral of the luminosity function for normalization.
        dmstep=dmstep
        dmrange=arange(round(-3*sig_m-dmstep,1), round(3*sig_m+dmstep,1), dmstep)
        lum_normalization=0.0
        for dm in dmrange:

            if review and da==1.0 and abs(dm) < dmstep/2.:
                plot_util.plot_lightcurve_stages(rest_age=rest_age, observed_frame_lightcurve=observed_frame_lightcurve,
                                                 Mg_to_Mx_color=Mg_to_Mx_color, Mx_to_MQ_color=Mx_to_MQ_color,
                                                 mu=mu, da=da, dm=dm, sens=m50, type=type[0], redshift=redshift,
                                                 best_rest_filter=best_rest_filter, observed_filter=observed_filter, 
                                                 diag_dir=diag_dir)

            # Go from nearest SDSS restframe lightcurve anchored by g-band peak to nearest SDSS restframe
            # light curve ("X"-band) anchored by "X"-band peak (applied with Mg_to_Mx color (x-g))
            # dm just steps through your peak SN luminosity function
            Mx = observed_frame_lightcurve[:,0] + dm + Mg_to_Mx_color

            # Go from nearest restframe SDSS light curve (X-band) to Q-band light curve,
            # where Q-band is your observed JWST filter in the restframe
            MQ = Mx + Mx_to_MQ_color

            # Apply your host galaxy extinction, which is computed in the restframe Q-band
            MQ_ext = MQ + da

            # Use the distance modulus and k-correction to convert the restframe Q-band light curve
            # to your observer-frame JWST light curve. Because Q-band directly relates to your observed
            # JWST filter via (JWST_filter/(1+z)), there is no need for a color term in the k-correction.
            # Only the redshift term is required for the k-correction
            m_jwst = MQ_ext + mu - 2.5*np.log10(1+redshift)

            # Convert this magnitude to a flux
            flux1 = 10**(-2./5.*m_jwst)
            
            # Same conversion but for the previous epoch light curve (one baseline ago)
            # This is what the reference image contains — the SN flux at the previous observation
            flux2 = 10**(-2./5.*(template_light_curve+dm+Mg_to_Mx_color+template_Mx_to_MQ+da+mu-2.5*np.log10(1+redshift)))
            
            # Compute the flux difference between the current epoch and the previous epoch. 
            # This is what a difference imaging pipeline actually detects
            # A positive diff_f means the SN got brighter since the last epoch (rising or near peak), 
            # while negative means it got fainter (declining).
            diff_flux = (flux1 - flux2)
            # Convert the flux difference back to magnitudes, only counting positive fluxes (detections)
            delta_mag = zeros(diff_flux.shape)
            tdx = where(diff_flux>0)
            delta_mag[tdx]=-2.5*log10(diff_flux[tdx])
            delta_mag[where(diff_flux<=0)]=99.99

            # Obtain the detection efficiency for delta_mag using the best-fit m50 and S values
            # that were previously obtained by fitting your input detection efficiency curve
            efficiency=det_eff(delta_mag, mc=m50, T=T, S=S)

            # Compute detection efficiency based on SN brightness relative to two baselines ago
            # This avoids double-counting sources in rolling surveys
            flux3 = 10**(-2./5.*(prev_light_curve+dm+Mg_to_Mx_color+prev_Mx_to_MQ+da+mu-2.5*np.log10(1+redshift)))
            diff_flux2 = (flux2 - flux3)
            delta_mag2 = zeros(diff_flux2.shape)
            tdx = where(diff_flux2 > 0)
            delta_mag2[tdx]=-2.5*log10(diff_flux2[tdx])
            delta_mag2[where(diff_flux2<=0)]=99.99
            efficiency2=det_eff(delta_mag2,mc=m50, T=T, S=S)

            # Evaluate the luminosity function probability at this dm offset. This creates a Gaussian 
            # distribution centered on the mean peak absolute magnitude absmags[type[0]][0] with 
            # width sig_m, then evaluates its probability density at absmags[type[0]][0]+dm. 
            # So P_lum is the probability that a SN of this type has a peak absolute magnitude offset 
            # of dm from the mean — it's the weight applied to the control time contribution from SNe 
            # with this particular luminosity. 
            P_lum= scipy.stats.norm(absmags[type[0]][0],sig_m).pdf(absmags[type[0]][0]+dm)
                
            if prev > 0:
                idx = where(efficiency2 < 0.5) #if eff2 > 0.5 assume would have been detected in previous epoch
                tot_ctrl += nansum(efficiency[idx])*P_lum*P_ext*dstep*dmstep*dastep
            else:
                tot_ctrl += nansum(efficiency)*P_lum*P_ext*dstep*dmstep*dastep

            lum_normalization += P_lum*dmstep

    tot_ctrl=tot_ctrl/(lum_normalization*ext_normalization)
    return tot_ctrl/365.25

# -----------------------------------------------------------------------------------------------------

def det_eff(delta_mag,mc=25.8, T=1.0, S=0.4):
    result=T/(1+exp((delta_mag-mc)/S))
    return(result)

# -----------------------------------------------------------------------------------------------------

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
    
# -----------------------------------------------------------------------------------------------------

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
            #filter_path = filter_path.replace('SDSS','SDSS/SDSS_web2001')
            elam=get_central_wavelength(filter_path, wavemult=0.1)
            filters.append(elam)
        if line.startswith('EPOCH'):
            # Putting the SN rest-frame age and multi-band magnitudes at that age into a list
            c = list(map(float,line.split()[1:]))
            lcdata.append(c)
        if line.startswith('SNTYPE'):
            type = line.split()[1]
    return(array(filters), array(lcdata), type)
            
# -----------------------------------------------------------------------------------------------------
       
def match_peak(model,sed_path):
    '''
    Look up the peak magnitude offset for a given SN template in the SNANA calibration file. 
    This offset is needed to anchor each template to a standard magnitude system so that 
    light curves from different templates can be meaningfully compared and combined. 
    Magoff is essentially used to de-redshift the observed SN templates (i.e., standardizing them)
    '''
    modelname = os.path.basename(model).replace('.DAT','').lower()
    f = open(sed_path+'/SIMGEN_INCLUDE_NON1A.INPUT')
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

# -----------------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------------

def rest_frame_lightcurve(types,lightcurve_path,sed_path,sndata_root,dstep=3,verbose=True):
    """
    Loads and interpolates the non1a SED template light curves onto a uniform
    rest-frame age grid for a given set of SN subtypes. Templates are
    normalized using magnitude offsets from SIMGEN_INCLUDE_NON1A.INPUT so
    that only light curve shape information is retained, decoupled from the
    intrinsic luminosity of each individual template.

    Parameters
    ----------
    types: list of str
    lightcurve_path: str
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
    models = glob.glob(lightcurve_path+'/*.DAT')
    rest_age = arange(-50,730.5,dstep)
    mag_dict={}
    models_used=[]
    for model in models:
        filters,mdata,type=read_lc_model(model,sndata_root)
        magoff = match_peak(model,sed_path)
        
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

# -----------------------------------------------------------------------------------------------------

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
  
# -----------------------------------------------------------------------------------------------------

def synthetic_mag_AB_local(wave_sed, flux_sed, wave_filt, trans_filt):
    """
    Computes synthetic AB magnitude from f_lambda SED through a filter.
    Returns NaN if integration fails.
    """
    f_interp    = scipy_interp1d(wave_filt, trans_filt,
                                 bounds_error=False, fill_value=0.0)
    trans       = np.clip(f_interp(wave_sed), 0., None)
    dlambda     = np.gradient(wave_sed)
    c_light     = 2.998e18  # AA/s

    numerator   = np.sum(flux_sed * trans * dlambda)
    denominator = np.sum(trans * c_light / wave_sed**2 * dlambda)

    if numerator <= 0. or denominator <= 0.:
        return np.nan
    return -2.5 * np.log10(numerator / denominator) - 48.6
 
# -----------------------------------------------------------------------------------------------------

def Mx_to_MQ(rest_age, models_used_dict, best_rest_filter, filter_dict, observed_filter, redshift,
             passskiprow, passwavemult):
    """
    Computes a phase-dependent color correction between the best matching
    rest-frame SDSS filter (x) and the JWST observed filter blueshifted to
    the rest frame (Q) at each rest-frame age step. For SN subtypes with
    multiple templates, returns the median color correction across templates
    at each age step.

    The color correction at each age is defined as:
        color_cor(age) = M_Q(age) - M_X(age)
    where:
        M_X = synthetic AB magnitude through the best matching SDSS filter
        M_Q = synthetic AB magnitude through the JWST filter blueshifted
              to the rest frame (i.e. T_Q(lambda) = T_JWST(lambda*(1+z)))

    Parameters
    ----------
    rest_age : numpy.ndarray
        1D array of rest-frame ages in days on the uniform age grid.
    models_used_dict : dict
        Dictionary mapping model names to their full SED data arrays.
        Each SED array has columns: age (days), wavelength (Angstroms),
        f_lambda flux.
    best_rest_filter : int
        Central wavelength in nm of the best matching rest-frame SDSS filter (x)
    filter_dict : dict
        Dictionary mapping filter central wavelengths (nm) to filter
        file paths, containing the SDSS filter transmission files.
    observed_filter : str
        Path to the observed JWST filter transmission file.
    redshift : float
        Redshift of the SN, used to blueshift the JWST filter to the
        rest frame.
    passskiprow : int
        Number of header rows to skip when reading the JWST filter file.
    passwavemult : float
        Wavelength multiplier for the JWST filter file (e.g. 1000 for
        microns to nm, then x10 for nm to Angstroms).

    Returns
    -------
    color_cor_array : numpy.ndarray
        1D array of shape (N_ages,) containing the median color correction
        M_Q - M_X at each rest-frame age step. NaN where no valid
        spectra exist near that age.
        This should be applied such as: color=M_Q-M_x --> M_Q=M_x+color
    """

    # Load the best matching rest-frame SDSS filter (x) 
    sdss_filter_path = filter_dict[best_rest_filter]
    f_x = loadtxt(sdss_filter_path)
    # f_X[:,0] in Angstroms, f_X[:,1] is transmission
    wave_x  = f_x[:, 0]
    trans_x = f_x[:, 1]

    # Load the JWST observed filter and blueshift to rest frame (Q) 
    f_JWST = loadtxt(observed_filter, skiprows=passskiprow)
    # convert JWST wavelengths to Angstroms
    wave_JWST_obs  = f_JWST[:, 0] * passwavemult * 10.
    trans_JWST     = f_JWST[:, 1]
    # blueshift to rest frame: lambda_rest = lambda_obs / (1+z)
    wave_JWST_rest = wave_JWST_obs / (1. + redshift)

    #print('  Phase-dependent color correction:')
    #print('    SDSS filter X: %d nm (%.0f-%.0f AA)' % (
    #      best_rest_filter, wave_X.min(), wave_X.max()))
    #print('    JWST filter Q (rest frame): %.0f-%.0f AA' % (
    #      wave_JWST_rest.min(), wave_JWST_rest.max()))

    # Loop over each age step 
    color_cor_array = np.full(len(rest_age), np.nan)

    for i, age in enumerate(rest_age):

        # Collect color corrections from all models at this age
        colors_at_age = []

        for model_name, spec in models_used_dict.items():

            # Find rows (restframe SEDs) within 3 days of this age 
            # If none exist or if the flux is 0 (pre-explosion), skip this age (nan in color array at this age)
            idx = np.where(np.abs(spec[:, 0] - age) < 3.)[0]
            if len(idx) == 0:
                continue
            if np.sum(spec[idx, 2]) == 0.:
                continue

            wave = spec[idx, 1]
            flux = spec[idx, 2]

            # Sort by wavelength and remove unphysical values
            sort_idx   = np.argsort(wave)
            wave, flux = wave[sort_idx], flux[sort_idx]
            valid      = wave > 100.
            wave, flux = wave[valid], flux[valid]

            if len(wave) < 2 or np.sum(flux) <= 0.:
                continue

            # Extrapolate restframe SED to cover both filter ranges
            # Anchor to zero at 1000 AA and 60000 AA 
            wave_plus  = np.arange(wave[-1], 60000., 10.)
            wave_minus = np.arange(1000., wave[0], 10.)
            anchored_x = np.array([1000.] + list(wave) + [60000.])
            anchored_y = np.array([0.]    + list(flux) + [0.])

            # Interpolate the restframe SED out to the two anchors (have it taper to 0)
            f_ext = scipy_interp1d(anchored_x, anchored_y,bounds_error=False, fill_value=0.0)
            xx = np.concatenate([wave_minus, wave, wave_plus])
            yy = f_ext(xx)

            # Sort extended restframe SED
            sort_idx = np.argsort(xx)
            xx, yy   = xx[sort_idx], yy[sort_idx]

            # Compute synthetic AB magnitude of restframe SN SED through SDSS filter X
            M_x = synthetic_mag_AB_local(xx, yy, wave_x, trans_x)

            # Compute synthetic AB magnitude of restframe SN SED through JWST filter Q (rest frame)
            M_Q = synthetic_mag_AB_local(xx, yy, wave_JWST_rest, trans_JWST)

            if np.isfinite(M_x) and np.isfinite(M_Q):
                # color = M_Q - M_x
                # M_Q = M_x + color (this is how color should be applied)
                colors_at_age.append(M_Q - M_x)

        # Median color across all models at this age
        if len(colors_at_age) > 0:
            valid_colors = [c for c in colors_at_age if np.isfinite(c)]
            if len(valid_colors) > 0:
                color_cor_array[i] = np.nanmedian(valid_colors)

    #print('    Color correction range: %.3f to %.3f mag' % (
    #      np.nanmin(color_cor_array),
    #      np.nanmax(color_cor_array)))
    #print('    Color correction at peak (age~0): %.3f mag' % (
    #      color_cor_array[np.argmin(np.abs(rest_age))]))

    return color_cor_array

# -----------------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------------

def get_Av_over_AQ_calzetti_ratio(observed_filter,redshift,passskiprow,passwavemult,sndata_root,Rv=4.05):
    V_trans = sndata_root + '/filters/Bessell90/Bessell90_K09/Bessell90_V.dat'
    wave_v = get_central_wavelength(V_trans, wavemult=0.1)/1e3
    wave_obs = get_central_wavelength(observed_filter, skip=passskiprow, wavemult=passwavemult)/1e3
    wave_rest = wave_obs / (1.0+redshift) # Q-band (observed JWST filter in the restframe)
    Av_calzetti = calzetti(array([wave_v]),Rv=Rv)
    AQ_calzetti = calzetti(array([wave_rest]),Rv=Rv)

    return Av_calzetti[0] / AQ_calzetti[0]

# -----------------------------------------------------------------------------------------------------

def prob_Av(Av, obs_extin='nominal'):
    from scipy.optimize import curve_fit

    if obs_extin=='nominal': #shallowest
        lambda_v = 0.187
    elif obs_extin=='steep':
        lambda_v= 5.36 #from HP02
        #lambda_v=9.72 #from HBD98
    elif obs_extin =='shallow':
        lambda_v = 2.27 # Dahlen 2012
    elif obs_extin == 'kelly':
        lambda_v = 1 # from Kelly 2012
    elif obs_extin == 'arp299':
        lambda_v = 0.025 ## nuclear region of Arp299, see ref. in Bondi et al. 2012
    else:
        raise ValueError("Invalid extinction distribution provided")

    PAv = scipy.stats.expon.pdf(Av,scale=1/lambda_v)
    return PAv

# -----------------------------------------------------------------------------------------------------

def calzetti(x,Rv=4.05): # in microns
    y=2.659*(-2.156 + 1.509*(x)**(-1.)-0.198*(x)**(-2.)+0.011*(x)**(-3))+Rv
    ii = where (x>0.63)
    y[ii]=2.659*(-1.857 + 1.040*(x[ii])**(-1.))+Rv
    y[where(y < 0)]=1e-4 ## arbitrary, non-negative
    return(y)

# -----------------------------------------------------------------------------------------------------

'''
# Code previously used to apply the k-correction in the main run() loop
    # K-correction
    if redshift > 1.5:
        vega_spec = loadtxt(base_root+'/templates/vega_model.dat')
    else:
        vega_spec = loadtxt(base_root+'/templates/vega_model_mod.dat')

    start_time = time.time()
    if parallel:
        #if verbose: print('... running parallel kcor by model SN age on %d processors' %Nproc)
        run_kcor_x= partial(kcor, f1=f1, f2=f2, models_used_dict=models_used_dict, redshift=redshift, 
                            vega_spec=vega_spec, AB=True)
        pool = multiprocessing.Pool(processes=Nproc)
        result_list = pool.map(run_kcor_x, rest_age)
        # Convert result_list into a 2D numpy array of shape (N_ages, 2) 
        # where column 0 is the mean K-correction and column 1 is the std at each age.
        obs_kcor=array(result_list)
        pool.close()

    else:
        obs_kcor=[]
        #if verbose: print('... running serial kcor iterating over model SN age')
        for age in rest_age:
            mkcor,skcor=kcor(age, f1,f2,models_used_dict,redshift,vega_spec, AB=True)
            if verbose > 1: print(age,mkcor)
            obs_kcor.append([mkcor,skcor])
        obs_kcor=array(obs_kcor)
    #if verbose: print('kcor processing time = %2.1f seconds'%(time.time()-start_time))

    
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
'''

'''
def fline(x,*p):
    m,b = p
    return m*x+b

def fline2(x,*p):
    m,b = p
    return (m*x+b)*(1.0+x)

def det_eff_box(delta_mag,mc=25.8):
    result = zeros(delta_mag.shape)
    result[where(delta_mag <=25.8)]=1.0
    return(result)


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


def ext_dist(ext,observed_filter,redshift,passskiprow,passwavemult,sndata_root,Rv=4.05,obs_extin='nominal'):
    from scipy.optimize import curve_fit

    if obs_extin=='nominal': #shallowest
        lambda_v = 0.187
    elif obs_extin=='steep':
        lambda_v= 5.36 #from HP02
        #lambda_v=9.72 #from HBD98
    else:## assuming 'shallow'
        ## lambda_v = 1 #from Kelly12
        ## lambda_v = 0.025 ## nuclear region of Arp299, see ref. in Bondi et al. 2012
        lambda_v =2.27 ## for dahlen 2012.
    f1 = sndata_root + '/filters/Bessell90/Bessell90_K09/Bessell90_V.dat'
    w1 = get_central_wavelength(f1, wavemult=0.1)/1e3
    w2 = get_central_wavelength(observed_filter, skip=passskiprow, wavemult=passwavemult)/1e3/(1.0+redshift)
    A_1 = calzetti(array([w1]),Rv=Rv)
    A_2 = calzetti(array([w2]),Rv=Rv)

    AL = ext*A_2/A_1
    PAL = abs(1/lambda_v)*scipy.stats.expon.pdf(AL,scale=1/lambda_v)
    return(PAL[0])


def kcor(best_age,f1,f2,models_used_dict,redshift,vega_spec, extrapolated=True, AB=True):
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
                synth_obs = sum(xx[idx2]*array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2])) # original
                #synth_obs = sum(array(restf1)/xx[idx2]*yy[idx2])*my_nanmean(diff(xx[idx2])) # same as synth_AB
                #synth_obs = sum(array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2]))
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
                nearest_obs = sum(xx[idx3]*array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3])) #original
                #nearest_obs = sum(array(restf2)/xx[idx3]*yy[idx3])*my_nanmean(diff(xx[idx3])) # same as nearest_AB
                #nearest_obs = sum(array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3]))
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
                #kc = -1*(-2.5*log10(1.+redshift)+2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_AB/nearest_AB)) # has sign error
                #kc = -1*(2.5*log10(1.+redshift)+2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_AB/nearest_AB)) # same as below but written differently
                kc = 2.5*np.log10((1./(1.+redshift)) * (nearest_obs * synth_AB) / (nearest_AB * synth_obs))
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
'''
'''
if __name__=='__main__':

    # types = ['ia']
    types = ['iip']#,'iil','iin','ib','ic']
    #types = ['slsn']
    redshift = 1.0
    baseline = 365
    m50 = 29.8
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
            tc1=run(redshift-0.2,baseline,m50,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction)
            tc2=run(redshift+0.2,baseline,m50,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction, biascor=biascor)
            xx =array([redshift - 0.2, redshift+0.2])
            yy = array([tc1,tc2])
            p0=[1.0,0.0]
            pout = curve_fit(fline,xx,yy,p0=p0)[0]
            tc = quad(fline2,xx[0],xx[1],args=tuple(pout))[0]/diff(xx)
        else:
            tc=run(redshift,baseline,m50,type=type,dstep=dstep,dmstep=dmstep,dastep=dastep,verbose=verbose,plot=plot,parallel=parallel,Nproc=Nproc, prev=previous, extinction=extinction, biascor=biascor)#, obs_extin='extra')
            tc_tot+=tc
    print("Total Control Time = %2.4f years" %(tc_tot))
    nevents = tc*dvol*area*rate*multiplier
    print("%2.4f total events" %all_events)
'''
