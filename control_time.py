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

def load_vol_frac(name=None, path=None):
    """Load a named subtype-fraction set from vol_fractions.json and
    normalize CC-subtype values to sum to 1 ('ia','slsn' preserved as-is).
    Keys starting with '_' (metadata) are stripped."""
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

vol_frac = load_vol_frac()  # module-level default (li_2011)

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

absmags_li_2011 = {
    'iip': [-15.66, 1.23, 0.16],
    'iin': [-16.86, 1.61, 0.59],
    'iil': [-17.44, 0.64, 0.22],
    'ib' : [-17.01, 0.41, 0.17],
    'ic' : [-16.04, 1.28, 0.31],
    'ibc': [-16.04, 1.28, 0.31],
    }

absmags_richardson_2014 = {
    'iip': [-16.80, 0.97, 0.37],
    #'iin': [-16.86, 1.61, 0.59],
    'iin': [-18.62, 1.48, 0.32],
    'iil': [-17.98, 0.90, 0.34],
    'ib' : [-17.54, 0.94, 0.33],
    'ic' : [-16.67, 1.04, 0.40],
    'ibc': [-16.67, 1.04, 0.40],
    'ia' : [-19.26, 0.51, 0.20],
    'slsn': [-21.7, 0.4,0.0], ## from Quimby+2013, by way of Gal-Yam 2018
    ## 'slsn': [-30, 2.5,0.0], ## from Whalen et al. 2013
    }
absmags_dahlen_2012 = {
    'iip': [-16.67, 1.12],
    'iin': [-18.82, 0.92],
    'iil': [-17.23, 0.38],
    'ib' : [-19.38, 0.46],
    'ic' : [-17.07, 0.49],
    }
    

absmags=absmags_richardson_2014


#absmag_new = {}
#for key in absmags.keys(): absmag_new[key]=[absmags[key][0]-absmags[key][2],absmags[key][1],absmags[key][2]]
#absmags=absmag_new

color_cor_Ia={
    360:1.8,
    442:0.15,
    551:0.0,
    663:-0.61,
    806:-0.56
    }

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


def run(redshift, baseline, sens, base_root, sndata_root, model_path, run_name,
        type, dstep=3, dmstep=0.5, dastep=0.5,
        parallel=False, extinction=True, obs_extin=True, Nproc=23, prev=45.,
        passband = None, passskiprow=1, passwavemult=1000.,
        plot=False, verbose=False, review=False, biascor='flat',
        subtype_combination='divide_average', vol_frac_set=None, cosmology=None):

    # Resolve the subtype-fraction dict: per-call override, else module default.
    vol_frac_local = load_vol_frac(vol_frac_set) if vol_frac_set else vol_frac
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
        observed_frame_lightcurve[:,0] = array(rflc[best_rest_filter]) - template_peak[type[0]]+absmags[type[0]][0]
    elif 'ia' not in type:
        if verbose: print('getting best rest-frame lightcurve...')

        # Load and interpolate non1a SED templates onto a uniform rest-frame age grid, keyed by filter central wavelength
        rest_age,rflc,models_used = rest_frame_lightcurve(type,model_path,sndata_root,dstep=dstep,verbose=verbose)

        # Find the SDSS filter that most closely matches the rest-frame wavelength of your observed filter
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)

        # Note: despite the name, this is still in the rest frame at this point.
        # The conversion to observed frame (K-correction + distance modulus) happens
        # inside the extinction/luminosity function loop below.
        observed_frame_lightcurve=mean_pop(mag_array=array(rflc[best_rest_filter]), rest_age=rest_age)#-template_peak[type[0]]+absmags[type[0]][0]
        #print("Observed-Frame Light Curve")
        #print(observed_frame_lightcurve)
        #print("")

        observed_frame_lightcurve[:,0]= convolve(observed_frame_lightcurve[:,0], Gaussian1DKernel(dstep), boundary='extend') #smoothing out composite
        observed_frame_lightcurve = observed_frame_lightcurve -template_peak[type[0]]+absmags[type[0]][0]
    else:
        if verbose: print('getting best rest-frame lightcurve SNIA ...')
        rest_age, rflc = rest_frame_Ia_lightcurve(dstep=dstep,verbose=verbose)
        best_rest_filter = min(rflc.keys(), key=lambda x:abs(x-(ofilter_cen/(1+redshift))))
        if verbose: print('best rest frame filter match wavelength= %4.1f nm'%best_rest_filter)
        observed_frame_lightcurve = zeros((len(array(rflc[best_rest_filter])),5))
        observed_frame_lightcurve[:,0] = array(rflc[best_rest_filter]) - template_peak[type[0]]+absmags[type[0]][0]
        
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
            ages = list(set(data[:,0]))
            models_used_dict[model]=data
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

    ## if review:
    ##     for k,v in rflc.items():
    ##         clf()
    ##         ax = subplot(111)
    ##         v=array(v)
    ##         for ii in range(len(v)):
    ##             ax.plot(rest_age, v[ii], label='%s'%models_used[ii])
    ##         ax.set_ylim(-10,-20)
    ##         ax.set_title('Filter = %d' %k)
    ##         ax.legend()
    ##         show()
    ##     pdb.set_trace()

    f1 = loadtxt(observed_filter,skiprows=passskiprow)
    f1[:,0]=f1[:,0]*passwavemult*10.
    
    f2 = loadtxt(filter_dict[best_rest_filter])
    if 'ia' in type:
        color_cor = color_cor_Ia
    elif 'slsn' in type:
        color_cor = color_cor_slsn
    else:
        color_cor = color_cor_gen
    ccnl =  min(color_cor.keys(), key=lambda x:abs(x-best_rest_filter))
    ccn = color_cor[ccnl] ## color correcting filers...

    if redshift > 1.5:
        vega_spec = loadtxt(base_root+'/templates/vega_model.dat')
    else:
        vega_spec = loadtxt(base_root+'/templates/vega_model_mod.dat')

    start_time = time.time()
    if parallel:
        if verbose: print('... running parallel kcor by model SN age on %d processors' %Nproc)
        run_kcor_x= partial(kcor, f1=f1, f2=f2, models_used_dict=models_used_dict, redshift=redshift, vega_spec=vega_spec)
        pool = multiprocessing.Pool(processes=Nproc)
        result_list = pool.map(run_kcor_x, rest_age)
        obs_kcor=array(result_list)
        pool.close()
    else:
        obs_kcor=[]
        if verbose: print('... running serial kcor iterating over model SN age')
        for age in rest_age:
            mkcor,skcor=kcor(age, f1,f2,models_used_dict,redshift,vega_spec)
            if verbose > 1: print(age,mkcor)
            obs_kcor.append([mkcor,skcor])
        obs_kcor=array(obs_kcor)
    if verbose: print('kcor processing time = %2.1f seconds'%(time.time()-start_time))

    ### try a low-order smooth over kcor for valid points, then extrapolate
    if 'ia' in type:
        obs_kcor[:,0][-1]=nanmean(obs_kcor[:,0])## add anchoring at end of kcor curve
    else: 
        obs_kcor[:,0][-1]=0.0## add am anchoring at end of kcor curve
       
    obs_kcor[:,0] = convolve(obs_kcor[:,0], Gaussian1DKernel(dstep), boundary='extend')#'fill', fill_value=nanmean(obs_kcor[:,0]))
    ## if review:    
    ##     clf() ; ax=subplot(111);  ax.plot(rest_age, obs_kcor[:,0], 'k-'); show()
    ##     pdb.set_trace()
        
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

    if review:
        clf() ; ax=subplot(111);  ax.plot(rest_age, obs_kcor[:,0], 'k-'); savefig('kcorrecton.png')

    ### distance modulus and time dilation
    d, mu, peak = cosmocalc.run(redshift, **cosmo)
    td = (1.+redshift)

    ## control times
    template_light_curve=[]
    prev_light_curve=[]
    template_kcor = []
    prev_kcor = []
    rest_base=baseline/td
    for i,age in enumerate(rest_age):
        if age - rest_base < min(rest_age):
            template_light_curve.append(999.0)
            template_kcor.append(0)
        else:
            idx = where((abs(age - rest_base - rest_age)<=dstep) & (abs(age - rest_base - rest_age)==min(abs(age - rest_base - rest_age))))
            template_light_curve.append(observed_frame_lightcurve[idx][:,0][0])
            template_kcor.append(apl_kcor[idx][0])
        if age - rest_base-prev/td < min(rest_age):
            prev_light_curve.append(999.0)
            prev_kcor.append(0)
        else:
            idx2 = where(abs(age-rest_base-prev/td - rest_age+dstep)<=dstep)
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
    ## loop on extinction function
    ext_normalization=0.0
    if extinction:
        dastep = dastep
        darange = arange(0.,10.0+dastep,dastep)
    else:
        dastep = 1.0
        darange = [0.]
    for da in darange:
    ## loop on luminosity function
        dmstep=dmstep
        dmrange=arange(-5,5+dmstep,dmstep)
        lum_normalization=0.0
        for dm in dmrange:
            f1 = 10**(-2./5.*(apl_kcor+observed_frame_lightcurve[:,0]+mu+dm+da+ccn))
            f2 = 10**(-2./5.*(template_kcor+template_light_curve+mu+dm+da+ccn))
            diff_f = (f1 - f2)
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

                
            sig_m = absmags[type[0]][1]
            ## Holz & Linder GL LumFunc smoothing
            sig_gl = 0.093*(redshift)
            sig_m = 1*sqrt(sig_m**2+sig_gl**2)

            P_lum= scipy.stats.norm(absmags[type[0]][0],sig_m).pdf(absmags[type[0]][0]+dm)
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
        ax3.plot(scipy.stats.norm(absmags[type[0]][0],sig_m).pdf(absmags[type[0]][0]+dmrange),
                 absmags[type[0]][0]+dmrange+mu+ccn+apl_kcor[where(rest_age == min(abs(rest_age)))],
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
            rel_lum = 10.**((absmags[type[0]][0]-(sens-mu+mean(apl_kcor)))/(-2.5))
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
        sig = sqrt(absmags[type[0]][1]**2.+obs_kcor[:,1]**2.)
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
                         observed_frame_lightcurve[:,0]+absmags[type[0]][1],
                         observed_frame_lightcurve[:,0]-absmags[type[0]][1],
                         facecolor='black', alpha=0.3,interpolate=True)
        ax2.set_ylim(ymax,ymin)
        ax2.set_xlim(xmin,xmax)
        ax2.set_xlabel('Rest Frame Age (Days)')
        ax2.set_ylabel('Closest Template Abs Mag (%.1f nm)' %best_rest_filter)
        tight_layout()
        savefig('lightcurves.png')
    return(tot_ctrl/365.25)


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

def mean_pop(mag_array, mag_threshold=50,rest_age=None):
    data =[]
    #print('N templates:', len(mag_array))
    medians = []
    modes = []
    for i in range(len(mag_array[0])):
        mags = mag_array[:, i]

        # Mask out placeholder/unphysical values before computing statistics
        mags_clean = mags[mags < mag_threshold]
        
        if len(mags_clean) == 0:
            # All templates are unphysical at this age -- SN not yet visible
            data.append([999., 0., 0., 999., 999.])
            modes.append(nan)
            medians.append(nan)
            continue

        #try:
        #    avg = u.binmode(mag_array[:,i])[0]
        #except:
        #    avg = average(mag_array[:,i])
        med = median(mags_clean)
        try:
            mode = u.binmode(mags_clean)[0]
        except:
            mode = average(mags_clean)
        modes.append(mode)
        medians.append(med)
        sig = std(mags_clean)
        data.append([med,1.0*sig,2.0*sig,max(mags_clean),min(mags_clean)])

    # Diagnostic plot for mode light curve vs median light curve
    if rest_age is not None:
        fig, ax = subplots(1, 1, figsize=(10, 10))

        data_array = array(data)
        ax.plot(rest_age, array(modes), 'r--', lw=2, label='Mode')
        ax.plot(rest_age, array(medians), 'b:', lw=2, label='Median')
        ax.fill_between(rest_age, data_array[:,3], data_array[:,4],
                         alpha=0.2, color='gray', label='Min/Max range')
        ax.fill_between(rest_age, data_array[:,0]-data_array[:,1],
                         data_array[:,0]+data_array[:,1],
                         alpha=0.3, color='blue', label='1-sigma')
        ax.set_xlabel('Rest-frame age (days)')
        ax.set_ylabel('Absolute magnitude')
        ymin = nanmin(array(medians)[array(medians) < mag_threshold]) - 1
        ymax = nanmax(array(medians)[array(medians) < mag_threshold]) + 1
        ax.set_ylim(ymax, ymin-2)  # reversed order inverts the axis
        ax.set_xlim(-50, 200)
        ax.legend(fontsize=9)
        ax.set_title('Representative light curve comparison')

        tight_layout()
        savefig('diagnostic_plots/mean_pop_lightcurve_diagnostic.png')
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
    import warnings#,exceptions
    warnings.simplefilter("error",RuntimeWarning)
    def my_nanmean(a):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                x=np.nanmean(a)
            except RuntimeWarning:
                x=np.NaN
        return(x)
        
    kcor = []

    idx = where((vega_spec[:,0]>=min(f1[:,0]))&(vega_spec[:,0]<=max(f1[:,0])))
    (junk,restf1)=u.recast(vega_spec[idx][:,0],0.,f1[:,0],f1[:,1])
    synth_vega = sum(vega_spec[idx][:,0]*array(restf1)*vega_spec[idx][:,1])*my_nanmean(diff(vega_spec[idx][:,0]))
    synth_AB = sum(f1[:,0]**-1*f1[:,1])*my_nanmean(diff(f1[:,0]))
    idx = where((vega_spec[:,0]>=min(f2[:,0]))&(vega_spec[:,0]<=max(f2[:,0])))
    (junk,restf2)=u.recast(vega_spec[idx][:,0],0.,f2[:,0],f2[:,1])
    nearest_vega = sum(vega_spec[idx][:,0]*array(restf2)*vega_spec[idx][:,1])*my_nanmean(diff(vega_spec[idx][:,0]))
    nearest_AB = sum(f2[:,0]**-1*f2[:,1])*my_nanmean(diff(f2[:,0]))

    
    ### now sn spectrum
    for model in models_used_dict.keys():
        spec = models_used_dict[model]
        idx = where(abs(spec[:,0]-best_age)<3.)
        if (len(idx[0])==0.0) or (sum(spec[idx][:,2]) == 0.0): continue

        if extrapolated:
            ### extrapolated spectrum method
            wave_plus = arange(spec[idx][:,1][-1],60000.,10.)
            wave_minus = arange(1000.,spec[idx][:,1][-1],10.)
            anchored_x = array([1000.]+list(spec[idx][:,1])+[60000.])
            anchored_y = array([0.]+list(spec[idx][:,2])+[0.])
            j1, counts_plus = u.recast(wave_plus, 0., anchored_x, anchored_y)
            j1, counts_minus = u.recast(wave_minus, 0., anchored_x, anchored_y)
            xx = array(list(wave_minus)+list(spec[idx][:,1])+list(wave_plus))
            yy = array(list(counts_minus)+list(spec[idx][:,2])+list(counts_plus))
            xx, yy = zip(*sorted(zip(xx,yy)))
            xx, yy = array(xx), array(yy)
            
            idx2 = where((xx >=min(f1[:,0]/(1+redshift)))&(xx<=max(f1[:,0]/(1+redshift))))
            (junk,restf1)=u.recast(xx[idx2],0.,f1[:,0]/(1+redshift),f1[:,1])
            if AB:
                synth_obs = sum(xx[idx2]*array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2]))
            else:
                synth_obs = sum(xx[idx2]*array(restf1)*yy[idx2])*my_nanmean(diff(xx[idx2]))

            idx3 = where((xx >=min(f2[:,0]))&(xx<=max(f2[:,0])))
            (junk,restf2)=u.recast(xx[idx3],0.,f2[:,0],f2[:,1])
            if AB:
                nearest_obs = sum(xx[idx3]*array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3]))
            else:
                nearest_obs = sum(xx[idx3]*array(restf2)*yy[idx3])*my_nanmean(diff(xx[idx3]))

            ## idx2 = where((xx >=min(f2[:,0]))&(xx<=max(f2[:,0])))
            ## (junk,restf2)=u.recast(xx[idx2],0.,f2[:,0],f2[:,1])
            ## nearest_obs = sum(xx[idx2]*array(restf2)*yy[idx2])*my_nanmean(diff(xx[idx2]))

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

        try:
            kc = -1*(2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_vega/nearest_vega))
        except:
            kc = float('Nan')
        if AB:
            try:
                kc = -1*(-2.5*log10(1.+redshift)+2.5*log10(synth_obs/nearest_obs)-2.5*log10(synth_AB/nearest_AB))
            except:
                float('Nan')
        kcor.append(kc)
    if not kcor:
        result=(float('Nan'),float('Nan'))
    elif len(kcor)==1 and kcor[0]!=kcor[0]:
        result=(float('Nan'),float('Nan'))
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
