#!/usr/bin/env python
import os,sys,pdb,scipy,glob
from pylab import *
from scipy.interpolate import *
import numpy
import warnings

def my_colors_old(ii):
    my_colors=['#332288','#88CCEE','#44AA99','#117733','#999933','#DDCC77','#CC6677', '#882255', '#AA4499']
    return(my_colors[ii])


def my_colors(ii):
    myc = ['#FFE1C7',
           '#CCDFEE',
           '#CEE9CE',
           '#E6DCEF',
           '#DDCC77','#CC6677', '#882255', '#AA4499',
           '#332288','#88CCEE','#44AA99','#117733','#999933',
           ]
    return(myc[ii])


def binmode(data,bins=None):
    from scipy import stats
    data=array(data)
    mdx = where(~isnan(data))
    data = data[mdx]
    if bins != None:
        m,mbins = histogram(data, bins=bins)
    else:
        step=1/100.
        splits = arange(0,1+step,step)
        bin_edges=stats.mstats.mquantiles(data,splits)
        bins=sort(list(set(bin_edges)))
        #warnings.simplefilter("error",RuntimeWarning)
        if (max(bins[:-1])-min(bins[1:]))*step!=0.0:
            rebins = arange(min(bins[1:]), max(bins[:-1]),
                            (max(bins[:-1])-min(bins[1:]))*step)
            m,mbins = histogram(data,bins=rebins)
        else:
            m=zeros(1); mbins=zeros(2)
    
    mdx = where(m == max(m))
    mbin =0.5*(mbins[mdx[0][0]+1]+mbins[mdx[0][0]])
    #return (mbin,
    #        array(zip(*vstack([m,mbins[:-1]])[::-1])))
    return (mbin, array(list(zip(*vstack([m, mbins[:-1]])[::-1]))))


def col(data, colindex):
    return [row[colindex] for row in data]

def gauss(x, *p):
    A, mu, sigma = p
    return A*exp(-(x-mu)**2/(2.*sigma**2))

def gauss2(x, y, *p):
    ## where mu == 0
    A, B, D= p
    return  A*exp(-1.0*((0.5*(x/B)**2)+(0.5*(y/D)**2)))

def lognorm(x,*p):
    A, mu, sigma = p
    return A/(x*sigma)*exp(-(log(x)-mu)**2/(2.*sigma**2))

def exp_fit(x,*p):
    A,B = p
    return(A*exp(B*x))

def sigmoid(x,*p):
    x0, T, k, b = p
    b = sqrt(b**2)
    k = sqrt(k**2)
    T = sqrt(T**2)
    y = (T / (1 + exp(-k*(x-x0))))-b
    return y

def sigmoid_sn(x,*p):
    x0, T, k, b = p
    b = sqrt(b**2)
    T = sqrt(T**2)
    y = (T / (1 + exp(-k*(x-x0))))-b
    return y



def quadsum(data):
    qs=[]
    for item in data:
        qs.append(item**2.)
    qs=sum(qs)
    qs=sqrt(qs)
    return qs

def nanquadsum(data):
    qs=[]
    for item in data:
        qs.append(item**2.)
    qs=nansum(qs)
    qs=sqrt(qs)
    return qs


def iterstat(data,err,sigma=4):
    werr=[]
    ndata=[]
    nerr=[]
    for n in range(len(data)):
        if ((data[n] < 50.) & (data[n] > -50.)):
            ndata.append(data[n])
            nerr.append(err[n])
    #print nerr
    #raise

    if len(ndata) == 0:
        return (99.00,99.00)
    
    for value in nerr:
        if (value == 0.):
            mval = 0.
        else:
            mval=1./value
        werr.append(mval)
    
    #m=numpy.median(ndata)
    m=numpy.average(ndata,weights=werr)
    s=numpy.std(ndata)/sqrt(len(ndata))
    #for n in range(len(ndata)):
    #    print ndata[n],nerr[n]
    #print m,s
    
    for n in range(3):
        mdata=[]
        merr=[]
        mwerr=[]
        for i in range(len(ndata)):
            value=ndata[i]
            verr=nerr[i]
            if (((value <= (m+sigma*s))&(value >=(m-sigma*s))) | (((value - m) <1.0e-05) & (s == 0.0))):
                mdata.append(value)
                merr.append(verr)
                if (verr == 0.) :
                    mwerr.append(0.)
                else:
                    mwerr.append(1./verr)

                
        m=numpy.average(mdata,weights=mwerr)
        s=numpy.std(mdata)/sqrt(len(mdata))
        #print m,s,sigma,len(mdata)
    #print 'break'
    
    test=[]
    for value in mwerr:
        if (value == 0.) :
            test.append(0.)
        else:
            test.append((1./value)**2.0)
    ss=numpy.sum(test)/len(test)
    sq=sqrt(ss)    
    return (m,s)


def simple_iterstat(data, sigma_clip=5.):
    import random

    
    data=data.flatten()
    idx = random.sample(range(len(data)),int(len(data)*0.01))
    data=data[idx]
    
    mu_last = median(data)
    sigma_last = std(data)
    mu = mu_last
    sigma=sigma_last
    if sigma==0.:
        return(mu,sigma,0)
    
    for i in range(300):
        if sigma == 0.0:
            break
        
        idx = where (abs(data - mu)/sigma < sigma_clip)
        mu = median(data[idx])
        sigma=std(data[idx])
        if sigma == sigma_last:
            break
        else:
            sigma_last = sigma

    return(mu,sigma,i)



def adjust_spines(ax,spines):
    if sys.version_info >= (3,0):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward',1)) # outward by 10 points
                #spine.set_smart_bounds(True)
            else:
                spine.set_color('none') # don't draw spine
    else:        
        for loc, spine in ax.spines.iteritems():
            if loc in spines:
                spine.set_position(('outward',1)) # outward by 10 points
                #spine.set_smart_bounds(True)
            else:
                spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        #ax.set_yticklabels([])
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike

def recast(d,r,x,y):
    """
    used to get model values at data positions
    data on d,r
    model on x,y
    returns d,rn
    Note: IUS is much better!
    use
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    fn_out = IUS(x,y)
    rn = fn_out(d)
    
    """
    use_new = False
    if ((sys.version_info >= (3,0)) & use_new):
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        xx = array(list(zip(*sorted(zip(x,y))))[0])
        yy = array(list(zip(*sorted(zip(x,y))))[1])
        try:
            fn_out = IUS(xx,yy, k=1)
        except:
            pdb.set_trace()
        rn = fn_out(d)
    else:
        rn=[]
        f = interp1d(x,y)#, bounds_error=False, fill_value=0.0)#[::-1],y[::-1])
        f_x=extrap1d(f)
        for i in range(len(d)):
            try:
                yn=f(d[i])
            except:
                yn=f_x([d[i]])
            #rn.append(float(yn))
            rn.append(float(numpy.squeeze(yn))) # numpy.squeeze() added for compatibility with numpy 2.0+, which no longer allows float() to convert 0-dimensional arrays directly.
    return(d,rn)



def calzetti(xarr,Av=1.0,Rv=4.05):
    xarr = xarr/1.0E4
    #Rv=3.1 # MW
    #Rv=4.05 # Extragal
    #Rv=2.72 # SMC
    k = []
    for x in xarr:
        if x < 0.63:
            k.append(2.659*(-2.156+(1.509/x)-(0.198/x**2)+(0.011/x**3))+Rv)
        else:
            k.append(2.569*(-1.857+(1.040/x))+Rv)
    k = array(k)
    Al = k*Av/Rv
    return (Al)

def convjd(x):
    import string
    try:
        jd=float(x)
    except ValueError:
        return()
    jd=jd+0.5
    Z=int(jd)
    F=jd-Z
    alpha=int((Z-1867216.25)/36524.25)
    A=Z + 1 + alpha - int(alpha/4)

    B = A + 1524
    C = int( (B-122.1)/365.25)
    D = int( 365.25*C )
    E = int( (B-D)/30.6001 )

    dd = B - D - int(30.6001*E) + F

    if E<13.5:
        mm=E-1

    if E>13.5:
        mm=E-13

    if mm>2.5:
        yyyy=C-4716

    if mm<2.5:
        yyyy=C-4715

    months=["January", "February", "March", "April",
            "May", "June", "July", "August", "September",
            "October", "November", "December"]
    daylist=[31,28,31,30,31,30,31,31,30,31,30,31]
    daylist2=[31,29,31,30,31,30,31,31,30,31,30,31]

    h=int((dd-int(dd))*24)
    min=int((((dd-int(dd))*24)-h)*60)
    sec=86400*(dd-int(dd))-h*3600-min*60

    # Now calculate the fractional year. Do we have a leap year?
    if (yyyy%4 != 0):
        days=daylist2
    elif (yyyy%400 == 0):
        days=daylist2
    elif (yyyy%100 == 0):
        days=daylist
    else:
        days=daylist2

    #print x+" = "+months[mm-1]+" %i, %i, " % (dd, yyyy),
    #print string.zfill(h,2)+":"+string.zfill(min,2)+":"+string.zfill(sec,2)+" UTC"
    td = '%s %i %i %s:%s:%i' %(months[mm-1],dd,yyyy,string.zfill(h,2),
                               string.zfill(min,2),int(float(float(string.zfill(sec,2)))))
    td = datetime.datetime.strptime(td, '%B %d %Y %H:%M:%S')
    return(td)

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
    
    (junk,nul)=recast(n,0.0,ul[:,0],ul[:,1])
    (junk,nll)=recast(n,0.0,ll[:,0],ll[:,1])
    he=array(nul)-n
    le=n-array(nll)
    return(he,le)


def gimme_rebinned_data(tz, limit=None, switch='GT',
                        splits=arange(0,1.2,0.2).tolist(),
                        conservative=False, verbose=False):
    from scipy import stats
    import numpy
    ## assumes tz is in array of xx, yy, yl, yu
    if limit:
        p=[]
        tz=tz.tolist()
        for i in range(len(tz)):
            if switch == 'GT':
                if tz[i][0] < limit: p.append(tz[i])
            elif switch == 'LT':
                if tz[i][0] > limit: p.append(tz[i])
        tz=array(p)
            
    if verbose: print("#bl bu rate ql qu qt N")
    bin_edges = stats.mstats.mquantiles(tz[:,0], splits)
    z=[]
    r=[]
    u=[]
    l=[]
    a=[]
    b=[]
    for i in range(len(bin_edges)):
        if i==0: continue
        bz=[]
        br=[]
        wu=[]
        wl=[]
        qs_u=0.
        qs_l=0.
        for j in range(len(tz[:,0])):
            if (tz[:,0][j] >= bin_edges[i-1] and tz[:,0][j] <bin_edges[i]):
                bz.append(tz[:,0][j])
                br.append(tz[:,1][j])
                wu.append(1./tz[:,2][j]**2.)
                wl.append(1./tz[:,3][j]**2.)
                if conservative:
                    qs_u+=tz[:,2][j]**2.
                    qs_l+=tz[:,3][j]**2.
                else:
                    qs_u+=1./(tz[:,2][j]**2.)
                    qs_l+=1./(tz[:,3][j]**2.)
            elif (i == len(bin_edges)-1 and tz[:,0][j]>=bin_edges[i]):
                bz.append(tz[:,0][j])
                br.append(tz[:,1][j])
                wu.append(1./tz[:,2][j]**2.)
                wl.append(1./tz[:,3][j]**2.)
                if conservative:
                    qs_u+=(tz[:,2][j]**2.)
                    qs_l+=(tz[:,3][j]**2.)
                else:
                    qs_u+=1./(tz[:,2][j]**2.)
                    qs_l+=1./(tz[:,3][j]**2.)
        if conservative:
            qs_u=sqrt(qs_u)/sqrt(len(bz))#-1)
            qs_l=sqrt(qs_l)/sqrt(len(bz))#-1)
        else:
            qs_u=sqrt(1./qs_u)
            qs_l=sqrt(1./qs_l)
        mid_z=(bin_edges[i]+bin_edges[i-1])/2.
        z.append(mid_z)
        r.append(average(br,weights=wl))
        qs_t=numpy.std(br)
        u.append(qs_u)
        l.append(qs_l)
        a.append(abs(bin_edges[i-1]-mid_z))
        b.append(abs(bin_edges[i]-mid_z))
        if verbose: print(bin_edges[i-1], bin_edges[i],numpy.average(br,weights=wl), qs_l, qs_u, qs_t, len(br))

    brates = zeros((len(z),6))
    brates[:,0]=array(z)
    brates[:,1]=array(r)
    brates[:,2]=array(u)
    brates[:,3]=array(l)
    brates[:,4]=array(a)
    brates[:,5]=array(b)
    return(brates)


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def allblack(ax):
    lg=ax.legend(loc=1,frameon=False)
    for text in lg.get_texts():
        text.set_color('white')

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

def allblack2(ax, lg):
    for text in lg.get_texts():
        text.set_color('white')

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

def allblack0(ax):

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
