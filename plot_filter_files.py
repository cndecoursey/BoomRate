#!/usr/bin/env python
import os,sys,pdb,scipy,glob
from pylab import *
#from strolger_util import util as u
import util as u

def get_central_wavelength(filter_file, skip=0, wavemult=1.):
    filter_data = loadtxt(filter_file,skiprows=skip)
    filter_data[:,0]=filter_data[:,0]*wavemult
    fit_x = arange(min(filter_data[:,0])-25.,max(filter_data[:,0])+25.,5.)
    (junk,fit_y) = u.recast(fit_x,0.,filter_data[:,0],filter_data[:,1])
    elam = int(sum(fit_y*fit_x)/sum(fit_y)+0.5)
    return(elam)

if __name__=='__main__':
    
    passband_files = glob.glob('JWST_filters/F???W.txt')

    ax1 = subplot(211)
    ax2 = subplot(212)
    for ii,file in enumerate(sort(passband_files)):
        elam = get_central_wavelength(file, skip=1, wavemult=1000.)
        data = loadtxt(file,skiprows=1)
        dfs = abs(data[:,0]*1000-elam)
        idx = where(dfs == dfs.min())
        if (ii%2==0):
            ax1.plot(data[:,0]*1000., data[:,1], color=u.my_colors(ii), label='%s'%(file.split('.')[0]))
            ax1.annotate('%3d'%int(elam),xy=(data[idx][0][0]*1000,data[idx][0][1]), xycoords='data',
                         rotation=45, color=u.my_colors(ii))
        else:
            ax2.plot(data[:,0]*1000., data[:,1], color=u.my_colors(ii), label='%s'%(file.split('.')[0]))
            ax2.annotate('%3d'%int(elam),xy=(data[idx][0][0]*1000,data[idx][0][1]), xycoords='data',
                         rotation=45, color=u.my_colors(ii))

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Throughput')
    ax1.set_ylabel('Throughput')
    ax1.legend(loc=1,frameon=False)
    ax2.legend(loc=1,frameon=False)

    savefig('JWST_filters/figure_jwst_filers.png')
    
