#!/usr/bin/env python
"""
A cosmology calculator
Adapted from Ned Wright's COSMOCALC
Currently, just determines luminosity distance parameters

Syntax : cosmocalc.py [options]

"""
import os,sys
from scipy import *
from scipy.integrate import quad
from math import sqrt, log10, pi
co = 299792.458 #speed of light km/s


def main():
    import getopt
    qm = 0.27 # Omega matter
    ql = 0.73 # Omega lambda
    ho = 71 # in km/s/Mpc

    try:
        opt,arg = getopt.getopt(
            sys.argv[1:],"v,h",
            longopts=["help","ql=","qm=","ho="])
    except getopt.GetoptError:
        print("Error: missing or incorrect argumets")
        print(__doc__)
        sys.exit(1)
    for o,a in opt:
        if o in ["-h","--help"]:
            print(__doc__)
            return (0)
        elif o == "--qm":
            qm = float(a)
        elif o == "--ql":
            ql = float(a)
        elif o == "--ho":
            ho = float(a)

    redshift = float(arg[0])
    integral = quad(func,0.0,redshift, args=(qm, ql))[0]
    (d,mu,peak)=luminosity_distance(redshift,qm,ql,ho, integral)
    print(redshift,d,mu,peak)
    return
    #return(z,d,mu,peak)

def run(redshift, qm=0.27, ql=0.73, ho=71):
    integral = quad(func,0.0,redshift, args=(qm, ql))[0]
    (d,mu,peak)=luminosity_distance(redshift,qm,ql,ho, integral)
    return(d,mu,peak)

def func(z,qm, ql):
    ## this is the function that describes the integral
    ## in the cosmological luminosity density fomula
    out = (sqrt((1+z)**2*(1+qm*z)-z*(2+z)*ql))**(-1.)
    return (out)

def H (z, ho, qm, ql):
    ## Calculates the change in hubble value assuming 
    ## w'=constant & w~=-1
    w=-0.78
    h = sqrt((ho**2)*(qm*(1+z)**3+ql*(1+z)**(3*(1+w))))
    return (h)

def luminosity_distance(z, qm, ql, h, integral):
    Q = qm + ql
    if (Q < 1):
        qk = 1 - Q
        d = (1+z)*(1/(sqrt(abs(qk))))*co/h*sinh(sqrt(abs(qk))*integral)
    elif (Q > 1):
        qk = Q - 1
        d = (1+z)*(1/(sqrt(abs(qk))))*co/h*sin(sqrt(abs(qk))*integral)
    else: #Q=1
        d=(1+z)*(co/h*integral)
    mu = 5*log10(d)+25
    peak = mu - 19.5;
    return(d,mu,peak)
        

def volume(z, qm=0.27, ql=0.73, ho=71):
    (dl,mu,peak)=run(z, qm=qm, ql=ql, ho=ho)
    dm=dl/(1+z)
    qk=1.-qm-ql
    dh=3.0e5/ho
    if (qk > 0):
        vc=(4*pi*dh**3/(2*qk))*(dm/dh*sqrt(1+qk*(dm/dh)**2)-1/(sqrt(abs(qk)))*math.asinh(sqrt(abs(qk))*dm/dh))
    elif (qk < 0):
        vc=(4*pi*dh**3/(2*qk))*(dm/dh*sqrt(1+qk*(dm/dh)**2)-1/(sqrt(abs(qk)))*math.asin(sqrt(abs(qk))*dm/dh))
    else: #(qk == 0):
        vc=4*pi/3*dm**3
    return (vc)
    

if __name__ == '__main__':

    main()
