#************************************************************
# PythonAnnular for Python 2.7                              *
# created by David Gabriel                                  *
# Based on Annular by D. Wittman                            *
#                                                           *
#************************************************************


# IMPORTANTE NOTICE
#---------------------------------------
# Current build lacks filter option, and
# only a constant PSF may be used. Requires:
# argparse,textwrap,os,sys,time,re, numpy,
# matplotlib,scipy, inspect, and random.
# Much slower than original Annular, as it
# loads the entire catalog and then reads it.

#USAGE:
# In the terminal type: python PythonAnnular.py [optional arguments] catalog_name x_center y_center


import argparse
import textwrap
import os,sys,time
import re
import numpy as np
import random
import csv
from scipy.optimize import curve_fit
from AnnularModule import *

def read_catalog(catalog,verbose=False,determine_type=False):
    
    '''
    This funtion reads both Fiat and Source Extractor catalogs and produces
    a dictionary as an output.
    '''
    
    t0 = time.clock()
    try:
        V  = open(catalog,'r')
    except IOError:
        sys.stderr.write('File not found')
        exit()
    l  = V.readlines()
    k  = 0
    while '# ' in l[k]:
        k+=1
    if l[0].split()[1]=='fiat': # this is a Fiat catalog
        cat_type = ['Fiat',0]
        k2       = 0
        k3       = 1
        while l[k2].split()[1]!='ttype1':  #EDITED: 'TTYPE1' >> 'ttype1'
            k2+=1
    if l[0].split()[1]!='fiat': # this is a Source Extractor catalog
        cat_type = ['Source Extractor',1]
        k2       = 0
        k3       = 0
    if determine_type:
        return [[],cat_type[1]]
    CATALOG    = np.loadtxt(catalog,skiprows=k)
    dictionary = {}
    for i in range(k2,k):
        if len(CATALOG.T)!=k-k2:
            sys.stderr.write('Catalog is not in the correct format')
            exit()
        dictionary[str(l[i].split()[2+k3])]=CATALOG[:,int(re.findall(r'\d+', l[i].split()[1])[0])-1]
    if verbose:
        sys.stderr.write('Catalog type: '+str(cat_type[0])+'\n')
        tf=time.clock()
        sys.stderr.write('Loading time: '+str(tf-t0)+' s \n')
    V.close()
    return [dictionary,cat_type[1]]


def initial(cat_type=0):
    parser = argparse.ArgumentParser(
    prog='PythonAnnular.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    PythonAnnular
    -------------
    Based on Annular. Make an annular mass profile.
    Reads both Fiat and Source Extractor catalogs.
    Default = 'X Y ixx iyy ixy' for Fiat and
    'X_IMAGE Y_IMAGE X2_IMAGE Y2_IMAGE XY_IMAGE'
    for Source Extractor.
    '''))
    parser.add_argument('catalog', metavar='catalog', type=str, nargs=1,
                        help='Fiat or Source Extractor catalog')
    parser.add_argument('xcenter', metavar='xcenter', type=float, nargs=1,
                        help='center of the x coordinate')
    parser.add_argument('ycenter', metavar='ycenter', type=float, nargs=1,
                        help='center of the y coordinate')
    parser.add_argument('-n', metavar='nbins',action='store', default=15,
                        help='number of bins. Default=15')
    parser.add_argument('-c', metavar='columns',action='store' , default=['X Y ixx iyy ixy','X_IMAGE Y_IMAGE X2_IMAGE Y2_IMAGE XY_IMAGE'][cat_type],
                        help="columns to use. Default='X Y ixx iyy ixy'")
    parser.add_argument('-s', metavar='startrad',action='store', default=0,
                        help='starting radius. Default=0')
    parser.add_argument('-e', metavar='endrad',action='store', default=3000,
                        help='ending radius. Default=3000')
    parser.add_argument('-m', metavar='minell',action='store', default=0,
                        help='minimum ellipticity. Default=0.0')
    parser.add_argument('-x', metavar='maxell',action='store', default=1.0,
                        help='maximum ellipticity. Default=1.0')
    parser.add_argument('-p', metavar='psfsize',action='store', default=0.0,
                        help='size of the PSF. Use only when ncols= 6 or 7. Default=0.0')
    parser.add_argument('-v', action='store_true',
                        help='verbose mode. Default = False')
    parser.add_argument('-d', action='store_true',
                        help='dump positions, corrections and weights. Use only when ncols= 6 or 7.')
    parser.add_argument('-D', action='store_true',
                        help='dump positions, e1 and e2. Do not use with -d.')
    parser.add_argument('-shuffle', action='store_true',
                        help='shuffle the positions of objects.')
    parser.add_argument('-plot', action='store_true',
                        help='plot shear1_unwtd and shear2_unwtd . If ncols= 6 or 7, shear1_wtd and shear2_wtd.')
    args      = parser.parse_args()
    return args

def get_true_M200(catalog):
    '''
    Only works for the simulated catalog with M200 value in it.
    Get true value of M200 by reading the line starting with '# M200 ='
    '''
    try:
        V  = open(catalog,'r')
    except IOError:
        sys.stderr.write('File not found')
        exit()
    for line in V:
        if '# M200 = ' in line:
            return float(line[9:])
    V.close()
    sys.stderr.write('M200 is not given in the catalog')
    exit()


class Array():
    def __init__(self):
        self.e1 , self.e2                                   = 0,0
        self.e1sq , self.e2sq                               = 0,0
        self.e1_wtd , self.e2_wtd                           = 0,0
        self.e                                              = 0
        self.wt , self.r1 , self.r2 , self.var1 , self.var2 = 0,0,0,0,0
        self.sigma1 , self.sigma1_sq , self.sigma2          = 0,0,0
        self.rad                                            = 0
        self.n                                              = 0


def mainloop(file_name):

    args_0         = initial()  
    CATALOG        = read_catalog(file_name,False,True)[0]      #-------|
    args           = initial(read_catalog(file_name,False)[1])  
    xc             = args.xcenter[0]
    yc             = args.ycenter[0]
    nbin           = int(args.n)
    rmin           = float(args.s)
    rmax           = float(args.e)
    minell         = float(args.m)
    maxell         = float(args.x)
    psfsize        = float(args.p)
    verbose        = args.v
    dflag          = args.d
    Dflag          = args.D
    pl             = args.plot
    shuffle        = args.shuffle
    if verbose:
        if len((args.c).split()) ==4:
            sys.stderr.write('No seeing correction. \n')
            sys.stderr.write('Using ellipticity components. \n')
        elif len((args.c).split()) ==5:
            sys.stderr.write('No seeing correction. \n')
            sys.stderr.write('Using moments. \n')
        elif len((args.c).split()) ==6:
            sys.stderr.write('Using seeing correction. \n')
            sys.stderr.write('Using ellipticity components. \n')
        elif len((args.c).split()) ==7:
            sys.stderr.write('Using seeing correction. \n')
            sys.stderr.write('Using moments. \n')
        else:
            sys.stderr.write('Incorrect number of columns, exiting')
            exit()
        sys.stderr.write('Reading catalog...\n')
    CATALOG        = read_catalog(file_name,verbose)[0]
    binsize        = (rmax-rmin)/nbin
    Rmax           = 0.8
    sigma_e        = 0.0
    mean_rho_4th   = 0.0
    SIGMA_SHAPE_SQ = 0.10
    bins           = list(np.zeros(abs(nbin)))
    continues      = 0
    for i in range(len(bins)):
        bins[i]=Array()
    if shuffle:
        r = zip(CATALOG[args.c.split()[0]],CATALOG[args.c.split()[1]])
        r = np.array(r)
        random.shuffle(r)
        CATALOG[args.c.split()[0]] = r[:,0]
        CATALOG[args.c.split()[1]] = r[:,1]    
    for i in range(len(CATALOG[CATALOG.keys()[0]])):      
        x         = CATALOG[args.c.split()[0]][i]
        y         = CATALOG[args.c.split()[1]][i]
        dx        = x-xc
        dy        = y-yc
        r2        = dx*dx+dy*dy
        r         = np.sqrt(r2)
        if r>=rmax or r<rmin:
            continue
            continues += 1
        if nbin>0:
            indx = int((r-rmin)/binsize)
        else:
            indx = int(np.log(r/rmin)/np.log(rmax/rmin)*abs(nbin))
        if len((args.c).split()) ==4:
            ex   = CATALOG[args.c.split()[2]][i]
            ey   = CATALOG[args.c.split()[3]][i]
        elif len((args.c).split()) ==5:
            ixx      = CATALOG[args.c.split()[2]][i]
            iyy      = CATALOG[args.c.split()[3]][i]
            ixy      = CATALOG[args.c.split()[4]][i]
        elif len((args.c).split()) ==6:
            ex           = CATALOG[args.c.split()[2]][i]
            ey           = CATALOG[args.c.split()[3]][i]
            sigma_e      = CATALOG[args.c.split()[4]][i]
            mean_rho_4th = CATALOG[args.c.split()[3]][i]
        elif len((args.c).split()) ==7:
            ixx          = CATALOG[args.c.split()[2]][i]
            iyy          = CATALOG[args.c.split()[3]][i]
            ixy          = CATALOG[args.c.split()[4]][i]
            sigma_e      = CATALOG[args.c.split()[5]][i]
            mean_rho_4th = CATALOG[args.c.split()[6]][i]
        if len((args.c).split()) ==5 or len((args.c).split()) ==7:
            m   = ixx+iyy
            if m<= 0:
                continue
                continues +=1
            ex      = (ixx-iyy)/m
            ey      = 2.0*ixy/m
        e=eobs      = np.sqrt(ex*ex+ey*ey)
        if e<minell or e>maxell:
            continue
            continues +=1
        cos2phi = (dy*dy-dx*dx)/r2
        sin2phi = -2.0*dx*dy/r2
        e1      = ex*cos2phi + ey*sin2phi
        e2      = ex*sin2phi - ey*cos2phi
        psfsize = psfsize
        if psfsize<0.0:
            sys.stderr.write('negative PSF at: x='+str(x)+', y='+str(y)+'. Exiting')
            exit()
        elif psfsize>0.0:
            R = psfsize/(ixx+iyy)/(4.0/mean_rho_4th - 1.0)
            seeing_correction = 1.0/(1.0 - R)
            if sigma_e<0.0:
                continue
                continues += 1 
            e       *= seeing_correction
            e1      *= seeing_correction
            e2      *= seeing_correction
            sigma_e *= seeing_correction
            f        = SIGMA_SHAPE_SQ/(SIGMA_SHAPE_SQ + sigma_e*sigma_e)
            wt       = f/SIGMA_SHAPE_SQ*(1.0 - f*f*eobs*eobs + SIGMA_SHAPE_SQ*(4.0*f-1.0))
            if dflag:
                print('%-15f%-15f%-15f%-15f%-15f%-15f%-15f%' %(x,y,psfsize,R,seeing_correction,f,wt))
            if R<0 or R>Rmax:
                continue
                continues += 1
            if wt<0.0:
                continue
                continues += 1
        if np.isnan(e) or e<0.0 or e>1.0:   #EDITED: add isnan check
            continue
            continues += 1
        if Dflag:
            print('%-15f%-15f%-15f%-15f%-15f ' %(x,y,r,e1,e2))
        bins[indx].sigma1    += e1/r2
        bins[indx].sigma1_sq += e1*e1/(r2*r2)
        bins[indx].sigma2    += e2/r2
        bins[indx].e1        += e1
        bins[indx].e2        += e2
        bins[indx].e1sq      += e1*e1
        bins[indx].e2sq      += e2*e2
        bins[indx].n         += 1.0
        bins[indx].e         += e
        bins[indx].rad       += r
        if len((args.c).split()) ==6 or len((args.c).split()) ==7:
            bins[indx].wt     += wt
            bins[indx].e1_wtd += e1*wt
            bins[indx].e2_wtd += e2*wt
            bins[indx].r1     += seeing_correction
            bins[indx].var1   += e1*e1*wt*wt
            bins[indx].r2     += (1-f*e2*e2)*wt
            bins[indx].var2   += e2*e2*wt*wt

    if verbose:
        sys.stderr.write('Read '+str(i)+' objects \n')

    if len((args.c).split()) ==6 or len((args.c).split()) ==7:
        sys.stderr.write('Weighted moments are not supported (yet).')
        exit()

    sigma1 = sigma2 = sigma1_sq  = 0.0
    ntot   = int(0)
    OUT=np.zeros((nbin,16))
    for i in reversed(range(0,nbin)):
        if nbin>0:
            r1 = rmin+binsize*(i)
            r2 = rmin+binsize*(i+1);
        else :
            r1 = pow(rmax/rmin,(i)/abs(nbin))*rmin
            r2 = pow(rmax/rmin,(i-1)/abs(nbin))*rmin
        if bins[i].n == 0:
            OUT[i] = np.zeros(16)
            OUT[i,0] = (r1+r2)/2
            OUT[i,1] = ntot        = ntot+float(bins[i].n)
            continue
        OUT[i,0] = r           = float(1.0*bins[i].rad / bins[i].n)
        OUT[i,1] = ntot        = ntot+float(bins[i].n)
        sigma1                += float(bins[i].sigma1)
        sigma1_sq             += float(bins[i].sigma1_sq)
        if ntot == 1:
            sigma1_err             = np.sqrt(sigma1_sq - sigma1*sigma1/ntot)
        else:
            sigma1_err             = np.sqrt((sigma1_sq - sigma1*sigma1/ntot)/(ntot -1))
        OUT[i,2] = a           = sigma1 * rmax * rmax / ntot 
        OUT[i,3] = aerr        = sigma1_err * rmax * rmax / np.sqrt(ntot) 
        OUT[i,4] = n           = bins[i].n
        if n == 1:
            OUT[i,7] = sherr1      = np.sqrt(bins[i].e1sq - bins[i].e1*bins[i].e1/n)
            OUT[i,8] = sherr2      = np.sqrt(bins[i].e2sq - bins[i].e2*bins[i].e2/n)
        else:
            OUT[i,7] = sherr1      = np.sqrt((bins[i].e1sq - bins[i].e1*bins[i].e1/n)/(n-1.0)/(n-1.0))
            OUT[i,8] = sherr2      = np.sqrt((bins[i].e2sq - bins[i].e2*bins[i].e2/n)/(n-1.0)/(n-1.0))
        OUT[i,5] = bins[i].e1  = bins[i].e1/n
        OUT[i,6] = bins[i].e2  = bins[i].e2/n
        OUT[i,9] = bins[i].e   = bins[i].e/n


    true_M200 = get_true_M200(file_name)          # in Msun

    features = [ OUT[i][j] for j in (2,4,5,6) for i in range(0,nbin) ]
    # Features included are: a, n, e1, e2
    # Note that the list is transposed and flattened here.
    # The list is now something like [ a0, a1, ..., a9, n0, n1, ..., n9, etc. ]  

    features.append(true_M200)
    return features


# START
#----------------------------------------------------------------------------------------------------

args_0         = initial()            
file_name      = args_0.catalog[0]
args           = initial(read_catalog(file_name,False)[1])  
nbin           = int(args.n)
header = []
features_name = [ "a", "n", "e1", "e2" ]
for item in features_name:
    header += [ item + "_" + str(i) for i in range(0,nbin) ]
header.append("M200")

targetdir = os.path.dirname(file_name)
print("Reading Catalogs on: "+str(targetdir))
catlist = [ cat for cat in os.listdir(targetdir) if ".cat" in cat ] 
targetfile = os.path.join(targetdir,"data.csv")
f = open(targetfile, 'w')
writer = csv.writer(f)
writer.writerow(header)
for cat in catlist:
    targetcat = os.path.join(targetdir,cat)
    writer.writerow(mainloop(targetcat))
f.close()

