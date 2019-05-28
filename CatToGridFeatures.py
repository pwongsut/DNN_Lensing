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
        self.ex , self.ey                                   = 0,0


def gridloop(file_name,dimension):
    '''
    For now using xmax = ymax = 2*rmax, and grid dimension = (nbin,nbin)
    i.e. only works on square data.
    '''
    args_0         = initial()
    CATALOG        = read_catalog(file_name,False,True)[0]      #-------|
    args           = initial(read_catalog(file_name,False)[1])  
    xc             = args.xcenter[0]
    yc             = args.ycenter[0]
    rmin           = float(args.s)
    rmax           = float(args.e)
    xmax    = 2*rmax
    ymax    = 2*rmax
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
    binsize        = [ xmax/dimension[0], ymax/dimension[1] ]
    Rmax           = 0.8
    sigma_e        = 0.0
    mean_rho_4th   = 0.0
    SIGMA_SHAPE_SQ = 0.10
    continues      = 0
    bins           = [ [ Array() for j in range(dimension[1]) ] for i in range(dimension[0]) ]
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
        indx = [ int(x/binsize[0]), int(y/binsize[1]) ]
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
        if np.isnan(e) or e<0.0 or e>1.0:   #EDITED: add isnan check
            continue
            continues += 1
        cos2phi = (dy*dy-dx*dx)/r2
        sin2phi = -2.0*dx*dy/r2
        e1      = ex*cos2phi + ey*sin2phi
        e2      = ex*sin2phi - ey*cos2phi 
        bins[indx[0]][indx[1]].n    += 1.0
        bins[indx[0]][indx[1]].ex   += ex
        bins[indx[0]][indx[1]].ey   += ey
        bins[indx[0]][indx[1]].e1   += e1
    OUT=np.zeros((dimension[0],dimension[1],4))
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            OUT[i][j][0]    = n     = bins[i][j].n
            if n == 0:
                continue
            OUT[i][j][1]            = bins[i][j].ex/n
            OUT[i][j][2]            = bins[i][j].ey/n
            OUT[i][j][3]            = bins[i][j].e1/n
    true_M200 = get_true_M200(file_name)          # in Msun

    features = [ OUT[i][j][k] for k in range(4) for i in range(dimension[0]) for j in range(dimension[1]) ]
    # Features included are: n, ex, ey, e1
    # Note that the list is flattened here.
    # The list is now something like [ a0, a1, ..., a9, n0, n1, ..., n9, etc. ]  

    features.append(true_M200)
    return features


# START
#----------------------------------------------------------------------------------------------------

args_0         = initial()            
file_name      = args_0.catalog[0]
args           = initial(read_catalog(file_name,False)[1])  
nbin           = int(args.n)
dimension      = [nbin,nbin]
header = []
features_name = [ "n" , "ex" , "ey", "e1" ]
for item in features_name:
    header += [ item + "_" + str(i) + "_" + str(j) for i in range(dimension[0]) for j in range(dimension[1]) ]
header.append("M200")

targetdir = os.path.dirname(file_name)
print("Reading Catalogs on: "+str(targetdir))
catlist = [ cat for cat in os.listdir(targetdir) if ".cat" in cat ] 
targetfile = os.path.join(targetdir,"griddata.csv")
f = open(targetfile, 'w')
writer = csv.writer(f)
writer.writerow(header)
for cat in catlist:
    targetcat = os.path.join(targetdir,cat)
    writer.writerow(gridloop(targetcat,dimension))
f.close()

